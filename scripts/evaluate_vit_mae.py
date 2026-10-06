"""
Evaluate a trained ViT-MAE checkpoint on the development synthetic benchmark.

Reproduces the anomaly-scoring methodology from
notebooks/09_vit_mae_anomaly_scoring.ipynb (cell 3b43c245 — "FINAL selected
validation scores"):

  scorer       : masked_topk_scores
  top_fraction : 0.002
  masks        : 1 deterministic mask per batch (seed = base_seed + batch_start)
  batch_size   : 16
  base_seed    : 42
  benchmark    : data/synthetic/validation/  (development benchmark)

Run from the project root:

    python scripts/evaluate_vit_mae.py \\
        --config configs/experiments/vit_mae_v2_highres_medium.yaml \\
        --checkpoint results/checkpoints/vit_mae_v2_highres_medium/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from deepseti.models.vit_mae import ViTMAE, patchify
from deepseti.utils.device import detect_device


# ---------------------------------------------------------------------------
# Mask helpers — ported from notebook 09, parameterized from config
# ---------------------------------------------------------------------------

def make_shared_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic shared mask for a clean/injected pair.

    Replicates notebook cell d854ae69, but num_patches and mask_ratio are
    taken from the model config rather than hardcoded to 256 / 0.75.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    noise = torch.rand(batch_size, num_patches, generator=generator)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    num_visible = int(num_patches * (1.0 - mask_ratio))
    ids_keep = ids_shuffle[:, :num_visible]

    mask = torch.ones(batch_size, num_patches)
    mask[:, :num_visible] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return ids_keep, ids_restore, mask


def forward_with_fixed_mask(
    model: ViTMAE,
    x: torch.Tensor,
    ids_keep: torch.Tensor,
    ids_restore: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run encoder + decoder with a pre-computed mask, bypassing random_masking.

    Replicates notebook cell d854ae69, but uses model.patch_size rather than
    the hardcoded (4, 64) that was correct only for the V1 architecture.
    """
    target_patches = patchify(x, patch_size=model.patch_size)

    tokens = model.patch_embedding(x)
    tokens = model.position_embedding(tokens)
    embed_dim = tokens.shape[-1]

    ids_keep = ids_keep.to(x.device)
    ids_restore = ids_restore.to(x.device)
    mask = mask.to(x.device)

    visible_tokens = torch.gather(
        tokens,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim),
    )

    encoded_tokens = model.encoder(visible_tokens)
    predicted_patches = model.decoder(encoded_tokens, ids_restore)

    return predicted_patches, target_patches, mask


def masked_topk_scores(
    predicted_patches: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
    top_fraction: float,
) -> torch.Tensor:
    """Mean of the top-k squared errors among masked patches only.

    k is computed relative to the full frame pixel count (num_patches *
    patch_dim), matching notebook cell 7c36f2a4 exactly.
    """
    pixel_error = (predicted_patches - target_patches) ** 2

    batch, num_patches, patch_dim = pixel_error.shape
    full_frame_pixels = num_patches * patch_dim
    k = max(1, int(full_frame_pixels * top_fraction))

    scores = []
    for i in range(batch):
        masked_patch_error = pixel_error[i, mask[i].bool()]
        masked_pixel_error = masked_patch_error.reshape(-1)
        top_values = torch.topk(masked_pixel_error, k=k).values
        scores.append(top_values.mean())

    return torch.stack(scores)


def score_paired_dataset_topk(
    model: ViTMAE,
    clean_frames: np.ndarray,
    injected_frames: np.ndarray,
    num_patches: int,
    mask_ratio: float,
    top_fraction: float = 0.002,
    batch_size: int = 16,
    base_seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray]:
    """Score every paired (clean, injected) frame using the final V1 method.

    Replicates notebook cells 83731676 + 3b43c245 exactly, with num_patches
    and mask_ratio derived from config to support any architecture.

    Seed scheme: seed = base_seed + batch_start_index, giving a different
    deterministic mask for each batch while keeping clean/injected masks
    identical within each pair.
    """
    clean_all: list[np.ndarray] = []
    injected_all: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(clean_frames), batch_size):
            stop = min(start + batch_size, len(clean_frames))
            current_batch_size = stop - start

            clean_batch = (
                torch.tensor(clean_frames[start:stop], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            injected_batch = (
                torch.tensor(injected_frames[start:stop], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            ids_keep, ids_restore, shared_mask = make_shared_mask(
                batch_size=current_batch_size,
                num_patches=num_patches,
                mask_ratio=mask_ratio,
                seed=base_seed + start,
            )

            clean_pred, clean_target, clean_mask = forward_with_fixed_mask(
                model, clean_batch, ids_keep, ids_restore, shared_mask,
            )
            injected_pred, injected_target, injected_mask = forward_with_fixed_mask(
                model, injected_batch, ids_keep, ids_restore, shared_mask,
            )

            clean_score = masked_topk_scores(
                clean_pred, clean_target, clean_mask, top_fraction,
            )
            injected_score = masked_topk_scores(
                injected_pred, injected_target, injected_mask, top_fraction,
            )

            clean_all.append(clean_score.cpu().numpy())
            injected_all.append(injected_score.cpu().numpy())

    return np.concatenate(clean_all), np.concatenate(injected_all)


# ---------------------------------------------------------------------------
# Metric helpers — replicates notebook cell 1c6dc0aa
# ---------------------------------------------------------------------------

def compute_metrics(
    clean_scores: np.ndarray,
    injected_scores: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(injected_scores)),
    ])
    scores = np.concatenate([clean_scores, injected_scores])

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "paired_win_rate": float(np.mean(injected_scores > clean_scores)),
        "n": int(len(clean_scores)),
    }


def subset_metrics(
    indices: np.ndarray,
    clean_scores: np.ndarray,
    injected_scores: np.ndarray,
) -> dict[str, Any]:
    return compute_metrics(clean_scores[indices], injected_scores[indices])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Scoring hyperparameters are frozen from notebook 09 — do not tune on V2.
_TOP_FRACTION: float = 0.002
_BATCH_SIZE: int = 16
_BASE_SEED: int = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ViT-MAE on the development benchmark (validation split)."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="PATH",
        help="Experiment YAML config (e.g. configs/experiments/vit_mae_v2_highres_medium.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="PATH",
        help="Checkpoint file (e.g. results/checkpoints/vit_mae_v2_highres_medium/best.pt)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # --- Config ---
    with args.config.open() as f:
        config = yaml.safe_load(f)

    experiment_name: str = config["experiment"]["name"]
    model_cfg: dict = config["model"]

    # Guard: refuse to overwrite an existing results file.
    output_path = project_root / "results" / "metrics" / f"{experiment_name}.json"
    if output_path.exists():
        raise FileExistsError(
            f"Results file already exists: {output_path}\n"
            "Remove it manually if you intentionally want to overwrite."
        )

    # --- Device ---
    device = detect_device()
    print(f"Device           : {device}")

    # --- Build model from config ---
    model = ViTMAE(
        frame_shape=tuple(model_cfg["frame_shape"]),
        patch_size=tuple(model_cfg["patch_size"]),
        channels=model_cfg.get("channels", 1),
        embed_dim=model_cfg["embed_dim"],
        encoder_depth=model_cfg["encoder_depth"],
        encoder_heads=model_cfg["encoder_heads"],
        encoder_mlp_dim=model_cfg["encoder_mlp_dim"],
        decoder_dim=model_cfg["decoder_dim"],
        decoder_depth=model_cfg["decoder_depth"],
        decoder_heads=model_cfg["decoder_heads"],
        decoder_mlp_dim=model_cfg["decoder_mlp_dim"],
        mask_ratio=model_cfg["mask_ratio"],
    ).to(device)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    checkpoint_epoch = ckpt.get("epoch", "unknown")
    best_val_loss = ckpt.get("best_val_loss", "unknown")

    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Epoch            : {checkpoint_epoch}")
    print(f"Best val loss    : {best_val_loss}")
    print(f"Parameters       : {sum(p.numel() for p in model.parameters()):,}")

    # --- Derive patch geometry from config ---
    frame_shape = tuple(model_cfg["frame_shape"])
    patch_size = tuple(model_cfg["patch_size"])
    mask_ratio: float = model_cfg["mask_ratio"]

    num_time_patches = frame_shape[0] // patch_size[0]
    num_freq_patches = frame_shape[1] // patch_size[1]
    num_patches = num_time_patches * num_freq_patches

    print(f"Patch size       : {patch_size}")
    print(f"Patch grid       : {num_time_patches} × {num_freq_patches} = {num_patches} patches")
    print(f"Mask ratio       : {mask_ratio}  ({int(num_patches * mask_ratio)} masked / {num_patches} total)")

    # --- Load development benchmark (validation split only) ---
    bench_dir = project_root / "data" / "synthetic" / "validation"
    print(f"\nBenchmark        : {bench_dir}  [DEVELOPMENT — not the final paper test]")

    clean_frames: np.ndarray = np.load(bench_dir / "clean_frames.npy")
    injected_frames: np.ndarray = np.load(bench_dir / "injected_frames.npy")
    with open(bench_dir / "metadata.json") as f:
        metadata: list[dict] = json.load(f)

    print(f"Clean frames     : {clean_frames.shape}")
    print(f"Injected frames  : {injected_frames.shape}")
    print(f"Pairs            : {len(metadata)}")

    # --- Score ---
    print(
        f"\nScoring  (scorer=masked_topk, top_fraction={_TOP_FRACTION}, "
        f"batch_size={_BATCH_SIZE}, base_seed={_BASE_SEED}) ..."
    )

    clean_scores, injected_scores = score_paired_dataset_topk(
        model=model,
        clean_frames=clean_frames,
        injected_frames=injected_frames,
        num_patches=num_patches,
        mask_ratio=mask_ratio,
        top_fraction=_TOP_FRACTION,
        batch_size=_BATCH_SIZE,
        base_seed=_BASE_SEED,
        device=device,
    )

    # --- Overall metrics ---
    overall = compute_metrics(clean_scores, injected_scores)

    print(f"\n{'='*60}")
    print(f"  {experiment_name}")
    print(f"{'='*60}")
    print(f"  AUROC            : {overall['auroc']:.6f}")
    print(f"  AUPRC            : {overall['auprc']:.6f}")
    print(f"  Paired win rate  : {overall['paired_win_rate']:.3%}")
    print(f"  N (pairs)        : {overall['n']}")

    # --- Per-SNR breakdown ---
    snr_values = sorted({item["robust_snr"] for item in metadata})
    by_snr: dict[str, Any] = {}

    print(f"\nBY SNR")
    for snr in snr_values:
        indices = np.array([
            i for i, item in enumerate(metadata) if item["robust_snr"] == snr
        ])
        result = subset_metrics(indices, clean_scores, injected_scores)
        by_snr[str(snr)] = result
        print(
            f"  SNR {snr:>4} | n={result['n']:3d} | "
            f"AUROC {result['auroc']:.3f} | "
            f"AUPRC {result['auprc']:.3f} | "
            f"paired {result['paired_win_rate']:.1%}"
        )

    # --- Per-morphology breakdown ---
    morphologies = sorted({item["morphology"] for item in metadata})
    by_morphology: dict[str, Any] = {}

    print(f"\nBY MORPHOLOGY")
    for morphology in morphologies:
        indices = np.array([
            i for i, item in enumerate(metadata) if item["morphology"] == morphology
        ])
        result = subset_metrics(indices, clean_scores, injected_scores)
        by_morphology[morphology] = result
        print(
            f"  {morphology:22s} | n={result['n']:3d} | "
            f"AUROC {result['auroc']:.3f} | "
            f"AUPRC {result['auprc']:.3f} | "
            f"paired {result['paired_win_rate']:.1%}"
        )

    # --- Save results ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "best_val_loss": (
            float(best_val_loss)
            if isinstance(best_val_loss, (int, float))
            else str(best_val_loss)
        ),
        "benchmark": "development (data/synthetic/validation)",
        "scoring": {
            "method": "masked_top_error",
            "top_fraction": _TOP_FRACTION,
            "num_inference_masks": 1,
            "batch_size": _BATCH_SIZE,
            "base_seed": _BASE_SEED,
            "mask_ratio": mask_ratio,
            "num_patches": num_patches,
            "patch_size": list(patch_size),
        },
        "overall": overall,
        "by_snr": by_snr,
        "by_morphology": by_morphology,
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to : {output_path}")


if __name__ == "__main__":
    main()
