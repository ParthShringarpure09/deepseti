"""One-command Fine V2 ConvAE main experiment runner.

    .venv\\Scripts\\python.exe scripts\\run_fine_v2_convae_experiment.py

drives the entire experiment end to end: data preflight, device checks,
ConvAE construction, training on the frozen TRAIN split (normal only),
best-checkpoint selection by CLEAN VALIDATION reconstruction loss, frozen
synthetic VALIDATION benchmark evaluation, metrics, figures and a result
summary. No separate train/evaluate scripts are required.

Frozen scientific state (must not change here):
  * TRAIN: 15,120 real normal Fine V2 frames, synthetic signals never enter
    training (see ``deepseti.evaluation.convae_preflight``).
  * Checkpoint selection uses CLEAN VALIDATION reconstruction loss only;
    the synthetic benchmark is scored only AFTER the best epoch is fixed.
  * Synthetic VALIDATION benchmark: frozen, 1,200 paired examples
    (``deepseti.evaluation.fine_benchmark`` / commit 876dac4). Never
    regenerated here.
  * Anomaly scores: whole-frame mean squared residual, and the mean of the
    top 0.2% largest-residual pixels (``top_fraction = 0.002``, frozen;
    see ``deepseti.evaluation.pca_scoring.TOP_FRACTION``).
  * ``data/processed/fine_v2/test`` (final test) is never opened.

Seed: 20261013 (``experiment.seed`` in
``configs/experiments/fine_v2_convae.yaml``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_fine_v2_pca_baseline as pca_baseline
from extract_fine_v2 import (
    validate_split_access,
)

from deepseti.evaluation import convae_preflight as preflight
from deepseti.evaluation import pca_scoring as scoring
from deepseti.models.conv_autoencoder import ConvAutoencoderV3
from deepseti.training.convae_dataset import FrameArrayDataset
from deepseti.training.convae_epoch import (
    evaluate_convae_epoch,
    train_convae_epoch,
)
from deepseti.utils.device import detect_device

FRAME_SHAPE = (16, 4096)
DRY_RUN_TRAIN_FRAMES = 32
DRY_RUN_VALIDATION_FRAMES = 16
DRY_RUN_BENCHMARK_EXAMPLES = 8
DRY_RUN_EPOCHS = 1


# ---------------------------------------------------------------------------
# small shared infra (deliberately duplicated from scripts/train_ssl_cnn.py;
# there is no shared "script infra" module in this repository)
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/fine_v2_convae.yaml"),
        help="Path to the ConvAE experiment YAML configuration.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the full pipeline on a tiny sample for one epoch and exit; "
             "never performs full training.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run preflight and environment checks only; no training.",
    )
    parser.add_argument(
        "--allow-non-cuda", action="store_true",
        help="Permit running without CUDA. Required on any machine without "
             "a CUDA GPU, including for --dry-run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional override for results_dir (keeps dry-run/smoke-test "
             "output separate from the frozen experiment results).",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"expected YAML configuration dict, got {type(config)}")
    return config


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def guard_final_test_untouched(data_cfg: dict) -> dict:
    """Refuse to continue if the final test split could be, or has been, opened."""

    final_test_dir = resolve(data_cfg["final_test_dir"])
    if final_test_dir.exists():
        raise preflight.PreflightError(
            f"REFUSED: {final_test_dir} exists; final test must remain untouched")

    access_cfg_path = resolve(data_cfg["validation_access_config"])
    access_cfg = yaml.safe_load(access_cfg_path.read_text())
    refusal = validate_split_access("test", access_cfg)
    if refusal is None:
        raise preflight.PreflightError(
            "REFUSED: split 'test' must remain refused by the extraction access "
            "config; refusing to proceed")

    return {"final_test_dir_absent": True, "test_split_refused": True, "reason": refusal}


def run_data_preflight(data_cfg: dict) -> dict:
    """Verify TRAIN, clean VALIDATION and the frozen synthetic benchmark.

    Raises ``convae_preflight.PreflightError`` with a message naming the
    exact missing/mismatched directory on any failure.
    """

    report: dict = {"final_test_guard": guard_final_test_untouched(data_cfg)}

    report["train"] = preflight.verify_split_frames(
        metadata_path=resolve(data_cfg["train_metadata"]),
        root=resolve(data_cfg["train_root"]),
        expected_split="train",
        expected_frames=int(data_cfg["expected_train_frames"]),
        frame_shape=tuple(data_cfg["frame_shape"]),
    )

    report["validation"] = preflight.verify_split_frames(
        metadata_path=resolve(data_cfg["validation_metadata"]),
        root=resolve(data_cfg["validation_root"]),
        expected_split="validation",
        expected_frames=int(data_cfg["expected_validation_frames"]),
        frame_shape=tuple(data_cfg["frame_shape"]),
    )

    report["benchmark"] = preflight.verify_benchmark(
        benchmark_dir=resolve(data_cfg["benchmark_dir"]),
        audit_path=resolve(data_cfg["benchmark_audit"]),
        expected_examples=int(data_cfg["expected_benchmark_examples"]),
    )

    return report


# ---------------------------------------------------------------------------
# environment / --check-only
# ---------------------------------------------------------------------------


def gather_environment_info() -> dict:
    cuda_available = torch.cuda.is_available()
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "gpu_name": None,
        "vram_gb": None,
    }
    if cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    return info


def check_only_report(cfg: dict) -> int:
    env = gather_environment_info()
    print("DeepSeti Fine V2 ConvAE -- environment check")
    print("=" * 50)
    for key, value in env.items():
        print(f"{key:16s}: {value}")
    print()

    try:
        data_report = run_data_preflight(cfg["data"])
    except preflight.PreflightError as exc:
        print("NOT READY:")
        print(f"  {exc}")
        return 1

    print(f"Train frames       : {data_report['train']['n_frames']}")
    print(f"Validation frames  : {data_report['validation']['n_frames']}")
    print(f"Benchmark examples : {data_report['benchmark']['n_examples']}")
    print("Benchmark checksums: match frozen audit")
    print("Final test access  : disabled")
    print()

    if not env["cuda_available"]:
        print("NOT READY:")
        print("  no CUDA device detected; the production run requires CUDA "
              "(pass --allow-non-cuda only for --dry-run testing)")
        return 1

    print("READY FOR TRAINING")
    return 0


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_frames(metadata: pd.DataFrame, root: Path) -> np.ndarray:
    """Load exactly the frames named by ``metadata`` (one scan file read per
    unique ``npy`` value), in ``metadata``'s row order."""

    metadata = metadata.reset_index(drop=True)
    frames = np.empty((len(metadata), *FRAME_SHAPE), dtype=np.float32)
    for npy_rel, group in metadata.groupby("npy", sort=False):
        arr = np.load(root / npy_rel)
        frames[group.index.to_numpy()] = arr[group["npy_index"].to_numpy()]
    return frames


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_model() -> tuple[nn.Module, dict]:
    model = ConvAutoencoderV3()
    n_params = sum(p.numel() for p in model.parameters())
    doc = {
        "architecture": "ConvAutoencoderV3",
        "input_shape": [1, *FRAME_SHAPE],
        "encoder": "4 Conv2d+ReLU blocks; first stride (1, 2) (frequency-only "
                   "reduction), remaining three stride (2, 2)",
        "latent_shape": [128, 2, 256],
        "decoder": "3 ConvTranspose2d+ReLU blocks stride (2, 2), then one "
                   "ConvTranspose2d kernel (3, 4) stride (1, 2) restoring "
                   "frequency without changing the time dimension",
        "reconstructed_output_shape": [1, *FRAME_SHAPE],
        "parameter_count": int(n_params),
    }
    return model, doc


def sanity_check_batch_size(
    model: nn.Module, device: torch.device, batch_size: int, min_batch_size: int,
    use_amp: bool = False,
) -> int:
    """Engineering-only GPU-memory check: halve ``batch_size`` until a forward
    + backward pass fits, or raise once below ``min_batch_size``. A no-op on
    non-CUDA devices. Never uses synthetic-benchmark performance."""

    if device.type != "cuda":
        return batch_size

    bs = batch_size
    while bs >= min_batch_size:
        try:
            dummy = torch.randn(bs, 1, *FRAME_SHAPE, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(dummy)
                loss = nn.functional.mse_loss(out, dummy)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del dummy, out, loss
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = bs // 2
    raise RuntimeError(
        f"GPU out of memory even at the minimum batch size ({min_batch_size})")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def run_training(
    model: nn.Module,
    train_frames: np.ndarray,
    validation_frames: np.ndarray,
    device: torch.device,
    training_cfg: dict,
    batch_size: int,
    epochs: int,
    seed: int,
    checkpoint_path: Path,
) -> dict:
    """Train on TRAIN (normal only); select the best epoch by CLEAN
    VALIDATION reconstruction loss only. Saves the best checkpoint and
    returns the training record (history, best epoch, timing, settings)."""

    train_dataset = FrameArrayDataset(train_frames)
    validation_dataset = FrameArrayDataset(validation_frames)

    num_workers = int(training_cfg["num_workers"]) if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_generator,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=max(batch_size, 1), shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    use_amp = bool(training_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp) if device.type == "cuda" else None
    grad_clip_norm = float(training_cfg["grad_clip_norm"])
    patience = int(training_cfg["early_stopping_patience"])

    history: dict[str, list] = {"epoch": [], "train_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_convae_epoch(
            model, train_loader, optimizer, device,
            grad_clip_norm=grad_clip_norm, scaler=scaler, use_amp=use_amp,
        )
        validation_metrics = evaluate_convae_epoch(model, validation_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["validation_loss"].append(validation_metrics["loss"])

        print(f"Epoch {epoch:03d}/{epochs:03d} | "
              f"train_loss={train_metrics['loss']:.6f} | "
              f"clean_val_loss={validation_metrics['loss']:.6f}")

        is_best = validation_metrics["loss"] < best_validation_loss
        if is_best:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_validation_loss": best_validation_loss,
                "seed": seed,
            })
            print(f"  -> saved new best checkpoint (clean_val_loss={best_validation_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> early stopping (no improvement for {patience} epochs)")
                break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_duration_seconds": time.time() - start,
        "batch_size": batch_size,
        "epochs_run": history["epoch"][-1] if history["epoch"] else 0,
        "learning_rate": float(training_cfg["learning_rate"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "optimizer": "AdamW",
        "loss": "MSE",
        "amp": use_amp,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# benchmark scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_frames(model: nn.Module, device: torch.device, frames: np.ndarray,
                 top_fraction: float, batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    reconstructed = np.empty_like(frames)
    for start in range(0, len(frames), batch_size):
        chunk = torch.from_numpy(frames[start:start + batch_size]).unsqueeze(1).to(device)
        out = model(chunk).squeeze(1).cpu().numpy()
        reconstructed[start:start + batch_size] = out
    residual_sq = scoring.residual_squared(frames, reconstructed)
    return {
        "mean": scoring.mean_score(residual_sq),
        "top_fraction": scoring.top_fraction_score(residual_sq, top_fraction),
        "reconstructed": reconstructed,
    }


def evaluate_benchmark(model: nn.Module, device: torch.device, clean_frames: np.ndarray,
                       injected_frames: np.ndarray, metadata: pd.DataFrame,
                       top_fraction: float, seed: int, n_boot: int) -> dict:
    clean = score_frames(model, device, clean_frames, top_fraction)
    injected = score_frames(model, device, injected_frames, top_fraction)
    groups = metadata["group_id"].to_numpy()

    summary = {}
    for key, label in (("mean", "mean_score"), ("top_fraction", "top_fraction_score")):
        auc = scoring.auroc_auprc(clean[key], injected[key])
        ci = scoring.grouped_bootstrap_ci(
            clean[key], injected[key], groups,
            metric_fn=lambda c, i: scoring.auroc_auprc(c, i)["auroc"],
            n_boot=n_boot, seed=seed,
        )
        summary[label] = {
            **auc,
            "auroc_grouped_bootstrap_ci95": ci,
            "paired_success_fraction": scoring.paired_success_fraction(clean[key], injected[key]),
            "score_delta_mean": float(np.mean(injected[key] - clean[key])),
            "score_delta_median": float(np.median(injected[key] - clean[key])),
        }

    by_snr = {key: pca_baseline.breakdown_by(metadata, clean[key], injected[key], "requested_snr")
             for key in ("mean", "top_fraction")}
    by_morphology = {key: pca_baseline.breakdown_by(metadata, clean[key], injected[key], "morphology")
                    for key in ("mean", "top_fraction")}
    drift_meta = metadata[metadata["morphology"] == "linear_drift"]
    drift_idx = drift_meta.index.to_numpy()
    by_drift = {key: pca_baseline.breakdown_by(
                    drift_meta, clean[key][drift_idx], injected[key][drift_idx],
                    "drift_hz_s", transform=abs)
               for key in ("mean", "top_fraction")}

    return {
        "clean": clean, "injected": injected, "summary": summary,
        "by_requested_snr": by_snr, "by_morphology": by_morphology,
        "by_abs_drift_hz_s": by_drift,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(training_record: dict, benchmark: dict, metadata: pd.DataFrame,
                 clean_frames: np.ndarray, injected_frames: np.ndarray,
                 pca_cfg: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    history = training_record["history"]
    clean, injected, summary = benchmark["clean"], benchmark["injected"], benchmark["summary"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train (TRAIN, normal only)")
    ax.plot(history["epoch"], history["validation_loss"], label="clean validation")
    ax.axvline(training_record["best_epoch"], color="gray", linestyle="--",
              label=f"best epoch ({training_record['best_epoch']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Fine V2 ConvAE: training / clean-validation loss")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "training_history.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("mean", "top_fraction"),
                              ("whole-frame mean score", "top-0.2% local score"), strict=True):
        ax.hist(clean[key], bins=40, alpha=0.6, label="clean", density=True)
        ax.hist(injected[key], bins=40, alpha=0.6, label="injected", density=True)
        ax.set_title(f"Fine V2 validation development benchmark: {title}")
        ax.legend()
    fig.tight_layout()
    p = out_dir / "score_distributions.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["mean score", "top-0.2% score"]
    pca_auroc = [pca_cfg["mean_auroc"], pca_cfg["local_auroc"]]
    convae_auroc = [summary["mean_score"]["auroc"], summary["top_fraction_score"]["auroc"]]
    x = np.arange(2)
    ax.bar(x - 0.15, pca_auroc, width=0.3, label="PCA (frozen)")
    ax.bar(x + 0.15, convae_auroc, width=0.3, label="ConvAE")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Fine V2 validation development benchmark: PCA vs ConvAE")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "pca_vs_convae.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    by_snr = benchmark["by_requested_snr"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    snr_levels = sorted(float(k) for k in by_snr["mean"])
    ax.plot(snr_levels, [by_snr["mean"][str(k)]["auroc"] for k in snr_levels], "o-", label="mean score")
    ax.plot(snr_levels, [by_snr["top_fraction"][str(k)]["auroc"] for k in snr_levels], "s-",
           label="top-0.2% score")
    ax.set_xscale("log")
    ax.set_xlabel("requested SNR")
    ax.set_ylabel("AUROC")
    ax.set_title("Fine V2 ConvAE: AUROC vs SNR")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "auroc_vs_snr.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    by_morphology = benchmark["by_morphology"]
    fig, ax = plt.subplots(figsize=(7, 4))
    morphs = list(by_morphology["mean"])
    xm = np.arange(len(morphs))
    ax.bar(xm - 0.15, [by_morphology["mean"][m]["auroc"] for m in morphs], width=0.3, label="mean score")
    ax.bar(xm + 0.15, [by_morphology["top_fraction"][m]["auroc"] for m in morphs], width=0.3,
          label="top-0.2% score")
    ax.set_xticks(xm, morphs, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Fine V2 ConvAE: AUROC by morphology")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "auroc_by_morphology.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    by_drift = benchmark["by_abs_drift_hz_s"]
    if by_drift["mean"]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        drift_levels = sorted(float(k) for k in by_drift["mean"])
        ax.plot(drift_levels, [by_drift["mean"][str(k)]["auroc"] for k in drift_levels], "o-",
               label="mean score")
        ax.plot(drift_levels, [by_drift["top_fraction"][str(k)]["auroc"] for k in drift_levels], "s-",
               label="top-0.2% score")
        ax.set_xscale("log")
        ax.set_xlabel("|drift rate| (Hz/s)")
        ax.set_ylabel("AUROC")
        ax.set_title("Fine V2 ConvAE: linear-drift AUROC vs drift rate")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "auroc_vs_drift_rate.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))

    idx = 0
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    axes[0].imshow(clean_frames[idx], aspect="auto", cmap="viridis")
    axes[0].set_title("clean")
    axes[1].imshow(injected_frames[idx], aspect="auto", cmap="viridis")
    axes[1].set_title("injected")
    residual = (injected_frames[idx] - injected["reconstructed"][idx]) ** 2
    axes[2].imshow(residual, aspect="auto", cmap="inferno")
    axes[2].set_title("ConvAE residual (injected)")
    fig.suptitle("Fine V2 validation development benchmark: representative reconstruction")
    fig.tight_layout()
    p = out_dir / "reconstruction_example.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_config(config_path)

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)

    if args.check_only:
        return check_only_report(cfg)

    device = detect_device()
    if device.type != "cuda" and not args.allow_non_cuda:
        print("WARNING: this experiment is intended to run on a CUDA GPU.")
        print(f"  Detected device: {device}")
        print("  Pass --allow-non-cuda to run anyway (dry-run testing only); "
              "refusing to start a full production run without CUDA.")
        return 1

    try:
        data_report = run_data_preflight(cfg["data"])
    except preflight.PreflightError as exc:
        print(f"NOT READY: {exc}")
        return 1

    top_fraction = float(cfg["evaluation"]["top_fraction"])
    if top_fraction != scoring.TOP_FRACTION:
        raise SystemExit(
            f"REFUSED: configured top_fraction={top_fraction} != frozen "
            f"deepseti.evaluation.pca_scoring.TOP_FRACTION={scoring.TOP_FRACTION}")

    model, model_doc = build_model()
    model = model.to(device)

    output_cfg = cfg["output"]
    if args.output_dir is not None:
        results_dir = args.output_dir
    else:
        results_dir = resolve(output_cfg["results_dir"])
        if args.dry_run:
            # Never let a dry run clobber the real experiment's results.
            results_dir = results_dir / "dry_run"
    figures_dir = results_dir / "figures"
    checkpoint_dir = resolve(output_cfg["checkpoint_dir"])
    checkpoint_name = output_cfg["checkpoint_name"]
    if args.dry_run:
        checkpoint_name = f"dry_run_{checkpoint_name}"
    checkpoint_path = checkpoint_dir / checkpoint_name

    print()
    print("DeepSeti Fine V2 ConvAE")
    print(f"Device               : {device}")
    if device.type == "cuda":
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM                 : "
              f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Train frames         : {data_report['train']['n_frames']}")
    print(f"Validation benchmark : {data_report['benchmark']['n_examples']}")
    print(f"Seed                 : {seed}")
    print("Final test access    : disabled")
    print(f"Parameters           : {model_doc['parameter_count']:,}")
    print()

    training_cfg = dict(cfg["training"])
    if args.dry_run:
        train_metadata = pd.read_csv(resolve(cfg["data"]["train_metadata"])).head(DRY_RUN_TRAIN_FRAMES)
        validation_metadata = pd.read_csv(
            resolve(cfg["data"]["validation_metadata"])).head(DRY_RUN_VALIDATION_FRAMES)
        epochs = DRY_RUN_EPOCHS
        configured_batch_size = min(8, DRY_RUN_TRAIN_FRAMES)
        min_batch_size = min(2, configured_batch_size)
    else:
        train_metadata = pd.read_csv(resolve(cfg["data"]["train_metadata"]))
        validation_metadata = pd.read_csv(resolve(cfg["data"]["validation_metadata"]))
        epochs = int(training_cfg["epochs"])
        configured_batch_size = int(training_cfg["batch_size"])
        min_batch_size = int(training_cfg["min_batch_size"])

    print("Loading TRAIN frames (normal only)...")
    train_frames = load_frames(train_metadata, resolve(cfg["data"]["train_root"]))
    print("Loading clean VALIDATION frames (checkpoint selection only)...")
    validation_frames = load_frames(validation_metadata, resolve(cfg["data"]["validation_root"]))

    batch_size = sanity_check_batch_size(
        model, device, configured_batch_size, min_batch_size,
        use_amp=bool(training_cfg["amp"]))
    if batch_size != configured_batch_size:
        print(f"  GPU-memory sanity check reduced batch size "
              f"{configured_batch_size} -> {batch_size}")

    training_record = run_training(
        model, train_frames, validation_frames, device, training_cfg,
        batch_size=batch_size, epochs=epochs, seed=seed, checkpoint_path=checkpoint_path,
    )
    training_record["device"] = str(device)
    training_record["gpu_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    training_record["parameter_count"] = model_doc["parameter_count"]

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    print()
    print("Scoring the frozen synthetic VALIDATION benchmark "
          "(after checkpoint selection)...")
    benchmark_dir = resolve(cfg["data"]["benchmark_dir"])
    if args.dry_run:
        clean_frames = np.load(benchmark_dir / "clean_frames.npy")[:DRY_RUN_BENCHMARK_EXAMPLES]
        injected_frames = np.load(benchmark_dir / "injected_frames.npy")[:DRY_RUN_BENCHMARK_EXAMPLES]
        benchmark_metadata = pd.read_csv(
            benchmark_dir / "metadata.csv").head(DRY_RUN_BENCHMARK_EXAMPLES).reset_index(drop=True)
        n_boot = 20
    else:
        clean_frames = np.load(benchmark_dir / "clean_frames.npy")
        injected_frames = np.load(benchmark_dir / "injected_frames.npy")
        benchmark_metadata = pd.read_csv(benchmark_dir / "metadata.csv")
        n_boot = int(cfg["evaluation"]["bootstrap_resamples"])

    benchmark = evaluate_benchmark(
        model, device, clean_frames, injected_frames, benchmark_metadata,
        top_fraction, seed, n_boot,
    )

    figure_paths = make_figures(
        training_record, benchmark, benchmark_metadata, clean_frames, injected_frames,
        cfg["pca_comparison"], figures_dir,
    )

    pca_cfg = cfg["pca_comparison"]
    comparison_rows = [
        {"score": "mean_score", "metric": "auroc",
         "pca": pca_cfg["mean_auroc"], "convae": benchmark["summary"]["mean_score"]["auroc"]},
        {"score": "mean_score", "metric": "auprc",
         "pca": pca_cfg["mean_auprc"], "convae": benchmark["summary"]["mean_score"]["auprc"]},
        {"score": "top_fraction_score", "metric": "auroc",
         "pca": pca_cfg["local_auroc"], "convae": benchmark["summary"]["top_fraction_score"]["auroc"]},
        {"score": "top_fraction_score", "metric": "auprc",
         "pca": pca_cfg["local_auprc"], "convae": benchmark["summary"]["top_fraction_score"]["auprc"]},
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir / "comparison_with_pca.csv", index=False)
    write_json(results_dir / "comparison_with_pca.json", comparison_rows)

    write_json(results_dir / "training_history.json", training_record["history"])

    metrics = {
        "top_fraction": top_fraction,
        "n_benchmark_examples": len(benchmark_metadata),
        "summary": benchmark["summary"],
        "by_requested_snr": benchmark["by_requested_snr"],
        "by_morphology": benchmark["by_morphology"],
        "by_abs_drift_hz_s": benchmark["by_abs_drift_hz_s"],
        "figure_paths": figure_paths,
    }
    write_json(results_dir / "metrics.json", metrics)

    experiment_summary = {
        "experiment": cfg["experiment"]["name"],
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "dry_run": args.dry_run,
        "seed": seed,
        "model": model_doc,
        "training": {
            "best_epoch": training_record["best_epoch"],
            "best_validation_loss": training_record["best_validation_loss"],
            "epochs_run": training_record["epochs_run"],
            "batch_size": training_record["batch_size"],
            "learning_rate": training_record["learning_rate"],
            "weight_decay": training_record["weight_decay"],
            "optimizer": training_record["optimizer"],
            "loss": training_record["loss"],
            "amp": training_record["amp"],
            "training_duration_seconds": training_record["training_duration_seconds"],
            "gpu_name": training_record["gpu_name"],
        },
        "pca_comparison": pca_cfg,
        "checkpoint": str(checkpoint_path),
        "results_dir": str(results_dir),
        "start_time": utc_now(),
        "data_preflight": {
            "train_frames": data_report["train"]["n_frames"],
            "validation_frames": data_report["validation"]["n_frames"],
            "benchmark_examples": data_report["benchmark"]["n_examples"],
            "final_test_access": "disabled",
        },
    }
    write_json(results_dir / "experiment_summary.json", experiment_summary)

    local_auroc = benchmark["summary"]["top_fraction_score"]["auroc"]
    mean_auroc = benchmark["summary"]["mean_score"]["auroc"]
    print()
    print("=" * 50)
    print("DeepSeti Fine V2 ConvAE "
          f"{'DRY RUN COMPLETE' if args.dry_run else 'COMPLETE'}")
    print("=" * 50)
    print(f"GPU: {training_record['gpu_name']}")
    print(f"Best epoch: {training_record['best_epoch']}")
    print(f"Training time: {training_record['training_duration_seconds']:.1f}s")
    print()
    print(f"Mean AUROC: {mean_auroc:.4f}")
    print(f"Mean AUPRC: {benchmark['summary']['mean_score']['auprc']:.4f}")
    print()
    print(f"Local AUROC: {local_auroc:.4f}")
    print(f"Local AUPRC: {benchmark['summary']['top_fraction_score']['auprc']:.4f}")
    print()
    print(f"PCA local AUROC: {pca_cfg['local_auroc']}")
    print(f"ConvAE local AUROC: {local_auroc:.4f}")
    print()
    print(f"Results:\n{results_dir}")
    print()
    print(f"Checkpoint:\n{checkpoint_path}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
