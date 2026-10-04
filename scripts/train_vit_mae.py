"""
Train ViT-MAE on Breakthrough Listen observations.

Run from the project root:

    python scripts/train_vit_mae.py \\
        --config configs/experiments/vit_mae_v1_gpu_reproduction.yaml

Resume an interrupted run (explicit path required):

    python scripts/train_vit_mae.py \\
        --config configs/experiments/vit_mae_v1_gpu_reproduction.yaml \\
        --resume results/checkpoints/vit_mae_v1_gpu_reproduction/latest.pt

Override device (default: auto-detect CUDA -> MPS -> CPU):

    python scripts/train_vit_mae.py \\
        --config configs/experiments/vit_mae_v1_gpu_reproduction.yaml \\
        --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Allow running without an editable install of the package.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from deepseti.injection.dataset import load_split_frame_pool
from deepseti.models.vit_mae import ViTMAE, masked_reconstruction_loss
from deepseti.preprocessing.frames import normalize_frames
from deepseti.utils.device import detect_device


# ---------------------------------------------------------------------------
# DataLoader worker seed — top-level function, spawn-safe on Windows
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from the base generator seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unavailable"


def _build_metadata(
    config: dict[str, Any],
    device: torch.device,
    num_parameters: int,
) -> dict[str, Any]:
    cuda_version = "n/a"
    gpu_name = "n/a"
    if device.type == "cuda":
        cuda_version = torch.version.cuda or "unavailable"
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "device": str(device),
        "gpu_name": gpu_name,
        "seed": config["experiment"]["seed"],
        "model_parameters": num_parameters,
        "config": config,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_time": None,
    }


# ---------------------------------------------------------------------------
# RNG state capture and restore
# ---------------------------------------------------------------------------

def _capture_rng_state(
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Snapshot all RNG states needed to faithfully resume training."""
    snapshot: dict[str, Any] = {
        "cpu_rng_state": torch.get_rng_state(),
        "train_generator_state": generator.get_state(),
    }
    if device.type == "cuda":
        snapshot["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        snapshot["mps_rng_state"] = torch.mps.get_rng_state()
    return snapshot


def _restore_rng_state(
    snapshot: dict[str, Any],
    device: torch.device,
    generator: torch.Generator,
) -> None:
    """Restore all RNG states from a snapshot produced by _capture_rng_state."""
    if "cpu_rng_state" in snapshot:
        torch.set_rng_state(snapshot["cpu_rng_state"])
    if "train_generator_state" in snapshot:
        generator.set_state(snapshot["train_generator_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in snapshot:
        torch.cuda.set_rng_state_all(snapshot["cuda_rng_state_all"])
    if device.type == "mps" and "mps_rng_state" in snapshot:
        if hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(snapshot["mps_rng_state"])


# ---------------------------------------------------------------------------
# Resume config validation
# ---------------------------------------------------------------------------

def _validate_resume_config(
    ckpt_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> None:
    """Raise ValueError if the checkpoint was created with a different config."""
    if ckpt_config is not None and ckpt_config != current_config:
        raise ValueError(
            "Resume checkpoint was created with a different configuration "
            "than the one currently loaded. Use the same --config as the "
            "original run, or start a new run without --resume."
        )


# ---------------------------------------------------------------------------
# Atomic checkpoint write
# ---------------------------------------------------------------------------

def _save_atomic(state: dict[str, Any], path: Path) -> None:
    """Write checkpoint to a temp file then atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Logging (stdout + append to file)
# ---------------------------------------------------------------------------

def _log(msg: str, log_path: Path) -> None:
    print(msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)

    exp_cfg   = config["experiment"]
    model_cfg = config["model"]
    data_cfg  = config["data"]
    train_cfg = config["training"]
    out_cfg   = config["output"]

    seed: int = int(exp_cfg["seed"])

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = detect_device()

    # Seed global RNG before model construction to reproduce weight init.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Output paths (relative to working directory — run from project root)
    checkpoint_dir = Path(out_cfg["checkpoint_dir"])
    log_path       = Path(out_cfg["log_file"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Model — V1 architecture, unchanged
    model = ViTMAE(
        frame_shape=tuple(model_cfg["frame_shape"]),
        patch_size=tuple(model_cfg["patch_size"]),
        channels=int(model_cfg["channels"]),
        embed_dim=int(model_cfg["embed_dim"]),
        encoder_depth=int(model_cfg["encoder_depth"]),
        encoder_heads=int(model_cfg["encoder_heads"]),
        encoder_mlp_dim=int(model_cfg["encoder_mlp_dim"]),
        decoder_dim=int(model_cfg["decoder_dim"]),
        decoder_depth=int(model_cfg["decoder_depth"]),
        decoder_heads=int(model_cfg["decoder_heads"]),
        decoder_mlp_dim=int(model_cfg["decoder_mlp_dim"]),
        mask_ratio=float(model_cfg["mask_ratio"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    num_params = sum(p.numel() for p in model.parameters())
    metadata   = _build_metadata(config, device, num_params)

    # Device banner
    print(f"Device:     {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:        {metadata['gpu_name']}")
        print(f"VRAM:       {props.total_memory / 1024 ** 3:.1f} GB")
    print(f"Parameters: {num_params:,}")
    print(f"Experiment: {exp_cfg['name']}")
    print(f"Git commit: {metadata['git_commit']}")
    print()

    # Resumption — explicit path only, never automatic
    start_epoch   = 1
    train_history: list[float] = []
    val_history:   list[float] = []
    best_val_loss = float("inf")
    # Checkpoint retained so _restore_rng_state can be called after the
    # DataLoader generator is constructed below. No PyTorch device RNG is
    # consumed between here and that call, so the ordering is preserved.
    _ckpt_rng: dict[str, Any] | None = None

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        # Reject mismatched configs before touching model weights.
        _validate_resume_config(ckpt.get("config"), config)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = int(ckpt["epoch"]) + 1
        train_history = list(ckpt.get("train_history", []))
        val_history   = list(ckpt.get("val_history", []))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        # All RNG states (CPU, CUDA/MPS, generator) are restored together
        # via _restore_rng_state after the generator is constructed below.
        _ckpt_rng = ckpt
        print(
            f"Resumed at epoch {start_epoch - 1}. "
            f"Best val loss so far: {best_val_loss:.6f}"
        )
        print()

    # Data loading
    # Training loader: only normal real BL observations (split="train").
    # Validation loader: only normal real BL observations (split="validation").
    # Synthetic benchmark data NEVER enters either loader.
    manifest_path = Path(data_cfg["manifest_path"])
    raw_root      = Path(data_cfg["raw_root"])

    print("Loading train observations...")
    raw_train, _ = load_split_frame_pool(
        manifest_path=manifest_path,
        raw_root=raw_root,
        split="train",
        f_start=float(data_cfg["f_start"]),
        f_stop=float(data_cfg["f_stop"]),
        time_window=int(data_cfg["time_window"]),
        freq_window=int(data_cfg["freq_window"]),
    )
    train_frames = normalize_frames(raw_train)

    print("Loading validation observations...")
    raw_val, _ = load_split_frame_pool(
        manifest_path=manifest_path,
        raw_root=raw_root,
        split="validation",
        f_start=float(data_cfg["f_start"]),
        f_stop=float(data_cfg["f_stop"]),
        time_window=int(data_cfg["time_window"]),
        freq_window=int(data_cfg["freq_window"]),
    )
    val_frames = normalize_frames(raw_val)

    train_tensor  = torch.tensor(train_frames, dtype=torch.float32).unsqueeze(1)
    val_tensor    = torch.tensor(val_frames,   dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(train_tensor)
    val_dataset   = TensorDataset(val_tensor)

    # Generator seeded separately so shuffle order is deterministic
    # independent of the global RNG consumed by model init.
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    # On resume: restore all RNG states (CPU, CUDA, MPS, and generator) in
    # a single call so masking and shuffle sequences continue faithfully.
    if _ckpt_rng is not None:
        _restore_rng_state(_ckpt_rng, device, train_generator)

    num_workers: int = int(train_cfg["dataloader_num_workers"])
    worker_fn = _seed_worker if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        worker_init_fn=worker_fn,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_fn,
        persistent_workers=num_workers > 0,
    )

    print(f"Train: {len(train_dataset)} frames, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} frames, {len(val_loader)} batches")
    print()

    # Write metadata now; end_time is filled after training completes.
    metadata_path = checkpoint_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Training loop
    total_epochs: int  = int(train_cfg["epochs"])
    grad_clip:    float = float(train_cfg["grad_clip"])

    _log(
        f"Training: epochs {start_epoch}–{total_epochs}  "
        f"device={device}  precision=float32",
        log_path,
    )

    for epoch in range(start_epoch, total_epochs + 1):

        # ----- Train -----
        model.train()
        train_sum = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predicted, target, mask = model(batch)
            loss = masked_reconstruction_loss(predicted, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_sum += loss.detach().item() * batch.size(0)
        train_loss = train_sum / len(train_dataset)

        # ----- Validate -----
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                predicted, target, mask = model(batch)
                loss = masked_reconstruction_loss(predicted, target, mask)
                val_sum += loss.item() * batch.size(0)
        val_loss = val_sum / len(val_dataset)

        train_history.append(train_loss)
        val_history.append(val_loss)

        # ----- Checkpoint -----
        # Update best_val_loss BEFORE constructing the state dict so that
        # both latest.pt and best.pt always carry the correct current best.
        is_new_best = val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss

        rng_snapshot = _capture_rng_state(device, train_generator)
        state: dict[str, Any] = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history":        train_history,
            "val_history":          val_history,
            "best_val_loss":        best_val_loss,
            "config":               config,
            **rng_snapshot,
        }
        _save_atomic(state, checkpoint_dir / "latest.pt")

        marker = ""
        if is_new_best:
            _save_atomic(state, checkpoint_dir / "best.pt")
            marker = " <- best"

        _log(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.6f} | "
            f"val {val_loss:.6f}"
            f"{marker}",
            log_path,
        )

    # ----- Final outputs -----
    history_path = checkpoint_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch":      list(range(1, len(train_history) + 1)),
                "train_loss": train_history,
                "val_loss":   val_history,
            },
            f,
            indent=2,
        )

    metadata["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    _log(f"\nDone. Best val loss: {best_val_loss:.6f}", log_path)
    _log(f"Checkpoints: {checkpoint_dir}", log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ViT-MAE on Breakthrough Listen observations. "
            "Run from the project root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Checkpoint .pt file to resume from (explicit path required).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Override auto device selection (CUDA -> MPS -> CPU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
