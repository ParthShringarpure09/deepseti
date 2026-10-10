"""Train the DeepSeti SSL-CNN with VICReg on normal radio observations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow running directly from the repository without requiring installation.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepseti.injection.dataset import load_split_frame_pool
from deepseti.models.ssl_encoder import SSLEncoder
from deepseti.models.ssl_projection import SSLProjectionHead
from deepseti.preprocessing.frames import normalize_frames
from deepseti.ssl.dataset import SSLFrameDataset
from deepseti.training.ssl_epoch import (
    evaluate_ssl_epoch,
    train_ssl_epoch,
)
from deepseti.utils.device import detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepSeti SSL-CNN with VICReg."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to SSL experiment YAML configuration.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional total epoch override. "
            "Useful for smoke testing without editing the frozen config."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory override. "
            "Useful for keeping smoke-test outputs separate."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint from which to resume training.",
    )

    return parser.parse_args()


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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected YAML configuration dictionary, got {type(config)}."
        )

    return config


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=ROOT,
                text=True,
            )
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    os.replace(
        temporary,
        path,
    )


def save_checkpoint(
    path: Path,
    state: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        state,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def count_observations(
    metadata: list[dict],
) -> int:
    return len(
        {
            item["observation_id"]
            for item in metadata
        }
    )


def main() -> None:
    args = parse_args()

    config_path = args.config

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(
        config_path
    )

    experiment_config = config["experiment"]
    data_config = config["data"]
    augmentation_config = config["augmentation"]
    model_config = config["model"]
    projection_config = config["projection"]
    vicreg_config = config["vicreg"]
    training_config = config["training"]
    output_config = config["output"]

    seed = int(
        experiment_config["seed"]
    )

    seed_everything(seed)

    device = detect_device()

    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(training_config["epochs"])
    )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path(
            output_config["checkpoint_dir"]
        )

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = Path(
        data_config["manifest"]
    )

    raw_root = Path(
        data_config["raw_root"]
    )

    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root

    print()
    print("DeepSeti SSL-CNN V1")
    print("====================")
    print(
        f"Experiment : "
        f"{experiment_config['name']}"
    )
    print(
        f"Device     : {device}"
    )

    if device.type == "cuda":
        print(
            "GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Seed       : {seed}"
    )
    print(
        f"Epochs     : {epochs}"
    )
    print(
        f"Output     : {output_dir}"
    )
    print()

    print("Loading normal training observations...")

    train_raw, train_metadata = load_split_frame_pool(
        manifest_path=manifest_path,
        raw_root=raw_root,
        split=data_config["train_split"],
        f_start=float(data_config["f_start"]),
        f_stop=float(data_config["f_stop"]),
        time_window=int(
            data_config["time_window"]
        ),
        freq_window=int(
            data_config["freq_window"]
        ),
    )

    print("Normalizing training frames...")

    train_frames = normalize_frames(
        train_raw
    )

    del train_raw

    print("Loading normal validation observations...")

    validation_raw, validation_metadata = (
        load_split_frame_pool(
            manifest_path=manifest_path,
            raw_root=raw_root,
            split=data_config["validation_split"],
            f_start=float(data_config["f_start"]),
            f_stop=float(data_config["f_stop"]),
            time_window=int(
                data_config["time_window"]
            ),
            freq_window=int(
                data_config["freq_window"]
            ),
        )
    )

    print("Normalizing validation frames...")

    validation_frames = normalize_frames(
        validation_raw
    )

    del validation_raw

    print()
    print(
        "Train observations :",
        count_observations(
            train_metadata
        ),
    )
    print(
        "Train frames       :",
        len(train_frames),
    )
    print(
        "Val observations   :",
        count_observations(
            validation_metadata
        ),
    )
    print(
        "Val frames         :",
        len(validation_frames),
    )

    sigma = float(
        augmentation_config[
            "gaussian_sigma"
        ]
    )

    train_dataset = SSLFrameDataset(
        train_frames,
        sigma=sigma,
        deterministic=False,
        base_seed=seed,
    )

    validation_dataset = SSLFrameDataset(
        validation_frames,
        sigma=sigma,
        deterministic=True,
        base_seed=seed + 4200,
    )

    batch_size = int(
        training_config["batch_size"]
    )

    num_workers = int(
        training_config["num_workers"]
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    encoder = SSLEncoder(
        embedding_dim=int(
            model_config["embedding_dim"]
        ),
        pooled_frequency_bins=int(
            model_config[
                "pooled_frequency_bins"
            ]
        ),
    ).to(device)

    projector = SSLProjectionHead(
        input_dim=int(
            model_config["embedding_dim"]
        ),
        hidden_dim=int(
            projection_config["hidden_dim"]
        ),
        output_dim=int(
            projection_config["output_dim"]
        ),
    ).to(device)

    parameters = (
        list(encoder.parameters())
        + list(projector.parameters())
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(
            training_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
    )

    encoder_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )

    projector_parameters = sum(
        parameter.numel()
        for parameter in projector.parameters()
    )

    total_parameters = (
        encoder_parameters
        + projector_parameters
    )

    vicreg_kwargs = {
        "invariance_weight": float(
            vicreg_config[
                "invariance_weight"
            ]
        ),
        "variance_weight": float(
            vicreg_config[
                "variance_weight"
            ]
        ),
        "covariance_weight": float(
            vicreg_config[
                "covariance_weight"
            ]
        ),
        "target_std": float(
            vicreg_config[
                "target_std"
            ]
        ),
        "eps": float(
            vicreg_config["eps"]
        ),
    }

    history_path = (
        output_dir
        / "train_history.json"
    )

    metadata_path = (
        output_dir
        / "run_metadata.json"
    )

    latest_path = (
        output_dir
        / "latest.pt"
    )

    best_path = (
        output_dir
        / "best.pt"
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_invariance": [],
        "train_variance": [],
        "train_covariance": [],
        "validation_loss": [],
        "validation_invariance": [],
        "validation_variance": [],
        "validation_covariance": [],
    }

    start_epoch = 1
    best_validation_loss = float("inf")

    if args.resume is not None:
        resume_path = args.resume

        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path

        print()
        print(
            f"Resuming from: {resume_path}"
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        encoder.load_state_dict(
            checkpoint[
                "encoder_state_dict"
            ]
        )

        projector.load_state_dict(
            checkpoint[
                "projector_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            int(checkpoint["epoch"])
            + 1
        )

        best_validation_loss = float(
            checkpoint[
                "best_validation_loss"
            ]
        )

        if "loader_generator_state" in checkpoint:
            loader_generator.set_state(
                checkpoint[
                    "loader_generator_state"
                ]
            )

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint[
                    "torch_rng_state"
                ].cpu()
            )

        if (
            device.type == "cuda"
            and checkpoint.get(
                "cuda_rng_state_all"
            )
            is not None
        ):
            torch.cuda.set_rng_state_all(
                checkpoint[
                    "cuda_rng_state_all"
                ]
            )

        if history_path.exists():
            with history_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                history = json.load(
                    handle
                )

    metadata = {
        "experiment": experiment_config[
            "name"
        ],
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "seed": seed,
        "effective_epochs": epochs,
        "batch_size": batch_size,
        "encoder_parameters": encoder_parameters,
        "projector_parameters": projector_parameters,
        "total_parameters": total_parameters,
        "train_frames": len(
            train_frames
        ),
        "validation_frames": len(
            validation_frames
        ),
        "train_observations": (
            count_observations(
                train_metadata
            )
        ),
        "validation_observations": (
            count_observations(
                validation_metadata
            )
        ),
        "config": config,
        "start_time": utc_now(),
        "end_time": None,
    }

    write_json(
        metadata_path,
        metadata,
    )

    print()
    print(
        f"Encoder parameters   : "
        f"{encoder_parameters:,}"
    )
    print(
        f"Projector parameters : "
        f"{projector_parameters:,}"
    )
    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )
    print(
        f"Train batches        : "
        f"{len(train_loader)}"
    )
    print(
        f"Validation batches   : "
        f"{len(validation_loader)}"
    )
    print()

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        train_metrics = train_ssl_epoch(
            encoder=encoder,
            projector=projector,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(
                training_config[
                    "grad_clip_norm"
                ]
            ),
            vicreg_kwargs=vicreg_kwargs,
        )

        validation_metrics = (
            evaluate_ssl_epoch(
                encoder=encoder,
                projector=projector,
                loader=validation_loader,
                device=device,
                vicreg_kwargs=vicreg_kwargs,
            )
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train={train_metrics['loss']:.4f} "
            f"(inv={train_metrics['invariance']:.3e}, "
            f"var={train_metrics['variance']:.4f}, "
            f"cov={train_metrics['covariance']:.4f}) | "
            f"val={validation_metrics['loss']:.4f} "
            f"(inv={validation_metrics['invariance']:.3e}, "
            f"var={validation_metrics['variance']:.4f}, "
            f"cov={validation_metrics['covariance']:.4f})"
        )

        history["epoch"].append(
            epoch
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )
        history["train_invariance"].append(
            train_metrics["invariance"]
        )
        history["train_variance"].append(
            train_metrics["variance"]
        )
        history["train_covariance"].append(
            train_metrics["covariance"]
        )

        history[
            "validation_loss"
        ].append(
            validation_metrics["loss"]
        )
        history[
            "validation_invariance"
        ].append(
            validation_metrics[
                "invariance"
            ]
        )
        history[
            "validation_variance"
        ].append(
            validation_metrics[
                "variance"
            ]
        )
        history[
            "validation_covariance"
        ].append(
            validation_metrics[
                "covariance"
            ]
        )

        is_best = (
            validation_metrics["loss"]
            < best_validation_loss
        )

        if is_best:
            best_validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )

        checkpoint_state = {
            "epoch": epoch,
            "encoder_state_dict": (
                encoder.state_dict()
            ),
            "projector_state_dict": (
                projector.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "train_metrics": (
                train_metrics
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "config": config,
            "loader_generator_state": (
                loader_generator.get_state()
            ),
            "torch_rng_state": (
                torch.get_rng_state()
            ),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None
            ),
        }

        save_checkpoint(
            latest_path,
            checkpoint_state,
        )

        if is_best:
            save_checkpoint(
                best_path,
                checkpoint_state,
            )

            print(
                "  -> saved new best checkpoint "
                f"(val={best_validation_loss:.4f})"
            )

        write_json(
            history_path,
            history,
        )

    metadata["end_time"] = (
        utc_now()
    )

    metadata[
        "best_validation_loss"
    ] = best_validation_loss

    write_json(
        metadata_path,
        metadata,
    )

    print()
    print("Training complete.")
    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}",
    )
    print(
        "Best checkpoint:",
        best_path,
    )


if __name__ == "__main__":
    main()