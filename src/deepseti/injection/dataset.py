from pathlib import Path
import json

import numpy as np
from blimpy import Waterfall

from deepseti.preprocessing.frames import extract_frames
from deepseti.injection.benchmark import (
    apply_benchmark_case,
    assign_cases_to_frames,
)
from deepseti.preprocessing.frames import robust_normalize_frame

def load_observation_frames(
    file_path: str | Path,
    f_start: float = 2200,
    f_stop: float = 2300,
    time_window: int = 16,
    freq_window: int = 4096,
) -> np.ndarray:
    """
    Load raw spectrogram frames from one Breakthrough Listen observation.

    Frames are intentionally NOT normalized here because synthetic
    signals must be injected into raw power data first.
    """

    file_path = Path(file_path)

    wf = Waterfall(
        str(file_path),
        f_start=f_start,
        f_stop=f_stop,
    )

    data = wf.data.squeeze()

    frames = extract_frames(
        data,
        time_window=time_window,
        freq_window=freq_window,
    )

    return frames.astype(np.float32)

from deepseti.data.manifest import load_manifest


def load_split_frame_pool(
    manifest_path: str | Path,
    raw_root: str | Path,
    split: str,
    f_start: float = 2200,
    f_stop: float = 2300,
    time_window: int = 16,
    freq_window: int = 4096,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load all raw frames belonging to one observation-level split.

    Returns the frame array plus provenance metadata for every frame.
    """

    raw_root = Path(raw_root)
    rows = load_manifest(manifest_path)

    split_rows = [
        row
        for row in rows
        if row["split"] == split
    ]

    if not split_rows:
        raise ValueError(
            f"No observations found for split '{split}'."
        )

    all_frames = []
    metadata = []

    for row in split_rows:
        file_path = (
            raw_root
            / row["raw_subdir"]
            / row["filename"]
        )

        frames = load_observation_frames(
            file_path,
            f_start=f_start,
            f_stop=f_stop,
            time_window=time_window,
            freq_window=freq_window,
        )

        all_frames.append(frames)

        for frame_index in range(len(frames)):
            metadata.append(
                {
                    "observation_id": row["observation_id"],
                    "source_name": row["source_name"],
                    "frame_index_within_observation": frame_index,
                }
            )

    frame_pool = np.concatenate(
        all_frames,
        axis=0,
    )

    return frame_pool, metadata



def build_injected_benchmark(
    raw_frames: np.ndarray,
    frame_metadata: list[dict],
    cases: list[dict],
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    """
    Build paired clean and injected benchmark frames.

    Each synthetic case is assigned to a real raw frame.
    The clean frame and injected frame are both normalized
    using the standard DeepSeti preprocessing.
    """

    assigned_cases = assign_cases_to_frames(
        cases,
        num_frames=len(raw_frames),
        seed=seed,
    )

    clean_frames = []
    injected_frames = []
    signal_masks = []
    benchmark_metadata = []

    for case_id, case in enumerate(assigned_cases):
        frame_index = case["frame_index"]

        raw_frame = raw_frames[frame_index]

        # Clean control version of the same real frame.
        normalized_clean = robust_normalize_frame(
            raw_frame
        )

        # Add the synthetic signal to the raw frame first.
        injected_raw, signal_mask = apply_benchmark_case(
            raw_frame,
            case,
        )

        # Normalize only after injection.
        normalized_injected = robust_normalize_frame(
            injected_raw
        )

        clean_frames.append(normalized_clean)
        injected_frames.append(normalized_injected)
        signal_masks.append(signal_mask)

        metadata = {
            "case_id": case_id,
            "frame_index": frame_index,
            **frame_metadata[frame_index],
            **case,
        }

        benchmark_metadata.append(metadata)

    return (
        np.stack(clean_frames),
        np.stack(injected_frames),
        np.stack(signal_masks),
        benchmark_metadata,
    )




def save_benchmark_dataset(
    output_dir: str | Path,
    clean_frames: np.ndarray,
    injected_frames: np.ndarray,
    signal_masks: np.ndarray,
    metadata: list[dict],
) -> None:
    """Save a generated synthetic benchmark to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "clean_frames.npy",
        clean_frames,
    )

    np.save(
        output_dir / "injected_frames.npy",
        injected_frames,
    )

    np.save(
        output_dir / "signal_masks.npy",
        signal_masks,
    )

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )