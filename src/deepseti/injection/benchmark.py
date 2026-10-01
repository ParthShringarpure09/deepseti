import json
from pathlib import Path
import numpy as np

from deepseti.injection.signals import (
    inject_burst,
    inject_intermittent,
    inject_linear_drift,
    inject_multiple_carriers,
    inject_nonlinear_drift,
    inject_stationary_narrowband,
)

def load_benchmark_config(path: str | Path) -> dict:
    """Load and validate a synthetic benchmark configuration."""

    path = Path(path)

    with path.open("r") as f:
        config = json.load(f)

    if config.get("training_injections_allowed") is not False:
        raise ValueError(
            "Synthetic injections must never be allowed in training."
        )

    allowed_splits = set(config.get("allowed_splits", []))

    if "train" in allowed_splits:
        raise ValueError(
            "Training split cannot be used for synthetic injections."
        )

    if not allowed_splits.issubset({"validation", "test"}):
        raise ValueError(
            "Synthetic injections may only use validation or test splits."
        )

    return config

def generate_stationary_cases(config: dict) -> list[dict]:
    """Generate reproducible stationary-narrowband benchmark cases."""

    rng = np.random.default_rng(config["seed"])

    num_freq_bins = config["frame_shape"][1]
    repeats = config["repeats_per_setting"]

    snr_levels = config["strength"]["robust_snr_levels"]
    bandwidths = config["stationary_narrowband"]["bandwidth_bins"]

    cases = []

    for snr in snr_levels:
        for bandwidth in bandwidths:
            for repeat in range(repeats):

                margin = bandwidth

                frequency_bin = int(
                    rng.integers(
                        margin,
                        num_freq_bins - margin,
                    )
                )

                cases.append(
                    {
                        "morphology": "stationary_narrowband",
                        "robust_snr": snr,
                        "bandwidth_bins": bandwidth,
                        "frequency_bin": frequency_bin,
                        "repeat": repeat,
                    }
                )

    return cases


def generate_linear_drift_cases(config: dict) -> list[dict]:
    """Generate reproducible linear-drift benchmark cases."""

    rng = np.random.default_rng(config["seed"] + 1)

    num_time_bins = config["frame_shape"][0]
    num_freq_bins = config["frame_shape"][1]

    repeats = config["repeats_per_setting"]
    snr_levels = config["strength"]["robust_snr_levels"]
    drift_rates = config["linear_drift"]["drift_bins_per_time"]

    cases = []

    for snr in snr_levels:
        for drift in drift_rates:
            max_shift = abs(drift) * (num_time_bins - 1)

            for repeat in range(repeats):
                if drift >= 0:
                    low = 0
                    high = num_freq_bins - int(max_shift)
                else:
                    low = int(max_shift)
                    high = num_freq_bins

                start_frequency_bin = int(
                    rng.integers(low, high)
                )

                cases.append(
                    {
                        "morphology": "linear_drift",
                        "robust_snr": snr,
                        "drift_bins_per_time": drift,
                        "start_frequency_bin": start_frequency_bin,
                        "repeat": repeat,
                    }
                )

    return cases

def test_generate_linear_drift_cases_is_reproducible():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],
        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "linear_drift": {
            "drift_bins_per_time": [-20, -10, -5, 5, 10, 20],
        },
    }

    cases_a = generate_linear_drift_cases(config)
    cases_b = generate_linear_drift_cases(config)

    assert len(cases_a) == 150
    assert cases_a == cases_b

def generate_nonlinear_drift_cases(config: dict) -> list[dict]:
    """Generate reproducible nonlinear-drift benchmark cases."""

    rng = np.random.default_rng(config["seed"] + 2)

    num_time_bins = config["frame_shape"][0]
    num_freq_bins = config["frame_shape"][1]

    repeats = config["repeats_per_setting"]
    snr_levels = config["strength"]["robust_snr_levels"]

    linear_rates = config["nonlinear_drift"][
        "linear_drift_bins_per_time"
    ]

    curvatures = config["nonlinear_drift"][
        "curvature_bins_per_time2"
    ]

    cases = []

    for snr in snr_levels:
        for linear_rate in linear_rates:
            for curvature in curvatures:

                offsets = [
                    linear_rate * t + curvature * t**2
                    for t in range(num_time_bins)
                ]

                min_offset = int(np.floor(min(offsets)))
                max_offset = int(np.ceil(max(offsets)))

                low = max(0, -min_offset)
                high = min(
                    num_freq_bins,
                    num_freq_bins - max_offset,
                )

                for repeat in range(repeats):
                    start_frequency_bin = int(
                        rng.integers(low, high)
                    )

                    cases.append(
                        {
                            "morphology": "nonlinear_drift",
                            "robust_snr": snr,
                            "linear_drift_bins_per_time": linear_rate,
                            "curvature_bins_per_time2": curvature,
                            "start_frequency_bin": start_frequency_bin,
                            "repeat": repeat,
                        }
                    )

    return cases

def generate_intermittent_cases(config: dict) -> list[dict]:
    """Generate reproducible intermittent-signal benchmark cases."""

    rng = np.random.default_rng(config["seed"] + 3)

    num_time_bins = config["frame_shape"][0]
    num_freq_bins = config["frame_shape"][1]

    repeats = config["repeats_per_setting"]
    snr_levels = config["strength"]["robust_snr_levels"]
    active_fractions = config["intermittent"]["active_fraction"]

    cases = []

    for snr in snr_levels:
        for active_fraction in active_fractions:
            num_active_bins = max(
                1,
                round(active_fraction * num_time_bins),
            )

            for repeat in range(repeats):
                frequency_bin = int(
                    rng.integers(0, num_freq_bins)
                )

                active_time_bins = sorted(
                    rng.choice(
                        num_time_bins,
                        size=num_active_bins,
                        replace=False,
                    ).tolist()
                )

                cases.append(
                    {
                        "morphology": "intermittent",
                        "robust_snr": snr,
                        "active_fraction": active_fraction,
                        "frequency_bin": frequency_bin,
                        "active_time_bins": active_time_bins,
                        "repeat": repeat,
                    }
                )

    return cases

def generate_multiple_carrier_cases(config: dict) -> list[dict]:
    """Generate reproducible multiple-carrier benchmark cases."""

    rng = np.random.default_rng(config["seed"] + 4)

    num_freq_bins = config["frame_shape"][1]

    repeats = config["repeats_per_setting"]
    snr_levels = config["strength"]["robust_snr_levels"]
    carrier_counts = config["multiple_carriers"]["num_carriers"]

    cases = []

    for snr in snr_levels:
        for num_carriers in carrier_counts:
            for repeat in range(repeats):

                frequency_bins = sorted(
                    rng.choice(
                        num_freq_bins,
                        size=num_carriers,
                        replace=False,
                    ).tolist()
                )

                cases.append(
                    {
                        "morphology": "multiple_carriers",
                        "robust_snr": snr,
                        "num_carriers": num_carriers,
                        "frequency_bins": frequency_bins,
                        "repeat": repeat,
                    }
                )

    return cases

def generate_burst_cases(config: dict) -> list[dict]:
    """Generate reproducible burst-signal benchmark cases."""

    rng = np.random.default_rng(config["seed"] + 5)

    num_time_bins = config["frame_shape"][0]
    num_freq_bins = config["frame_shape"][1]

    repeats = config["repeats_per_setting"]
    snr_levels = config["strength"]["robust_snr_levels"]

    durations = config["burst"]["duration_bins"]
    bandwidths = config["burst"]["bandwidth_bins"]

    cases = []

    for snr in snr_levels:
        for duration in durations:
            for bandwidth in bandwidths:
                for repeat in range(repeats):

                    start_time_bin = int(
                        rng.integers(
                            0,
                            num_time_bins - duration + 1,
                        )
                    )

                    start_freq_bin = int(
                        rng.integers(
                            0,
                            num_freq_bins - bandwidth + 1,
                        )
                    )

                    center_frequency_bin = (
                        start_freq_bin + bandwidth // 2
                    )

                    cases.append(
                        {
                            "morphology": "burst",
                            "robust_snr": snr,
                            "duration_bins": duration,
                            "bandwidth_bins": bandwidth,
                            "start_time_bin": start_time_bin,
                            "center_frequency_bin": center_frequency_bin,
                            "repeat": repeat,
                        }
                    )

    return cases

def generate_all_benchmark_cases(config: dict) -> list[dict]:
    """Generate all synthetic benchmark cases."""

    cases = []

    cases.extend(generate_stationary_cases(config))
    cases.extend(generate_linear_drift_cases(config))
    cases.extend(generate_nonlinear_drift_cases(config))
    cases.extend(generate_intermittent_cases(config))
    cases.extend(generate_multiple_carrier_cases(config))
    cases.extend(generate_burst_cases(config))

    return cases

def apply_benchmark_case(
    frame: np.ndarray,
    case: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one generated benchmark case to a real spectrogram frame."""

    morphology = case["morphology"]
    snr = case["robust_snr"]

    if morphology == "stationary_narrowband":
        return inject_stationary_narrowband(
            frame,
            frequency_bin=case["frequency_bin"],
            snr=snr,
            bandwidth_bins=case["bandwidth_bins"],
        )

    if morphology == "linear_drift":
        return inject_linear_drift(
            frame,
            start_frequency_bin=case["start_frequency_bin"],
            drift_bins_per_time=case["drift_bins_per_time"],
            snr=snr,
        )

    if morphology == "nonlinear_drift":
        return inject_nonlinear_drift(
            frame,
            start_frequency_bin=case["start_frequency_bin"],
            linear_drift_bins_per_time=case[
                "linear_drift_bins_per_time"
            ],
            curvature_bins_per_time2=case[
                "curvature_bins_per_time2"
            ],
            snr=snr,
        )

    if morphology == "intermittent":
        return inject_intermittent(
            frame,
            frequency_bin=case["frequency_bin"],
            snr=snr,
            active_time_bins=case["active_time_bins"],
        )

    if morphology == "multiple_carriers":
        return inject_multiple_carriers(
            frame,
            frequency_bins=case["frequency_bins"],
            snr=snr,
        )

    if morphology == "burst":
        return inject_burst(
            frame,
            start_time_bin=case["start_time_bin"],
            duration_bins=case["duration_bins"],
            center_frequency_bin=case["center_frequency_bin"],
            bandwidth_bins=case["bandwidth_bins"],
            snr=snr,
        )

    raise ValueError(
        f"Unknown benchmark morphology: {morphology}"
    )

def assign_cases_to_frames(
    cases: list[dict],
    num_frames: int,
    seed: int,
) -> list[dict]:
    """Assign each benchmark case to a real base-frame index."""

    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")

    rng = np.random.default_rng(seed)

    assigned_cases = []

    for case in cases:
        assigned_case = case.copy()

        assigned_case["frame_index"] = int(
            rng.integers(0, num_frames)
        )

        assigned_cases.append(assigned_case)

    return assigned_cases

def validate_injection_split(
    split: str,
    config: dict,
) -> None:
    """Ensure synthetic injections are only applied to allowed splits."""

    allowed_splits = set(config["allowed_splits"])

    if split not in allowed_splits:
        raise ValueError(
            f"Synthetic injection is not allowed for split '{split}'. "
            f"Allowed splits: {sorted(allowed_splits)}"
        )