import json

import pytest

from deepseti.injection.benchmark import *


def test_valid_benchmark_config_loads(tmp_path):
    config_path = tmp_path / "benchmark.json"

    config = {
        "training_injections_allowed": False,
        "allowed_splits": ["validation", "test"],
    }

    config_path.write_text(json.dumps(config))

    loaded = load_benchmark_config(config_path)

    assert loaded["training_injections_allowed"] is False
    assert loaded["allowed_splits"] == ["validation", "test"]


def test_training_injections_are_rejected(tmp_path):
    config_path = tmp_path / "benchmark.json"

    config = {
        "training_injections_allowed": True,
        "allowed_splits": ["validation", "test"],
    }

    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError):
        load_benchmark_config(config_path)


def test_train_split_is_rejected(tmp_path):
    config_path = tmp_path / "benchmark.json"

    config = {
        "training_injections_allowed": False,
        "allowed_splits": ["train", "test"],
    }

    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError):
        load_benchmark_config(config_path)

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

def test_validate_injection_split_rejects_train():
    config = {
        "allowed_splits": ["validation", "test"],
    }

    validate_injection_split("test", config)

    with pytest.raises(ValueError):
        validate_injection_split("train", config)
    
def test_generate_stationary_cases_is_reproducible():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],
        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "stationary_narrowband": {
            "bandwidth_bins": [1, 2, 4],
        },
    }

    cases_a = generate_stationary_cases(config)
    cases_b = generate_stationary_cases(config)

    assert len(cases_a) == 75
    assert cases_a == cases_b

def test_generate_nonlinear_drift_cases_is_reproducible():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],
        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "nonlinear_drift": {
            "linear_drift_bins_per_time": [3.0],
            "curvature_bins_per_time2": [-0.8, -0.4, 0.4, 0.8],
        },
    }

    cases_a = generate_nonlinear_drift_cases(config)
    cases_b = generate_nonlinear_drift_cases(config)

    assert len(cases_a) == 100
    assert cases_a == cases_b

def test_generate_intermittent_cases_is_reproducible():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],
        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "intermittent": {
            "active_fraction": [0.25, 0.5, 0.75],
        },
    }

    cases_a = generate_intermittent_cases(config)
    cases_b = generate_intermittent_cases(config)

    assert len(cases_a) == 75
    assert cases_a == cases_b

    for case in cases_a:
        assert len(case["active_time_bins"]) >= 1
        assert len(case["active_time_bins"]) == len(
            set(case["active_time_bins"])
        )
def test_generate_multiple_carrier_cases_is_reproducible():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],
        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "multiple_carriers": {
            "num_carriers": [2, 3, 5],
        },
    }

    cases_a = generate_multiple_carrier_cases(config)
    cases_b = generate_multiple_carrier_cases(config)

    assert len(cases_a) == 75
    assert cases_a == cases_b

    for case in cases_a:
        assert len(case["frequency_bins"]) == case["num_carriers"]
        assert len(case["frequency_bins"]) == len(
            set(case["frequency_bins"])
        )
def test_generate_burst_cases_is_reproducible():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],
        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "burst": {
            "duration_bins": [2, 4, 8],
            "bandwidth_bins": [4, 20, 64],
        },
    }

    cases_a = generate_burst_cases(config)
    cases_b = generate_burst_cases(config)

    assert len(cases_a) == 225
    assert cases_a == cases_b

    for case in cases_a:
        assert 0 <= case["start_time_bin"] < 16
        assert 0 <= case["center_frequency_bin"] < 4096

def test_generate_all_benchmark_cases():
    config = {
        "seed": 42,
        "repeats_per_setting": 5,
        "frame_shape": [16, 4096],

        "strength": {
            "robust_snr_levels": [0.5, 1.0, 2.0, 5.0, 10.0],
        },

        "stationary_narrowband": {
            "bandwidth_bins": [1, 2, 4],
        },

        "linear_drift": {
            "drift_bins_per_time": [-20, -10, -5, 5, 10, 20],
        },

        "nonlinear_drift": {
            "linear_drift_bins_per_time": [3.0],
            "curvature_bins_per_time2": [-0.8, -0.4, 0.4, 0.8],
        },

        "intermittent": {
            "active_fraction": [0.25, 0.5, 0.75],
        },

        "multiple_carriers": {
            "num_carriers": [2, 3, 5],
        },

        "burst": {
            "duration_bins": [2, 4, 8],
            "bandwidth_bins": [4, 20, 64],
        },
    }

    cases = generate_all_benchmark_cases(config)

    assert len(cases) == 700


def test_apply_benchmark_case():
    rng = np.random.default_rng(42)

    frame = rng.normal(
        loc=100.0,
        scale=5.0,
        size=(16, 4096),
    ).astype(np.float32)

    case = {
        "morphology": "stationary_narrowband",
        "robust_snr": 5.0,
        "frequency_bin": 2000,
        "bandwidth_bins": 1,
    }

    injected, mask = apply_benchmark_case(
        frame,
        case,
    )

    assert injected.shape == frame.shape
    assert mask.shape == frame.shape

    assert mask.sum() == 16
    assert mask[:, 2000].all()

    assert np.all(
        injected[:, 2000] > frame[:, 2000]
    )

def test_assign_cases_to_frames_is_reproducible():
    cases = [
        {"morphology": "stationary_narrowband"},
        {"morphology": "linear_drift"},
        {"morphology": "burst"},
    ]

    assigned_a = assign_cases_to_frames(
        cases,
        num_frames=20,
        seed=42,
    )

    assigned_b = assign_cases_to_frames(
        cases,
        num_frames=20,
        seed=42,
    )

    assert assigned_a == assigned_b

    for case in assigned_a:
        assert 0 <= case["frame_index"] < 20