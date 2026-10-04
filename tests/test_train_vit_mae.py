"""
Tests for the focused fixes in scripts/train_vit_mae.py:
  - _validate_resume_config rejects mismatched configs
  - _capture_rng_state / _restore_rng_state faithfully round-trip all states
"""
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Import helpers directly from the script without executing main()
# ---------------------------------------------------------------------------

_script = Path(__file__).parent.parent / "scripts" / "train_vit_mae.py"
_spec   = importlib.util.spec_from_file_location("train_vit_mae", _script)
_mod    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_validate_resume_config = _mod._validate_resume_config
_capture_rng_state      = _mod._capture_rng_state
_restore_rng_state      = _mod._restore_rng_state


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _make_config() -> dict:
    return {
        "experiment": {"name": "test", "seed": 42},
        "model": {"embed_dim": 128, "mask_ratio": 0.75},
        "training": {"epochs": 10, "batch_size": 16},
    }


def test_validate_resume_config_passes_when_identical():
    cfg = _make_config()
    _validate_resume_config(deepcopy(cfg), cfg)  # should not raise


def test_validate_resume_config_passes_when_checkpoint_config_absent():
    cfg = _make_config()
    _validate_resume_config(None, cfg)  # missing key in old checkpoint — tolerated


def test_validate_resume_config_raises_on_mismatch():
    cfg = _make_config()
    different = deepcopy(cfg)
    different["model"]["embed_dim"] = 256  # intentional mismatch
    with pytest.raises(ValueError, match="different configuration"):
        _validate_resume_config(different, cfg)


def test_validate_resume_config_raises_on_seed_mismatch():
    cfg = _make_config()
    different = deepcopy(cfg)
    different["experiment"]["seed"] = 99
    with pytest.raises(ValueError, match="different configuration"):
        _validate_resume_config(different, cfg)


# ---------------------------------------------------------------------------
# RNG capture / restore round-trip (CPU only — CI-portable)
# ---------------------------------------------------------------------------

def test_capture_rng_state_contains_expected_keys():
    device = torch.device("cpu")
    generator = torch.Generator()
    generator.manual_seed(0)
    snapshot = _capture_rng_state(device, generator)
    assert "cpu_rng_state" in snapshot
    assert "train_generator_state" in snapshot
    # CUDA/MPS keys must not be present on a CPU device
    assert "cuda_rng_state_all" not in snapshot
    assert "mps_rng_state" not in snapshot


def test_restore_rng_state_cpu_round_trip():
    """After capturing and restoring CPU RNG, the next random draw is identical."""
    device = torch.device("cpu")

    generator = torch.Generator()
    generator.manual_seed(42)

    torch.manual_seed(7)

    snapshot = _capture_rng_state(device, generator)

    # Advance both RNGs past the snapshot point
    _ = torch.rand(100)
    _ = torch.randint(0, 100, (10,), generator=generator)

    # Restore
    _restore_rng_state(snapshot, device, generator)

    # The next draw must now reproduce what would have followed the snapshot
    torch.manual_seed(7)                     # re-apply the same base seed
    _capture_rng_state(device, generator)    # consume the same state advance
    expected_global = torch.rand(5)
    expected_gen    = torch.randint(0, 100, (5,), generator=generator)

    # Restored state
    _restore_rng_state(snapshot, device, generator)
    actual_global = torch.rand(5)
    actual_gen    = torch.randint(0, 100, (5,), generator=generator)

    assert torch.equal(actual_global, expected_global), \
        "Global CPU RNG was not faithfully restored"
    assert torch.equal(actual_gen, expected_gen), \
        "DataLoader generator RNG was not faithfully restored"


def test_restore_rng_state_does_not_touch_cuda_when_cpu():
    """_restore_rng_state must not call CUDA APIs when device is CPU."""
    device = torch.device("cpu")
    generator = torch.Generator()
    generator.manual_seed(0)
    snapshot = _capture_rng_state(device, generator)
    # Inject a fake CUDA key to verify it is ignored on a CPU device
    snapshot["cuda_rng_state_all"] = object()
    # Should not raise even though cuda_rng_state_all is present but device=cpu
    _restore_rng_state(snapshot, device, generator)
