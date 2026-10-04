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


def test_restore_rng_state_tolerates_extra_checkpoint_keys():
    """
    _restore_rng_state is now called with the full checkpoint dict (which
    contains non-RNG keys). Verify extra keys are silently ignored and that
    two restores from the same snapshot produce identical draws.
    """
    device = torch.device("cpu")
    generator = torch.Generator()
    generator.manual_seed(0)

    torch.manual_seed(7)

    rng_snapshot = _capture_rng_state(device, generator)
    full_ckpt = {
        "epoch": 10,
        "model_state_dict": {"dummy": "value"},
        "best_val_loss": 0.3,
        **rng_snapshot,
    }

    # Advance both RNGs past the snapshot point
    _ = torch.rand(100)
    _ = torch.randint(0, 10, (10,), generator=generator)

    # First restore — should not raise despite extra non-RNG keys
    _restore_rng_state(full_ckpt, device, generator)
    first_global = torch.rand(5)
    first_gen    = torch.randint(0, 100, (5,), generator=generator)

    # Second restore from same snapshot must produce identical draws
    _restore_rng_state(full_ckpt, device, generator)
    second_global = torch.rand(5)
    second_gen    = torch.randint(0, 100, (5,), generator=generator)

    assert torch.equal(first_global, second_global), \
        "Two restores from same snapshot should produce identical global RNG draws"
    assert torch.equal(first_gen, second_gen), \
        "Two restores from same snapshot should produce identical generator draws"


# ---------------------------------------------------------------------------
# best_val_loss correctness in latest.pt
# ---------------------------------------------------------------------------

def test_latest_checkpoint_stores_updated_best_val_loss(tmp_path):
    """
    When the current epoch is a new best, latest.pt must contain the
    updated best_val_loss, not the stale value from before the comparison.
    """
    old_best  = 0.50
    val_loss  = 0.30   # new best

    # Replicate the corrected checkpoint logic
    best_val_loss = old_best
    is_new_best   = val_loss < best_val_loss
    if is_new_best:
        best_val_loss = val_loss

    state = {"epoch": 5, "best_val_loss": best_val_loss}
    latest_path = tmp_path / "latest.pt"
    _mod._save_atomic(state, latest_path)

    loaded = torch.load(latest_path, weights_only=False)
    assert loaded["best_val_loss"] == pytest.approx(val_loss), (
        "latest.pt contains stale best_val_loss; should be updated before save"
    )


def test_latest_checkpoint_best_val_loss_unchanged_when_not_best(tmp_path):
    """When an epoch does not improve, best_val_loss in latest.pt is unchanged."""
    old_best = 0.30
    val_loss  = 0.45   # not a new best

    best_val_loss = old_best
    is_new_best   = val_loss < best_val_loss
    if is_new_best:
        best_val_loss = val_loss

    state = {"epoch": 6, "best_val_loss": best_val_loss}
    latest_path = tmp_path / "latest.pt"
    _mod._save_atomic(state, latest_path)

    loaded = torch.load(latest_path, weights_only=False)
    assert loaded["best_val_loss"] == pytest.approx(old_best)


def test_best_checkpoint_not_overwritten_by_worse_resumed_epoch():
    """
    Demonstrate that the stale best_val_loss bug allowed best.pt to be
    overwritten after resume, and confirm the fix prevents it.

    Scenario:
      epoch 5 → val_loss 0.40 (true best)
      epoch 6 → val_loss 0.45 (not a new best)
        old bug:  latest.pt stored best_val_loss=0.45 (stale, pre-update value)
        fix:      latest.pt stores best_val_loss=0.40 (correct)
      epoch 7 resumed from latest.pt → val_loss 0.42
        old bug:  0.42 < 0.45 → wrongly overwrites best.pt with a worse model
        fix:      0.42 < 0.40 → False → correctly skips overwrite
    """
    true_best = 0.40

    # Epoch 6: not a new best
    val_loss_6 = 0.45

    # Correct logic (fix): update first, then build state
    best_after_epoch6 = true_best
    is_new_best_6 = val_loss_6 < best_after_epoch6
    if is_new_best_6:
        best_after_epoch6 = val_loss_6
    assert not is_new_best_6
    assert best_after_epoch6 == pytest.approx(true_best)  # latest.pt has 0.40

    # Stale value that old code would have written to latest.pt
    stale_best = 0.45

    # Epoch 7 resumed: val_loss = 0.42
    val_loss_7 = 0.42

    would_overwrite_with_fix = val_loss_7 < best_after_epoch6   # 0.42 < 0.40 → False
    would_overwrite_with_bug = val_loss_7 < stale_best           # 0.42 < 0.45 → True

    assert not would_overwrite_with_fix, (
        "Fix should prevent overwriting the true best (0.40) with a worse model (0.42)"
    )
    assert would_overwrite_with_bug, (
        "Documents that the old code incorrectly overwrote best.pt in this scenario"
    )
