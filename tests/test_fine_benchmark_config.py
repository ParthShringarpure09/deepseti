from pathlib import Path

import yaml

from deepseti.evaluation import fine_benchmark as fb
from deepseti.injection import fine as inj

CONFIG_PATH = Path("configs/experiments/fine_v2_synthetic_benchmark.yaml")


def _cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())["benchmark"]


def test_config_mirrors_fine_benchmark_module():
    cfg = _cfg()
    assert cfg["seed"] == fb.SEED
    assert cfg["snr_levels"] == list(fb.SNR_LEVELS)
    assert cfg["linear_drift_hz_per_s_levels"] == list(fb.DRIFT_HZ_PER_S_LEVELS)
    assert cfg["duty_cycle_levels"] == list(fb.DUTY_CYCLE_LEVELS)
    assert cfg["carrier_counts"] == list(fb.CARRIER_COUNTS)
    assert cfg["n_validation_cadence_groups"] == fb.N_VALIDATION_GROUPS
    assert cfg["repeats_per_setting"] == fb.REPEATS_PER_SETTING
    assert cfg["geometry"]["frame_shape"] == list(fb.FRAME_SHAPE)
    assert cfg["injection"]["outside_mask_diff_tolerance"] == fb.OUTSIDE_MASK_DIFF_TOLERANCE

    combos = fb.build_combo_list()
    assert cfg["n_distinct_settings"] == len(combos)
    assert cfg["n_examples"] == len(combos) * fb.REPEATS_PER_SETTING


def test_config_mirrors_injection_fine_module():
    cfg = _cfg()["injection"]
    assert cfg["bandwidth_channels"] == inj.BANDWIDTH_CHANNELS
    assert cfg["guard_channels"] == inj.GUARD_CHANNELS
    assert cfg["margin_channels"] == inj.MARGIN_CHANNELS
    assert cfg["carrier_spacing_channels"] == inj.CARRIER_SPACING_CHANNELS


def test_config_morphology_lists_match_held_out_policy():
    cfg = _cfg()["morphologies"]
    assert tuple(cfg["development"]) == inj.DEVELOPMENT_MORPHOLOGIES
    assert tuple(cfg["held_out"]) == inj.HELD_OUT_MORPHOLOGIES
    assert set(cfg["development"]).isdisjoint(cfg["held_out"])
