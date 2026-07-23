"""Synthetic 'quiet' spectrogram generation via setigen, for MAE pretraining data."""

import setigen as stg
import astropy.units as u
import numpy as np


def generate_quiet_frame(
    n_freq_bins: int = 4096,
    n_time_bins: int = 16,
    df_hz: float = 2.7939677238464355,
    dt_s: float = 18.253611008,
    fch1_mhz: float = 6095.214842353016,  # real GBT C-band value, per setigen's official tutorial
    seed: int | None = None,
) -> np.ndarray:
    """Generate one synthetic noise-only ('quiet') spectrogram.

    Returns a (n_time_bins, n_freq_bins) array with chi-squared background
    noise sampled from setigen's pre-loaded real GBT C-band observation
    statistics, and no injected signal — a negative/normal example for MAE
    pretraining.
    """
    if seed is not None:
        np.random.seed(seed)

    frame = stg.Frame(
    fchans=n_freq_bins,
    tchans=n_time_bins,
    df=df_hz * u.Hz,
    dt=dt_s * u.s,
    fch1=fch1_mhz * u.MHz,
    seed=seed,
)
    frame.add_noise_from_obs()
    return frame.get_data()


from pathlib import Path

def generate_quiet_batch(
    n_frames: int,
    out_dir: str,
    seed_start: int = 0,
) -> None:
    """Generate n_frames quiet spectrograms and save each as a .npy file."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        frame = generate_quiet_frame(seed=seed_start + i)
        np.save(out_path / f"quiet_{i:05d}.npy", frame)

    print(f"Saved {n_frames} frames to {out_path}")



def inject_synthetic_signal(
    n_freq_bins: int = 4096,
    n_time_bins: int = 16,
    df_hz: float = 2.7939677238464355,
    dt_s: float = 18.253611008,
    fch1_mhz: float = 6095.214842353016,
    snr: float = 10,
    drift_rate_hz_per_s: float = 0,
    seed: int | None = None,
):
    frame = stg.Frame(
        fchans=n_freq_bins,
        tchans=n_time_bins,
        df=df_hz * u.Hz,
        dt=dt_s * u.s,
        fch1=fch1_mhz * u.MHz,
        seed=seed,
    )
    frame.add_noise_from_obs()

    frame.add_signal(
        stg.constant_path(
            f_start=frame.get_frequency(index=n_freq_bins // 2),
            drift_rate=drift_rate_hz_per_s * u.Hz / u.s,
        ),
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=40 * u.Hz),
        stg.constant_bp_profile(level=1),
    )

    return frame.get_data()

import json
import random

def generate_injected_batch(
    n_frames: int,
    out_dir: str,
    snr_range: tuple[float, float] = (5, 20),
    drift_range_hz_per_s: tuple[float, float] = (-2, 2),
    seed_start: int = 0,
) -> None:
    """Generate n_frames signal-injected spectrograms with randomized SNR/drift,
    saving each frame plus a JSON file of ground-truth parameters."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    labels = []
    for i in range(n_frames):
        seed = seed_start + i
        rng = random.Random(seed)
        snr = rng.uniform(*snr_range)
        drift = rng.uniform(*drift_range_hz_per_s)

        frame = inject_synthetic_signal(snr=snr, drift_rate_hz_per_s=drift, seed=seed)
        np.save(out_path / f"injected_{i:05d}.npy", frame)

        labels.append({"file": f"injected_{i:05d}.npy", "snr": snr, "drift_rate_hz_per_s": drift, "seed": seed})

    with open(out_path / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Saved {n_frames} injected frames + labels.json to {out_path}")