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