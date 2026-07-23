import numpy as np
from pathlib import Path
from blimpy import Waterfall


def chunk_observation(h5_path: str, out_dir: str, chunk_width: int = 4096) -> None:
    """Load a real .h5 observation and slice it into fixed-width frames."""
    fil = Waterfall(str(h5_path))
    data = fil.data[:, 0, :]          # drop the feed axis -> (16, n_freq)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_chunks = data.shape[1] // chunk_width
    for i in range(n_chunks):
        chunk = data[:, i * chunk_width : (i + 1) * chunk_width]
        np.save(out_path / f"real_{i:05d}.npy", chunk)

    print(f"Saved {n_chunks} frames of shape (16, {chunk_width}) to {out_path}")