"""Remote extraction of Fine-resolution (16 x 4096) frames from BL HDF5 files.

Only HTTP Range requests are used. Every response must be HTTP 206 with a
matching Content-Range; a 200 (full body) response is closed without reading
its body and the file is aborted. Nothing is ever written to disk except the
extracted frames chosen by the caller.
"""

from __future__ import annotations

import http.client
import io
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

USER_AGENT = "deepseti-fine-extract/1.0 (range reads only; no bulk download)"

N_INTS = 16
FRAME_CHANNELS = 4096
CHANNEL_WIDTH_HZ = 2.7939677238464355
TSAMP_S = 18.253611008
REL_TOL = 1e-3

# Predeclared in the Stage 2 audit (scripts/audit_bl_fine_archive.py).
SBAND_MHZ = (1700.0, 2900.0)
MASK_MHZ = ((2300.0, 2360.0),)

METADATA_BLOCK_BYTES = 64 * 1024

# Chunk-aware rule (approved after pilot 1): one 1,048,576-channel slab per scan
# (= one gpuspec chunk = 16 rawspec chunks), 42 evenly spaced frames per slab.
SLAB_CHANNELS = 1_048_576
FRAMES_PER_SCAN = 42
N_STRATA = 15


class RangeError(RuntimeError):
    """Raised when the server does not honour a range request exactly."""


class FullBodyResponse(RangeError):
    """The server answered with a full body (HTTP 200) instead of 206."""


class FileUnavailable(RangeError):
    pass


@dataclass
class TransferStats:
    requests: int = 0
    bytes: int = 0
    statuses: dict[int, int] = field(default_factory=dict)
    full_body_responses: int = 0
    retries: int = 0

    def add(self, other: TransferStats) -> None:
        self.retries += other.retries
        self.requests += other.requests
        self.bytes += other.bytes
        self.full_body_responses += other.full_body_responses
        for k, v in other.statuses.items():
            self.statuses[k] = self.statuses.get(k, 0) + v


def parse_content_range(value: str) -> tuple[int, int, int]:
    """Parse ``bytes start-stop/total`` into integers."""

    unit, _, rest = value.strip().partition(" ")
    span, _, total = rest.partition("/")
    start, _, stop = span.partition("-")
    if unit != "bytes" or not (start.isdigit() and stop.isdigit() and total.isdigit()):
        raise RangeError(f"malformed Content-Range: {value!r}")
    return int(start), int(stop), int(total)


def check_range_response(status: int, content_range: str | None, start: int, stop: int) -> int:
    """Validate a range response; return the total file size.

    Raises ``FullBodyResponse`` for anything other than 206 so callers never
    read a full-file body.
    """

    if status != 206:
        raise FullBodyResponse(f"range not honoured (HTTP {status}); body not read")
    if not content_range:
        raise RangeError("206 response without Content-Range")
    got_start, got_stop, total = parse_content_range(content_range)
    if (got_start, got_stop) != (start, stop):
        raise RangeError(f"Content-Range {got_start}-{got_stop} != requested {start}-{stop}")
    return total


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a hard byte budget.

    Small reads (HDF5 metadata) go through a 64 KiB block cache. Reads larger
    than one block (compressed chunks) are fetched as one exact range.
    """

    def __init__(self, url: str, budget: int, timeout: float = 120.0, retries: int = 3,
                 opener=urllib.request.urlopen):
        super().__init__()
        self.url, self.budget, self.timeout, self.retries = url, budget, timeout, retries
        self._open = opener
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.spans: list[tuple[int, bytes]] = []
        self.stats = TransferStats()
        self.size: int | None = None
        self._block(0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def fetch(self, start: int, stop: int) -> bytes:
        """Fetch bytes ``start..stop`` inclusive with one validated 206 request."""

        if self.size is not None:
            stop = min(stop, self.size - 1)
        expected = stop - start + 1
        if self.stats.bytes + expected > self.budget:
            raise RangeError(f"byte budget exceeded ({self.stats.bytes}+{expected} > {self.budget})")
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{stop}"}
                )
                with self._open(req, timeout=self.timeout) as resp:
                    self.stats.statuses[resp.status] = self.stats.statuses.get(resp.status, 0) + 1
                    try:
                        total = check_range_response(
                            resp.status, resp.headers.get("Content-Range"), start, stop
                        )
                    except FullBodyResponse:
                        self.stats.full_body_responses += 1
                        raise
                    data = resp.read(expected + 1)
                if len(data) != expected:
                    raise RangeError(f"got {len(data)} bytes, expected {expected}")
                break
            except RangeError:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 410):
                    raise FileUnavailable(f"HTTP {exc.code}") from exc
                last = exc
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
                    ConnectionError) as exc:
                last = exc
            self.stats.retries += 1
            time.sleep(2 * attempt)
        else:
            raise RangeError(f"network failure: {last}")
        if self.size is None:
            self.size = total
        self.stats.requests += 1
        self.stats.bytes += len(data)
        return data

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * METADATA_BLOCK_BYTES
            self.cache[index] = self.fetch(start, start + METADATA_BLOCK_BYTES - 1)
        return self.cache[index]

    def prefetch(self, spans: list[tuple[int, int]]) -> None:
        """Fetch ``(offset, size)`` spans, merging adjacent ones into one request.

        Later reads fully inside a prefetched span are served locally.
        """

        merged: list[list[int]] = []
        for off, size in sorted(spans):
            if merged and off <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], off + size)
            else:
                merged.append([off, off + size])
        for lo, hi in merged:
            self.spans.append((lo, self.fetch(lo, hi - 1)))

    def _from_spans(self, pos: int, n: int) -> bytes | None:
        for lo, data in self.spans:
            if lo <= pos and pos + n <= lo + len(data):
                return data[pos - lo : pos - lo + n]
        return None

    def readinto(self, buf):
        n = min(len(buf), max(0, self.size - self.pos))
        local = self._from_spans(self.pos, n)
        if local is not None:
            buf[:n] = local
        elif n > METADATA_BLOCK_BYTES:
            buf[:n] = self.fetch(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            while len(out) < n:
                index, offset = divmod(self.pos + len(out), METADATA_BLOCK_BYTES)
                out += self._block(index)[offset : offset + n - len(out)]
            buf[:n] = out
        self.pos += n
        return n


@dataclass(frozen=True)
class FineGeometry:
    fch1: float
    foff: float
    nchans: int
    n_ints: int
    tsamp: float
    chunks: tuple[int, ...]
    compression: str | None

    def channel_freq_mhz(self, channel: int | np.ndarray) -> float | np.ndarray:
        """Centre frequency of a channel index (fch1 is channel 0)."""

        return self.fch1 + np.asarray(channel) * self.foff

    def window_freq_bounds(self, start: int, width: int = FRAME_CHANNELS) -> tuple[float, float]:
        """(low, high) outer edges in MHz of channels ``start .. start+width-1``."""

        a = self.channel_freq_mhz(start)
        b = self.channel_freq_mhz(start + width - 1)
        half = abs(self.foff) / 2
        return float(min(a, b) - half), float(max(a, b) + half)


def freq_to_channel(geom: FineGeometry, freq_mhz: float) -> int:
    """Nearest channel index for a frequency; works for negative ``foff``."""

    ch = round((freq_mhz - geom.fch1) / geom.foff)
    if not 0 <= ch < geom.nchans:
        raise ValueError(f"{freq_mhz} MHz outside file coverage")
    return ch


def check_fine_geometry(geom: FineGeometry) -> list[str]:
    """Return the list of geometry violations (empty when compatible)."""

    problems = []
    if geom.n_ints != N_INTS:
        problems.append(f"n_ints {geom.n_ints} != {N_INTS}")
    if abs(abs(geom.foff) * 1e6 - CHANNEL_WIDTH_HZ) > REL_TOL * CHANNEL_WIDTH_HZ:
        problems.append(f"channel width {abs(geom.foff) * 1e6} Hz")
    if abs(geom.tsamp - TSAMP_S) > REL_TOL * TSAMP_S:
        problems.append(f"tsamp {geom.tsamp}")
    if geom.nchans < FRAME_CHANNELS:
        problems.append(f"nchans {geom.nchans} < {FRAME_CHANNELS}")
    return problems


def eligible_window_starts(
    geom: FineGeometry,
    band_mhz: tuple[float, float] = SBAND_MHZ,
    masks_mhz: tuple[tuple[float, float], ...] = MASK_MHZ,
    width: int = FRAME_CHANNELS,
) -> np.ndarray:
    """Start channels of the fixed, non-overlapping ``width``-channel grid.

    Windows lie on the grid ``k * width`` (so they never straddle a chunk whose
    size is a multiple of ``width``). A window is eligible when it lies entirely
    inside ``band_mhz`` and does not touch any masked interval. Frequency-only;
    no data values are used.
    """

    starts = np.arange(0, geom.nchans - width + 1, width, dtype=np.int64)
    a = geom.channel_freq_mhz(starts)
    b = geom.channel_freq_mhz(starts + width - 1)
    half = abs(geom.foff) / 2
    lo, hi = np.minimum(a, b) - half, np.maximum(a, b) + half
    keep = (lo >= band_mhz[0]) & (hi <= band_mhz[1])
    for m_lo, m_hi in masks_mhz:
        keep &= (hi <= m_lo) | (lo >= m_hi)
    return starts[keep]


def select_windows(eligible: np.ndarray, n: int, seed: int, key: str) -> np.ndarray:
    """Stratified deterministic choice of ``n`` window starts.

    The eligible grid (in channel order) is cut into ``n`` contiguous strata of
    near-equal count; one window is drawn per stratum with
    ``random.Random(f"{seed}|{key}")``. Returns starts sorted by channel.
    """

    if n > len(eligible):
        raise ValueError(f"requested {n} windows but only {len(eligible)} eligible")
    rng = random.Random(f"{seed}|{key}")
    edges = np.linspace(0, len(eligible), n + 1).round().astype(int)
    picks = [eligible[rng.randrange(edges[i], edges[i + 1])] for i in range(n)]
    return np.array(sorted(picks), dtype=np.int64)


def read_geometry(ds) -> FineGeometry:
    """Geometry from an h5py ``data`` dataset (time, feed, frequency layout)."""

    attrs = ds.attrs
    shape = ds.shape
    if len(shape) != 3 or shape[1] != 1:
        raise ValueError(f"unexpected data shape {shape}")
    nchans = int(attrs["nchans"])
    if shape[2] != nchans:
        raise ValueError(f"shape {shape} inconsistent with nchans {nchans}")
    return FineGeometry(
        fch1=float(attrs["fch1"]),
        foff=float(attrs["foff"]),
        nchans=nchans,
        n_ints=int(shape[0]),
        tsamp=float(attrs["tsamp"]),
        chunks=tuple(ds.chunks) if ds.chunks else (),
        compression=str(ds.compression) if ds.compression else (
            ",".join(str(k) for k in ds._filters) or None
        ),
    )


def eligible_slab_starts(geom: FineGeometry, slab: int = SLAB_CHANNELS) -> np.ndarray:
    """Slab-grid starts (``k * slab``) passing the band and mask rules."""

    return eligible_window_starts(geom, width=slab)


def stratum_slab_choice(centres_mhz: np.ndarray, stratum: int, q: float = 0.5,
                        n_strata: int = N_STRATA) -> int:
    """Index (into ``centres_mhz``) of the slab chosen for fractional position ``q``.

    ``centres_mhz`` are the eligible slab centre frequencies in descending order.
    The list is cut into ``n_strata`` contiguous near-equal-count strata. The
    target frequency is ``f_first + q * (f_last - f_first)`` over the stratum's
    first and last slab, and the nearest eligible slab *inside the stratum*
    wins (ties -> higher frequency). ``q = 0.5`` is the stratum midpoint.
    """

    if len(centres_mhz) < n_strata:
        raise ValueError(f"{len(centres_mhz)} eligible slabs < {n_strata} strata")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q={q} outside [0, 1]")
    edges = np.linspace(0, len(centres_mhz), n_strata + 1).round().astype(int)
    a, b = edges[stratum], edges[stratum + 1]
    target = centres_mhz[a] + q * (centres_mhz[b - 1] - centres_mhz[a])
    return a + int(np.argmin(np.abs(centres_mhz[a:b] - target)))


def assign_strata(group_ids: list[str], seed: int, split: str,
                  n_strata: int = N_STRATA) -> dict[str, int]:
    """Deterministic, balanced group -> stratum assignment within one split."""

    return {g: k for g, (k, _, _) in stratum_slots(group_ids, seed, split, n_strata).items()}


def stratum_slots(group_ids: list[str], seed: int, split: str,
                  n_strata: int = N_STRATA) -> dict[str, tuple[int, int, int]]:
    """group -> (stratum k, slot j, m groups in that stratum).

    Sorted group ids are shuffled with ``random.Random(f"{seed}|chunk_strata|{split}")``;
    rank ``i`` gives ``k = i % n_strata`` and ``j = i // n_strata``, so ``j`` orders
    the groups inside a stratum by the same seeded rank.
    """

    order = sorted(group_ids)
    random.Random(f"{seed}|chunk_strata|{split}").shuffle(order)
    m = np.bincount([i % n_strata for i in range(len(order))], minlength=n_strata)
    return {g: (i % n_strata, i // n_strata, int(m[i % n_strata])) for i, g in enumerate(order)}


def slot_fraction(j: int, m: int) -> float:
    """Target fractional position ``(j + 0.5) / m`` of slot ``j`` of ``m``."""

    return (j + 0.5) / m


def slab_window_offsets(n: int = FRAMES_PER_SCAN, slab: int = SLAB_CHANNELS,
                        width: int = FRAME_CHANNELS) -> np.ndarray:
    """Evenly spaced, non-overlapping window offsets inside one slab.

    Window ``i`` is grid cell ``floor((i + 0.5) * cells / n)`` with
    ``cells = slab // width``; no randomness and no data values.
    """

    cells = slab // width
    if n > cells:
        raise ValueError(f"{n} windows do not fit in {cells} cells")
    idx = np.floor((np.arange(n) + 0.5) * cells / n).astype(np.int64)
    return idx * width


def slab_chunk_spans(ds, start: int, slab: int = SLAB_CHANNELS) -> list[tuple[int, int]]:
    """File ``(byte_offset, size)`` of every chunk the slab touches, all integrations."""

    chunk = ds.chunks[-1]
    if start % chunk or slab % chunk:
        raise ValueError("slab must be aligned to the chunk grid")
    spans = []
    for t in range(ds.shape[0]):
        for c in range(start, start + slab, chunk):
            info = ds.id.get_chunk_info_by_coord((t, 0, c))
            spans.append((info.byte_offset, info.size))
    return spans


def fetch_slab(ds, rf: HTTPRangeFile, start: int, slab: int = SLAB_CHANNELS,
               spans: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Read ``(n_ints, slab)`` channels from ``start`` with coalesced chunk prefetch.

    Chunk byte locations come from the HDF5 chunk index; every chunk touched by
    the slab in every integration is fetched exactly once, then h5py decodes it
    from memory.
    """

    rf.prefetch(spans if spans is not None else slab_chunk_spans(ds, start, slab))
    before = rf.stats.bytes
    arr = ds[:, 0, start : start + slab]
    if rf.stats.bytes != before:
        raise RangeError("slab decode triggered extra HTTP transfer")
    if arr.shape != (ds.shape[0], slab):
        raise ValueError(f"slab shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def slice_frames(slab_arr: np.ndarray, offsets: np.ndarray,
                 width: int = FRAME_CHANNELS) -> np.ndarray:
    """Cut ``(len(offsets), n_ints, width)`` frames from an in-memory slab."""

    frames = np.stack([slab_arr[:, o : o + width] for o in offsets]).astype(np.float32)
    if frames.shape[1:] != (slab_arr.shape[0], width):
        raise ValueError(f"frame shape {frames.shape[1:]}")
    return frames


def extract_windows(ds, starts: np.ndarray, width: int = FRAME_CHANNELS) -> np.ndarray:
    """Read ``(len(starts), 16, width)`` float32 frames from an open dataset.

    Frequency stays in file channel order (descending frequency when foff < 0).
    """

    frames = np.empty((len(starts), ds.shape[0], width), dtype=np.float32)
    for i, s in enumerate(starts):
        block = ds[:, 0, int(s) : int(s) + width]
        if block.shape != (ds.shape[0], width):
            raise ValueError(f"window at {s} has shape {block.shape}")
        frames[i] = block
    return frames
