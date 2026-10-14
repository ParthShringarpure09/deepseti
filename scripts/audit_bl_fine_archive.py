"""Read-only audit of the Breakthrough Listen Open Data archive for Dataset V2.

Answers whether a GBT / Fine / HDF5 / S-band dataset with ~90 independent
cadence groups (60 train / 15 validation / 15 final-test) is feasible, BEFORE any
observation data is downloaded.

What this script does:
  1. Enumerates every GBT Fine HDF5 record from the official query-files API,
     recursively splitting MJD windows to get under the API's 10,000-row cap.
  2. Derives URL/filename metadata and flags provisional S-band candidates.
  3. Reads HDF5 headers for a deterministic, stratified sample of files using
     small HTTP range requests (hard per-file byte budget; never full files).
  4. Infers ABACAD-like cadences conservatively from session/scan/target order.
  5. Writes a per-file catalog CSV and a machine-readable summary JSON.

What it deliberately does NOT do: download observation data, select on signal
content, mask frequencies, create train/validation/test splits, or claim that
inferred quantities (S-band membership, sessions, cadences) are archive-native.

Usage:
    uv run python scripts/audit_bl_fine_archive.py
    uv run python scripts/audit_bl_fine_archive.py --skip-headers
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Archive interface (validated interactively on 2026-10-13; see summary JSON)
# ---------------------------------------------------------------------------
API_URL = "http://seti.berkeley.edu/opendata/api/query-files"
# Parameter names come from the official search page's static/bl-data-search.js.
# An empty target means "all targets"; the API requires the key to be present.
API_FILTERS = {"target": "", "telescopes": "gbt", "file-types": "hdf5", "grades": "fine"}
API_ROW_CAP = 10_000
API_FIELDS = [
    "id", "target", "telescope", "file_type", "url", "size", "md5sum", "mjd",
    "utc", "ra", "decl", "center_freq", "quality", "tempX", "tempY",
]
USER_AGENT = "deepseti-archive-audit/1.0 (metadata audit; no bulk download)"

# Enumeration window. MJD 50000 (1995) is far earlier than any BL GBT data.
DEFAULT_MJD_START = 50_000.0
DEFAULT_WINDOW_DAYS = 365.25
MIN_WINDOW_DAYS = 1e-4  # ~9 s; a capped window this small means enumeration is impossible

# ---------------------------------------------------------------------------
# Scientific definitions (fixed before looking at any results)
# ---------------------------------------------------------------------------
# Deliberately broad centre-frequency range around GBT S-band (BL records
# ~1.8-2.8 GHz). Membership is PROVISIONAL: the API exposes no receiver field.
SBAND_CANDIDATE_MHZ = (1700.0, 2900.0)
# Single-node L-band files can have centres up to ~1.85 GHz. A scan whose
# highest centre frequency (across all of its files) is below this is flagged
# as edge-ambiguous rather than counted as an S-band candidate.
SBAND_SCAN_MAX_CENTER_MHZ = 2000.0
# Predefined region reported (NOT removed) for a later fixed frame-level mask.
PROBLEM_REGION_MHZ = (2300.0, 2360.0)

TARGET_CHANNEL_WIDTH_HZ = 2.7939677238464355
TARGET_TSAMP_S = 18.253611008
TARGET_N_INTS = 16
GEOMETRY_REL_TOL = 0.05  # +/-5% on channel width and integration time; n_ints exact

# Metadata-only flag for non-astronomical-pointing scans (diagnostics, noise
# diode, tests, blank names). Based on the target NAME, never on data content.
NONSTANDARD_TARGET_RE = re.compile(r"DIAG|TEST|NOISE|_ND($|_)|NDON|NDOFF|^\s*$")

# Cadence inference (conservative)
CADENCE_MAX_GAP_MIN = 15.0  # max start-to-start gap between consecutive scans

# Header verification
DEFAULT_HEADER_SAMPLE = 150
DEFAULT_SEED = 20261013
RANGE_BLOCK_BYTES = 64 * 1024
MAX_HEADER_BYTES_PER_FILE = 1024 * 1024
HEADER_ABORT_AFTER_INITIAL_FAILURES = 10

OUT_DIR = Path("results/data_audit")
CATALOG_NAME = "bl_fine_sband_catalog.csv"
SUMMARY_NAME = "bl_fine_sband_summary.json"

FNAME_RE = re.compile(
    r"guppi_(?P<imjd>\d{5})_(?P<isec>\d{5})_(?P<src>.+)_(?P<scan>\d{4})"
    r"\.(?P<pipe>gpuspec|rawspec)\.(?P<prod>\d{4})\.h5$"
)
GBT_SESSION_RE = re.compile(r"^AGBT\d{2}[A-Z]_\d+_\d+$")


class ArchiveError(RuntimeError):
    """Base class for archive failures that must stop the audit."""


class ArchiveNetworkError(ArchiveError):
    pass


class ArchiveSchemaError(ArchiveError):
    pass


class ArchiveEnumerationError(ArchiveError):
    pass


class HeaderReadError(RuntimeError):
    pass


class ArchiveFileUnavailable(HeaderReadError):
    """Catalogued file the data server reports as missing (not retried)."""

    def __init__(self, status: int, reason: str):
        super().__init__(f"HTTP {status} {reason}")
        self.status = status


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def mjd_now() -> float:
    return 40587.0 + time.time() / 86400.0


def mjd_to_datetime(mjd: pd.Series) -> pd.Series:
    return pd.to_datetime((mjd - 40587.0) * 86400.0, unit="s", utc=True)


def describe(values) -> dict | None:
    arr = pd.Series(values).dropna().astype(float)
    if arr.empty:
        return None
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(arr.median()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def counts(values, top: int | None = None) -> dict:
    c = Counter("<missing>" if pd.isna(v) else v for v in values)
    items = c.most_common(top) if top else sorted(c.items(), key=lambda kv: str(kv[0]))
    return {str(k): int(v) for k, v in items}


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())  # np.bytes_.item() is bytes
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def gib(n_bytes: float) -> float:
    return round(float(n_bytes) / 1024**3, 3)


# ---------------------------------------------------------------------------
# 1. Catalog enumeration
# ---------------------------------------------------------------------------
def api_get(params: dict, timeout: float, retries: int) -> tuple[str, dict]:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
            last_error = exc
            wait = 5 * 2 ** (attempt - 1)
            print(f"    ! network error ({exc}); retry {attempt}/{retries} in {wait}s")
            time.sleep(wait)
            continue
        try:
            return url, json.loads(body)
        except json.JSONDecodeError as exc:
            raise ArchiveSchemaError(
                f"Non-JSON response from {url}: {body[:200]!r}"
            ) from exc
    raise ArchiveNetworkError(f"Giving up after {retries} attempts: {url} ({last_error})")


def validate_payload(payload: dict, url: str, start: float, end: float) -> list[dict]:
    if not isinstance(payload, dict) or payload.get("result") != "success":
        raise ArchiveSchemaError(f"API did not report success for {url}: {payload!r:.300}")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ArchiveSchemaError(f"'data' is not a list for {url}")
    for row in rows:
        missing = [f for f in API_FIELDS if f not in row]
        if missing:
            raise ArchiveSchemaError(f"Row {row.get('id')} missing fields {missing} ({url})")
        mjd = row["mjd"]
        # The time filter is inclusive at both ends (verified). A row outside the
        # requested window means the API ignored the filter: enumeration unsafe.
        if mjd is None or not (start - 1e-6 <= float(mjd) <= end + 1e-6):
            raise ArchiveSchemaError(
                f"Row {row['id']} has mjd={mjd} outside window [{start}, {end}]; "
                "the time filter is not behaving as validated."
            )
    return rows


def enumerate_catalog(args) -> tuple[list[dict], list[dict], int]:
    leaves: list[dict] = []
    n_requests = 0

    def fetch(start: float, end: float, depth: int) -> list[dict]:
        nonlocal n_requests
        params = {**API_FILTERS, "time-start": f"{start:.8f}", "time-end": f"{end:.8f}"}
        url, payload = api_get(params, args.timeout, args.retries)
        n_requests += 1
        rows = validate_payload(payload, url, start, end)
        indent = "  " * depth
        if len(rows) >= API_ROW_CAP:
            if end - start < MIN_WINDOW_DAYS:
                raise ArchiveEnumerationError(
                    f"Window [{start}, {end}] still returns {len(rows)} rows; cannot "
                    "enumerate below the API cap."
                )
            print(f"  {indent}[{start:.4f}, {end:.4f}] capped at {len(rows)} -> split")
            mid = (start + end) / 2.0
            return fetch(start, mid, depth + 1) + fetch(mid, end, depth + 1)
        if rows:
            print(f"  {indent}[{start:.4f}, {end:.4f}] {len(rows)} rows")
        leaves.append(
            {"mjd_start": start, "mjd_end": end, "n_rows": len(rows), "depth": depth}
        )
        return rows

    all_rows: list[dict] = []
    start = args.mjd_start
    while start < args.mjd_end:
        end = min(start + args.window_days, args.mjd_end)
        all_rows.extend(fetch(start, end, 0))
        start = end
    return all_rows, leaves, n_requests


# ---------------------------------------------------------------------------
# 2. Derived metadata, filter validation, provisional S-band status
# ---------------------------------------------------------------------------
def derive_metadata(df: pd.DataFrame) -> pd.DataFrame:
    paths = df["url"].map(lambda u: urllib.parse.urlparse(u).path)
    df["derived_filename"] = paths.map(lambda p: p.rsplit("/", 1)[-1])

    def session_folder(path: str):
        parts = [p for p in path.split("/") if p]
        if "pipeline" in parts and parts.index("pipeline") + 1 < len(parts) - 1:
            return parts[parts.index("pipeline") + 1]
        return None

    df["derived_session_folder"] = paths.map(session_folder)
    df["derived_session_folder_is_gbt_pattern"] = df["derived_session_folder"].map(
        lambda s: isinstance(s, str) and bool(GBT_SESSION_RE.match(s))
    )
    m = df["derived_filename"].str.extract(FNAME_RE)
    df["derived_filename_parsed"] = m["scan"].notna()
    df["derived_scan_number"] = pd.to_numeric(m["scan"], errors="coerce").astype("Int64")
    df["derived_product_suffix"] = m["prod"]
    # gpuspec (2016-2020) and rawspec (2020+) both number the Fine product .0000.
    df["derived_reduction_pipeline"] = m["pipe"]
    df["derived_scan_start_key"] = m["imjd"] + "_" + m["isec"]
    df["derived_archive_era"] = np.select(
        [
            df["derived_filename"].str.startswith("spliced_"),
            df["derived_filename"].str.match(r"^blc\d+_guppi_"),
        ],
        ["spliced_multinode", "single_node"],
        default="other",
    )
    dt = mjd_to_datetime(df["mjd"].astype(float))
    df["derived_observing_day_utc"] = dt.dt.strftime("%Y-%m-%d")
    df["derived_year"] = dt.dt.year
    df["derived_target_norm"] = df["target"].fillna("").str.strip().str.upper()
    df["derived_nonstandard_target_name"] = df["derived_target_norm"].map(
        lambda t: bool(NONSTANDARD_TARGET_RE.search(t))
    )

    # Validate that the API filters really did what they claim (never trust).
    violations = []
    for _, r in df[["telescope", "file_type", "derived_product_suffix"]].iterrows():
        v = []
        if r["telescope"] != "GBT":
            v.append("telescope!=GBT")
        if r["file_type"] != "HDF5":
            v.append("file_type!=HDF5")
        if r["derived_product_suffix"] != "0000":
            v.append("not_fine_.0000_product")
        violations.append(";".join(v))
    df["derived_filter_violation"] = violations
    df["derived_filter_ok"] = df["derived_filter_violation"] == ""

    # Scan identity spans all node files of one scan (independent of frequency).
    df["derived_scan_key"] = (
        df["derived_session_folder"].fillna("?")
        + "|" + df["derived_scan_start_key"].fillna("?")
        + "|" + df["derived_scan_number"].astype(str)
    )
    df["inferred_session_id"] = df["derived_session_folder"]
    df["inferred_setup_key"] = df["center_freq"].astype(float).round(3)

    cf = df["center_freq"].astype(float)
    scan_max_cf = df.groupby("derived_scan_key")["center_freq"].transform("max")
    df["derived_scan_max_center_freq"] = scan_max_cf
    in_range = cf.between(*SBAND_CANDIDATE_MHZ)
    df["inferred_sband_status"] = np.select(
        [
            cf.isna(),
            in_range & (scan_max_cf >= SBAND_SCAN_MAX_CENTER_MHZ),
            in_range,
        ],
        ["unknown_no_center_freq", "candidate_provisional", "edge_ambiguous_possible_lband"],
        default="out_of_candidate_range",
    )
    return df


# ---------------------------------------------------------------------------
# 3. Header verification via HTTP range reads
# ---------------------------------------------------------------------------
class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP Range requests with a hard byte budget."""

    def __init__(self, url: str, timeout: float, retries: int, budget: int):
        super().__init__()
        self.url, self.timeout, self.retries, self.budget = url, timeout, retries, budget
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.bytes_fetched = 0
        self.n_requests = 0
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

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            return self.cache[index]
        start = index * RANGE_BLOCK_BYTES
        stop = start + RANGE_BLOCK_BYTES - 1
        if self.size is not None:
            stop = min(stop, self.size - 1)
        expected = stop - start + 1
        if self.bytes_fetched + expected > self.budget:
            raise HeaderReadError(
                f"byte budget exceeded ({self.bytes_fetched}+{expected} > {self.budget})"
            )
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{stop}"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status != 206:
                        # Server ignored Range: close WITHOUT reading the body.
                        raise HeaderReadError(f"range not honoured (HTTP {resp.status})")
                    total = resp.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                    if self.size is None and total.isdigit():
                        self.size = int(total)
                    data = resp.read(expected + 1)
                if len(data) > expected:
                    raise HeaderReadError("server returned more bytes than requested")
                break
            except HeaderReadError:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 410):
                    raise ArchiveFileUnavailable(exc.code, exc.reason) from exc
                last = exc
                time.sleep(2 * attempt)
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
                last = exc
                time.sleep(2 * attempt)
        else:
            raise HeaderReadError(f"network failure: {last}")
        if self.size is None:
            raise HeaderReadError("no Content-Range total size in response")
        self.cache[index] = data
        self.bytes_fetched += len(data)
        self.n_requests += 1
        return data

    def readinto(self, buf):
        n = min(len(buf), max(0, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            index, offset = divmod(self.pos + len(out), RANGE_BLOCK_BYTES)
            out += self._block(index)[offset : offset + n - len(out)]
        buf[:n] = out
        self.pos += n
        return n


def read_header(url: str, args) -> dict:
    import h5py

    rf = HTTPRangeFile(url, args.timeout, 3, MAX_HEADER_BYTES_PER_FILE)
    out: dict = {}
    try:
        with h5py.File(rf, "r") as h5:
            out["hdr_root_attrs"] = json.dumps(to_jsonable(dict(h5.attrs)))
            out["hdr_datasets"] = ",".join(sorted(h5.keys()))
            if "data" not in h5:
                raise HeaderReadError("no 'data' dataset")
            ds = h5["data"]
            attrs = {k: to_jsonable(v) for k, v in ds.attrs.items()}
            out["hdr_data_shape"] = "x".join(str(s) for s in ds.shape)
            out["hdr_data_dtype"] = str(ds.dtype)
            out["hdr_data_chunks"] = "x".join(str(s) for s in ds.chunks) if ds.chunks else ""
            out["hdr_data_attrs"] = json.dumps(attrs)
            shape = ds.shape
    finally:
        out["hdr_bytes_fetched"] = rf.bytes_fetched
        out["hdr_range_requests"] = rf.n_requests
        out["hdr_file_size_from_server"] = rf.size

    for key in ("fch1", "foff", "tsamp", "nchans", "nifs", "nbits", "tstart",
                "source_name", "telescope_id", "machine_id", "data_type",
                "rawdatafile", "DIMENSION_LABELS"):
        out[f"hdr_{key}"] = attrs.get(key)
    if len(shape) != 3:
        raise HeaderReadError(f"unexpected data rank {shape}")
    # The (time, feed, frequency) layout is identified from the shape and nchans,
    # not from DIMENSION_LABELS, which is known to be mislabelled in some files.
    nchans = attrs.get("nchans")
    out["hdr_n_ints"] = int(shape[0])
    out["hdr_nchans_shape"] = int(shape[2])
    out["hdr_nchans_consistent"] = nchans is not None and int(nchans) == int(shape[2])
    fch1, foff = attrs.get("fch1"), attrs.get("foff")
    if fch1 is not None and foff is not None:
        f_edge = float(fch1) + float(foff) * int(shape[2])
        out["hdr_channel_width_hz"] = abs(float(foff)) * 1e6
        out["hdr_f_lo_mhz"] = min(float(fch1), f_edge)
        out["hdr_f_hi_mhz"] = max(float(fch1), f_edge)
        out["hdr_bandwidth_mhz"] = out["hdr_f_hi_mhz"] - out["hdr_f_lo_mhz"]
    return out


def build_header_sample(df: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Deterministic stratified sample over era x year x 100-MHz centre bin,
    preferring session folders not yet sampled."""
    pool = df[
        df["derived_filter_ok"]
        & df["inferred_sband_status"].isin(
            ["candidate_provisional", "edge_ambiguous_possible_lband"]
        )
        & ~df["derived_nonstandard_target_name"]
    ].sort_values("id")
    rng = random.Random(seed)
    strata: dict[tuple, list] = defaultdict(list)
    for idx, r in pool.iterrows():
        key = (r["derived_archive_era"], r["derived_reduction_pipeline"],
               int(r["derived_year"]),
               int(r["center_freq"] // 100 * 100))
        strata[key].append((idx, r["derived_session_folder"]))
    for key in sorted(strata):
        rng.shuffle(strata[key])

    chosen: list[int] = []
    seen_sessions: set = set()
    keys = sorted(strata)
    while len(chosen) < n and any(strata[k] for k in keys):
        for key in keys:
            if len(chosen) >= n or not strata[key]:
                continue
            items = strata[key]
            pick = next((i for i, (_, s) in enumerate(items) if s not in seen_sessions), 0)
            idx, session = items.pop(pick)
            chosen.append(idx)
            seen_sessions.add(session)
    return chosen


def verify_headers(df: pd.DataFrame, args) -> dict:
    df["header_sampled"] = False
    df["header_status"] = ""
    df["header_http_status"] = pd.array([pd.NA] * len(df), dtype="Int64")
    sample = build_header_sample(df, args.header_sample, args.seed)
    df.loc[sample, "header_sampled"] = True
    print(f"\nHeader verification: {len(sample)} files (seed={args.seed})")

    n_ok = n_fail = n_unavailable = 0
    failures: Counter = Counter()
    failure_reasons: Counter = Counter()
    unavailable: list[dict] = []
    aborted = None
    total_bytes = 0
    for i, idx in enumerate(sample, 1):
        url = df.at[idx, "url"]
        try:
            hdr = read_header(url, args)
            for k, v in hdr.items():
                if k not in df.columns:
                    df[k] = pd.Series([None] * len(df), index=df.index, dtype=object)
                df.at[idx, k] = v
            df.at[idx, "header_status"] = "ok"
            total_bytes += hdr["hdr_bytes_fetched"]
            n_ok += 1
        except ArchiveFileUnavailable as exc:
            df.at[idx, "header_status"] = f"unavailable: {exc}"
            df.at[idx, "header_http_status"] = exc.status
            unavailable.append({"id": int(df.at[idx, "id"]), "url": url, "error": str(exc)})
            n_unavailable += 1
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, not hidden
            reason = f"{type(exc).__name__}: {exc}"[:200]
            df.at[idx, "header_status"] = f"failed: {reason}"
            failures[type(exc).__name__] += 1
            failure_reasons[reason] += 1
            n_fail += 1
        if i % 10 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)} ok={n_ok} failed={n_fail} "
                  f"unavailable={n_unavailable} bytes={total_bytes:,}")
        if n_ok == 0 and n_fail + n_unavailable >= HEADER_ABORT_AFTER_INITIAL_FAILURES:
            aborted = (
                f"First {n_fail} header reads all failed; stopped rather than "
                "escalating to larger downloads."
            )
            print(f"  ! {aborted}")
            break

    ok = df["header_status"] == "ok"
    if ok.any():
        width = df.loc[ok, "hdr_channel_width_hz"].astype(float)
        tsamp = df.loc[ok, "hdr_tsamp"].astype(float)
        nints = df.loc[ok, "hdr_n_ints"].astype(int)
        df.loc[ok, "hdr_width_match"] = (
            (width - TARGET_CHANNEL_WIDTH_HZ).abs() <= GEOMETRY_REL_TOL * TARGET_CHANNEL_WIDTH_HZ
        )
        df.loc[ok, "hdr_tsamp_match"] = (
            (tsamp - TARGET_TSAMP_S).abs() <= GEOMETRY_REL_TOL * TARGET_TSAMP_S
        )
        df.loc[ok, "hdr_nints_match"] = nints == TARGET_N_INTS
        df.loc[ok, "hdr_fine_geometry_match"] = (
            df.loc[ok, "hdr_width_match"].astype(bool)
            & df.loc[ok, "hdr_tsamp_match"].astype(bool)
            & df.loc[ok, "hdr_nints_match"].astype(bool)
        )
        lo = df.loc[ok, "hdr_f_lo_mhz"].astype(float)
        hi = df.loc[ok, "hdr_f_hi_mhz"].astype(float)
        df.loc[ok, "hdr_freq_range_within_sband_window"] = (
            (lo >= SBAND_CANDIDATE_MHZ[0]) & (hi <= SBAND_CANDIDATE_MHZ[1])
        )
        df.loc[ok, "hdr_overlaps_2300_2360_mhz"] = (
            (lo < PROBLEM_REGION_MHZ[1]) & (hi > PROBLEM_REGION_MHZ[0])
        )
        df.loc[ok, "hdr_tstart_matches_api_mjd"] = (
            (df.loc[ok, "hdr_tstart"].astype(float) - df.loc[ok, "mjd"].astype(float)).abs()
            < 1e-5
        )
    return {
        "sample_size": len(sample),
        "attempted": n_ok + n_fail + n_unavailable,
        "succeeded": n_ok,
        "failed": n_fail,
        "failure_types": dict(failures),
        "failure_reasons": dict(failure_reasons),
        "unavailable_archive_files": n_unavailable,
        "unavailable_archive_file_records": unavailable,
        "aborted": aborted,
        "total_bytes_fetched": int(total_bytes),
    }


# ---------------------------------------------------------------------------
# 4. Cadence inference
# ---------------------------------------------------------------------------
def check_abacad(w: list[dict]) -> list[str]:
    """Return a list of reasons the 6-scan window is NOT ABACAD-like (empty = ok)."""
    t = [r["target"] for r in w]
    a = t[0]
    reasons = []
    if not (t[2] == a and t[4] == a):
        reasons.append("A not at positions 1,3,5")
    offs = [t[1], t[3], t[5]]
    if a in offs:
        reasons.append("off position equals A")
    elif len(set(offs)) != 3:
        reasons.append("off targets B,C,D not distinct (e.g. ABABAB)")
    scans = [r["scan"] for r in w]
    if any(b <= a_ for a_, b in pairwise(scans)):
        reasons.append("scan numbers not strictly increasing")
    gaps = [(b["mjd"] - a_["mjd"]) * 1440.0 for a_, b in pairwise(w)]
    if max(gaps) > CADENCE_MAX_GAP_MIN:
        reasons.append(f"gap {max(gaps):.1f} min > {CADENCE_MAX_GAP_MIN:.0f} min")
    return reasons


def infer_cadences(df: pd.DataFrame) -> dict:
    df["inferred_cadence_id"] = None
    df["cadence_inference_status"] = "not_applicable"
    df["cadence_inference_reason"] = "not an S-band candidate / filter or name excluded"
    df["inferred_cadence_position"] = pd.array([pd.NA] * len(df), dtype="Int64")

    eligible = (
        df["derived_filter_ok"]
        & (df["inferred_sband_status"] == "candidate_provisional")
        & ~df["derived_nonstandard_target_name"]
    )
    no_session = eligible & (
        df["inferred_session_id"].isna() | df["derived_scan_number"].isna()
    )
    df.loc[no_session, "cadence_inference_status"] = "not_confirmed"
    df.loc[no_session, "cadence_inference_reason"] = "session folder or scan number not parseable"
    eligible &= ~no_session
    df.loc[eligible, "cadence_inference_status"] = "not_confirmed"
    df.loc[eligible, "cadence_inference_reason"] = "no A-B-A-C-A-D sequence around this scan"

    ambiguous_groups: dict[str, str] = {}
    sub = df[eligible]
    for (session, _setup), g in sub.groupby(["inferred_session_id", "inferred_setup_key"]):
        g = g.sort_values(["mjd", "derived_scan_number", "id"])
        dup = g["derived_scan_number"].duplicated(keep=False)
        for idx in g.index[dup]:
            df.at[idx, "cadence_inference_status"] = "ambiguous"
            df.at[idx, "cadence_inference_reason"] = "duplicate scan number in session/setup"
        g = g[~dup]
        seq = [
            {"idx": idx, "target": r["derived_target_norm"],
             "scan": int(r["derived_scan_number"]), "mjd": float(r["mjd"])}
            for idx, r in g.iterrows()
        ]
        n = len(seq)
        valid = [i for i in range(n - 5) if not check_abacad(seq[i : i + 6])]
        # Overlapping valid windows make cadence boundaries ambiguous: reject them.
        clusters: list[list[int]] = []
        for i in valid:
            if clusters and i < clusters[-1][-1] + 6:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        used: set[int] = set()
        for cl in clusters:
            members = range(cl[0], cl[-1] + 6)
            used.update(members)
            if len(cl) == 1:
                w = seq[cl[0] : cl[0] + 6]
                cid = f"{session}|{w[0]['target']}|scans{w[0]['scan']:04d}-{w[-1]['scan']:04d}"
                for pos, r in enumerate(w):
                    df.at[r["idx"], "inferred_cadence_id"] = cid
                    df.at[r["idx"], "inferred_cadence_position"] = pos
                    df.at[r["idx"], "cadence_inference_status"] = "inferred_abacad_like"
                    df.at[r["idx"], "cadence_inference_reason"] = (
                        "ordered targets A-B-A-C-A-D, same session folder and centre "
                        f"frequency, increasing scan numbers, gaps <= {CADENCE_MAX_GAP_MIN:.0f} min"
                    )
            else:
                first = seq[cl[0]]
                gid = f"{session}|{first['target']}|scan{first['scan']:04d}"
                ambiguous_groups[gid] = "overlapping candidate ABACAD windows"
                for j in members:
                    df.at[seq[j]["idx"], "cadence_inference_status"] = "ambiguous"
                    df.at[seq[j]["idx"], "cadence_inference_reason"] = (
                        "overlapping candidate ABACAD windows; boundaries not unique"
                    )

        # Partial / failed A-x-A runs among the remaining scans -> ambiguous, never grouped.
        i = 0
        while i < n:
            if i in used or i + 2 >= n or (i + 1) in used or (i + 2) in used:
                i += 1
                continue
            a = seq[i]["target"]
            if not (seq[i + 2]["target"] == a and seq[i + 1]["target"] != a):
                i += 1
                continue
            j = i + 2
            while (j + 2 < n and j + 1 not in used and j + 2 not in used
                   and seq[j + 2]["target"] == a and seq[j + 1]["target"] != a):
                j += 2
            if j + 1 < n and j + 1 not in used and seq[j + 1]["target"] != a:
                j += 1
            run = seq[i : j + 1]
            if len(run) >= 6:
                why = "; ".join(check_abacad(run[:6])) or "longer A-x-A run"
            else:
                why = f"partial A-x-A pattern ({len(run)} scans)"
            gid = f"{session}|{a}|scan{run[0]['scan']:04d}"
            ambiguous_groups.setdefault(gid, why)
            for r in run:
                df.at[r["idx"], "cadence_inference_status"] = "ambiguous"
                df.at[r["idx"], "cadence_inference_reason"] = why
            i = j + 1

    confirmed_prefixes = {
        cid.rsplit("|scans", 1)[0] + "|scan" + cid.rsplit("|scans", 1)[1][:4]
        for cid in df["inferred_cadence_id"].dropna().unique()
    }
    # Same scans may be ambiguous in one node's frequency slice but confirmed in another.
    independent_ambiguous = {
        k: v for k, v in ambiguous_groups.items() if k not in confirmed_prefixes
    }
    return {
        "ambiguous_groups_total": len(ambiguous_groups),
        "ambiguous_groups_not_overlapping_confirmed": len(independent_ambiguous),
        "ambiguous_reason_counts": dict(Counter(independent_ambiguous.values()).most_common()),
    }


# ---------------------------------------------------------------------------
# 5. Independence units and feasibility
# ---------------------------------------------------------------------------
def connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[str, set[str]] = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    return list(groups.values())


def cadence_feasibility(df: pd.DataFrame) -> dict:
    cad = df[df["cadence_inference_status"] == "inferred_abacad_like"]
    needed = 60 + 15 + 15
    if cad.empty:
        return {"inferred_cadences": 0, "appears_feasible": False, "needed_groups": needed}

    per_cad = cad.groupby("inferred_cadence_id").agg(
        session=("inferred_session_id", "first"),
        size_all_setups=("size", "sum"),
        n_files=("id", "size"),
        n_setups=("inferred_setup_key", "nunique"),
        era=("derived_archive_era", "first"),
        day=("derived_observing_day_utc", "first"),
    )
    a_target = (
        cad[cad["inferred_cadence_position"] == 0]
        .groupby("inferred_cadence_id")["derived_target_norm"].first()
    )
    all_targets = cad.groupby("inferred_cadence_id")["derived_target_norm"].agg(set)
    per_cad["a_target"] = a_target
    per_setup_size = cad.groupby(["inferred_cadence_id", "inferred_setup_key"])["size"].sum()

    def components(use_all_targets: bool):
        edges = []
        for cid, r in per_cad.iterrows():
            targets = all_targets[cid] if use_all_targets else {r["a_target"]}
            edges += [(f"S:{r['session']}", f"T:{t}") for t in targets]
        comps = connected_components(edges)
        sizes = sorted((sum(x.startswith("S:") for x in c) for c in comps), reverse=True)
        return len(comps), sizes

    n_comp_a, sizes_a = components(False)
    n_comp_all, sizes_all = components(True)
    sessions_per_a = per_cad.groupby("a_target")["session"].nunique()
    a_set = set(per_cad["a_target"])
    off_targets = set().union(*all_targets.values) - a_set
    a_as_off = {
        t for cid, ts in all_targets.items() for t in ts if t != per_cad.at[cid, "a_target"]
    } & a_set

    median_cad = float(per_cad["size_all_setups"].median())
    median_slice = float(per_setup_size.median())
    return {
        "needed_groups": needed,
        "inferred_cadences": len(per_cad),
        "inferred_cadences_by_era": counts(per_cad["era"]),
        "sessions_with_inferred_cadence": int(per_cad["session"].nunique()),
        "observing_days_with_inferred_cadence": int(per_cad["day"].nunique()),
        "unique_a_targets": len(a_set),
        "independence_components_session_x_A_target": n_comp_a,
        "largest_components_session_x_A_target_sessions": sizes_a[:10],
        "independence_components_session_x_all_targets": n_comp_all,
        "largest_components_session_x_all_targets_sessions": sizes_all[:10],
        "target_overlap_risk": {
            "a_targets_in_multiple_sessions": int((sessions_per_a > 1).sum()),
            "a_targets_in_multiple_cadences": int(
                (per_cad.groupby("a_target").size() > 1).sum()
            ),
            "a_targets_also_used_as_off_target": len(a_as_off),
            "off_only_targets": len(off_targets),
            "max_sessions_for_one_a_target": int(sessions_per_a.max()),
        },
        "storage_per_cadence_all_frequency_slices_gib": {
            k: gib(v) for k, v in describe(per_cad["size_all_setups"]).items() if k != "count"
        },
        "storage_per_cadence_single_frequency_slice_gib": {
            k: gib(v) for k, v in describe(per_setup_size).items() if k != "count"
        },
        "estimated_storage_90_cadences_all_slices_tib": round(needed * median_cad / 1024**4, 2),
        "estimated_storage_90_cadences_single_slice_tib": round(
            needed * median_slice / 1024**4, 2
        ),
        "appears_feasible": bool(n_comp_a >= needed),
        "appears_feasible_strict_all_targets": bool(n_comp_all >= needed),
        "feasibility_basis": (
            "Independence unit = connected component of the graph linking inferred "
            "session folders to the A targets of their inferred ABACAD-like cadences; "
            "a target observed in several sessions forces those sessions into one unit "
            "so a cadence-level split can be target-disjoint. 'strict' also links B/C/D "
            "off targets. Feasible if units >= 90. Raw file counts are NOT used."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Summary + report
# ---------------------------------------------------------------------------
def build_summary(df, args, leaves, n_requests, started, header_info, amb_info, feas):
    cand = df[
        df["derived_filter_ok"]
        & (df["inferred_sband_status"] == "candidate_provisional")
        & ~df["derived_nonstandard_target_name"]
    ]
    ok = df.get("header_status", pd.Series("", index=df.index)) == "ok"
    hdr = df[ok]

    def hdr_counts(col, rnd=None):
        if hdr.empty or col not in hdr:
            return {}
        s = hdr[col].astype(float)
        return counts(s.round(rnd) if rnd is not None else s)

    era_bw = (
        hdr.groupby("derived_archive_era")["hdr_bandwidth_mhz"].median().to_dict()
        if not hdr.empty else {}
    )
    est_overlap = None
    if era_bw:
        bw = cand["derived_archive_era"].map(era_bw)
        lo = cand["center_freq"] - bw / 2
        hi = cand["center_freq"] + bw / 2
        known = bw.notna()
        est_overlap = {
            "files_with_estimated_coverage": int(known.sum()),
            "files_estimated_to_overlap": int(
                ((lo < PROBLEM_REGION_MHZ[1]) & (hi > PROBLEM_REGION_MHZ[0]) & known).sum()
            ),
            "rule": "INFERRED: centre_freq +/- (median header-verified bandwidth of the "
                    "same archive era)/2; not verified per file",
            "era_median_bandwidth_mhz": {k: round(v, 3) for k, v in era_bw.items()},
        }

    status_counts = counts(cand["cadence_inference_status"])
    geom = {}
    if not hdr.empty:
        geom = {
            "verified_files": len(hdr),
            "channel_width_match": int(hdr["hdr_width_match"].astype(bool).sum()),
            "tsamp_match": int(hdr["hdr_tsamp_match"].astype(bool).sum()),
            "n_ints_match": int(hdr["hdr_nints_match"].astype(bool).sum()),
            "all_three_match": int(hdr["hdr_fine_geometry_match"].astype(bool).sum()),
            "all_three_match_by_era": {
                str(k): f"{int(v.astype(bool).sum())}/{len(v)}"
                for k, v in hdr.groupby("derived_archive_era")["hdr_fine_geometry_match"]
            },
            "all_three_match_by_pipeline": {
                str(k): f"{int(v.astype(bool).sum())}/{len(v)}"
                for k, v in hdr.groupby("derived_reduction_pipeline")["hdr_fine_geometry_match"]
            },
            "frequency_range_within_sband_window": int(
                hdr["hdr_freq_range_within_sband_window"].astype(bool).sum()
            ),
            "tstart_matches_api_mjd": int(hdr["hdr_tstart_matches_api_mjd"].astype(bool).sum()),
            "nchans_attr_matches_shape": int(hdr["hdr_nchans_consistent"].astype(bool).sum()),
            "tolerance": f"channel width and tsamp within +/-{GEOMETRY_REL_TOL:.0%}; "
                         f"n_ints == {TARGET_N_INTS}",
        }

    temp_avail = {
        col: counts(df[col].map(lambda v: "available" if v not in (None, "Unknown", "NULL")
                                and not pd.isna(v) else repr(v)))
        for col in ("tempX", "tempY")
    }
    quality = counts(df["quality"])
    graded = sum(v for k, v in quality.items() if k in {"A", "B", "C", "F"})

    summary = {
        "audit": "DeepSeti Dataset V2 - Breakthrough Listen GBT Fine HDF5 S-band archive audit",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - started, 1),
        "archive_source": {
            "api": API_URL,
            "web_interface": "http://seti.berkeley.edu/opendata",
            "parameter_names_source": "http://seti.berkeley.edu/opendata/static/bl-data-search.js",
            "api_row_cap": API_ROW_CAP,
            "n_api_requests": n_requests,
            "catalog_retrieved_utc": args.retrieved_utc,
            "raw_cache_used": str(args.raw_cache) if args.raw_cache else None,
        },
        "query_definition": {
            "api_filters": API_FILTERS,
            "mjd_enumeration_range": [args.mjd_start, args.mjd_end],
            "initial_window_days": args.window_days,
            "split_rule": f"split window at midpoint while it returns >= {API_ROW_CAP} rows; "
                          "time bounds are inclusive (verified), records deduplicated by id",
            "leaf_windows": leaves,
            "quality_filter": "NONE (Grade A unavailable; quality recorded as returned)",
            "sband_candidate_center_freq_mhz": list(SBAND_CANDIDATE_MHZ),
            "sband_scan_max_center_freq_mhz": SBAND_SCAN_MAX_CENTER_MHZ,
            "nonstandard_target_name_regex": NONSTANDARD_TARGET_RE.pattern,
            "target_geometry": {
                "channel_width_hz": TARGET_CHANNEL_WIDTH_HZ,
                "tsamp_s": TARGET_TSAMP_S,
                "n_ints": TARGET_N_INTS,
                "relative_tolerance": GEOMETRY_REL_TOL,
            },
        },
        "counts": {
            "api_rows_returned_incl_window_overlaps": int(args._raw_rows),
            "fine_gbt_hdf5_records_unique": len(df),
            "filter_violations": counts(df.loc[~df["derived_filter_ok"],
                                               "derived_filter_violation"]),
            "filename_unparseable": int((~df["derived_filename_parsed"]).sum()),
            "sband_status": counts(df["inferred_sband_status"]),
            "nonstandard_target_name_in_sband_window": int(
                (df["derived_nonstandard_target_name"]
                 & (df["inferred_sband_status"] == "candidate_provisional")).sum()
            ),
            "sband_candidate_records": len(cand),
            "unique_targets": int(cand["derived_target_norm"].nunique()),
            "unique_observing_days_utc": int(cand["derived_observing_day_utc"].nunique()),
            "unique_session_folders": int(cand["inferred_session_id"].nunique()),
            "session_folders_spanning_multiple_utc_days": int(
                (cand.groupby("inferred_session_id")["derived_observing_day_utc"]
                 .nunique() > 1).sum()
            ),
            "by_archive_era": counts(cand["derived_archive_era"]),
            "by_reduction_pipeline": counts(cand["derived_reduction_pipeline"]),
            "by_year": counts(cand["derived_year"]),
        },
        "quality_field": {
            "distribution_all_records": quality,
            "distribution_sband_candidates": counts(cand["quality"]),
            "graded_A_B_C_F_records": graded,
            "tempX_tempY_availability": temp_avail,
        },
        "file_size_bytes_sband_candidates": {
            **(describe(cand["size"]) or {}),
            "total_bytes": int(cand["size"].sum()),
            "total_tib": round(cand["size"].sum() / 1024**4, 2),
            "median_gib": gib(cand["size"].median()) if len(cand) else None,
        },
        "file_size_by_era_gib": {
            str(k): {kk: gib(vv) for kk, vv in describe(v).items() if kk != "count"}
            for k, v in cand.groupby("derived_archive_era")["size"]
        },
        "center_freq_mhz_sband_candidates": {
            "distribution_rounded_1mhz_top30": counts(cand["center_freq"].round(0), top=30),
            "histogram_100mhz_bins": counts((cand["center_freq"] // 100 * 100).astype(int)),
        },
        "header_verification": {
            **header_info,
            "byte_budget_per_file": MAX_HEADER_BYTES_PER_FILE,
            "range_block_bytes": RANGE_BLOCK_BYTES,
            "seed": args.seed,
            "strata": "archive era x reduction pipeline x year x 100-MHz centre-frequency bin; prefers unsampled "
                      "session folders",
            "channel_width_hz_distribution": hdr_counts("hdr_channel_width_hz", 4),
            "tsamp_s_distribution": hdr_counts("hdr_tsamp", 4),
            "n_ints_distribution": hdr_counts("hdr_n_ints"),
            "nchans_distribution": hdr_counts("hdr_nchans_shape"),
            "bandwidth_mhz_distribution": hdr_counts("hdr_bandwidth_mhz", 1),
            "frequency_coverage_mhz_distribution": counts(
                hdr["hdr_f_lo_mhz"].astype(float).round(0).astype(int).astype(str) + "-"
                + hdr["hdr_f_hi_mhz"].astype(float).round(0).astype(int).astype(str), top=40
            ) if not hdr.empty else {},
            "dimension_labels_seen": counts(hdr["hdr_DIMENSION_LABELS"].map(str))
            if not hdr.empty else {},
            "telescope_id_seen": hdr_counts("hdr_telescope_id"),
            "machine_id_seen": hdr_counts("hdr_machine_id"),
            "fine_geometry": geom,
        },
        "problem_region_2300_2360_mhz": {
            "action": "REPORTED ONLY. Not removed at observation-selection stage; "
                      "predefined region to be masked (fixed mask) at frame extraction.",
            "verified_header_files_overlapping": int(hdr["hdr_overlaps_2300_2360_mhz"]
                                                     .astype(bool).sum()) if not hdr.empty else 0,
            "verified_header_files": len(hdr),
            "catalog_estimate": est_overlap,
        },
        "cadence_inference": {
            "status_counts_sband_candidate_files": status_counts,
            "inferred_abacad_like_cadences": feas.get("inferred_cadences", 0),
            **amb_info,
            "rules": {
                "session": "INFERRED: inferred_session_id = URL path segment after "
                           "'pipeline/' (GBT session folder, e.g. AGBT16B_999_39)",
                "setup": "INFERRED: same centre frequency (rounded to 1 kHz) = same "
                         "frequency setup / compute-node slice",
                "order": "sort by API mjd then filename scan number",
                "abacad": "6 consecutive scans in one session+setup with targets "
                          "A,B,A,C,A,D (case-insensitive name match), B/C/D distinct and "
                          f"!= A, strictly increasing scan numbers, gaps <= "
                          f"{CADENCE_MAX_GAP_MIN:.0f} min",
                "ambiguity": "overlapping valid windows, partial A-x-A runs, ABABAB, "
                             "duplicate scan numbers -> 'ambiguous', never grouped",
                "cadence_id": "session|A target|first-last scan; shared across "
                              "frequency slices of the same scans",
            },
        },
        "split_feasibility_60_15_15": feas,
        "missing_metadata_fields": {
            "not_exposed_by_api": [
                "receiver/band", "channel width", "number of channels",
                "number of time integrations", "time resolution",
                "frequency start/end", "backend/instrument", "session id",
                "cadence id", "scan type / ON-OFF role", "usable Tsys (tempX/tempY)",
                "A/B/C/F quality grade (all Ungraded)",
            ],
            "available_only_from_headers": [
                "fch1", "foff", "tsamp", "nchans", "data shape (n_ints)",
                "telescope_id", "machine_id", "rawdatafile (single-node files)",
            ],
            "not_found_in_headers": ["receiver name", "session id", "cadence id",
                                     "ON/OFF role"],
        },
        "scientific_limitations": [
            (
                "Public Quality A grading was unavailable: "
                f"{graded} of {len(df)} records carry an A/B/C/F grade; Grade A is NOT "
                "a selection criterion and Ungraded data is not described as Grade A."
            ),
            (
                f"Quality field distribution is {quality}; tempX/tempY (Tsys) are not "
                "usable; Tsys grades were not reconstructed."
            ),
            (
                "S-band membership is INFERRED from centre frequency (receiver not "
                "exposed); only header-verified files have a measured frequency range, "
                "and even that does not name the receiver."
            ),
            (
                "Session and cadence IDs are INFERRED from URL folder, filename scan "
                "number, MJD order and target sequence; they are not archive-native."
            ),
            (
                "Fine product selection IS explicit via the API filter grades=fine; "
                f"{int((~df['derived_filter_ok']).sum())} returned rows violated the "
                "telescope/file-type/.{gpuspec|rawspec}.0000.h5 checks and were excluded."
            ),
            (
                "GBT and HDF5 filters ARE explicit (telescopes=gbt, file-types=hdf5) and "
                "were re-checked on every returned row."
            ),
            (
                f"API results are capped at {API_ROW_CAP}; complete retrieval required "
                "recursive MJD-window enumeration. Records with a missing MJD cannot be "
                "retrieved by time windows and would be absent."
            ),
            (
                "Detailed geometry (channel width, tsamp, n_ints, frequency range) is "
                "known only for header-verified sample files; catalog-wide geometry is "
                "not verified."
            ),
            (
                "No signal-content-based selection was performed; the 2300-2360 MHz "
                "region was not removed."
            ),
        ],
        "outputs": {
            "catalog_csv": str(args.out_dir / CATALOG_NAME),
            "catalog_scope": "ALL unique GBT Fine HDF5 records returned (any frequency); "
                             "filter on inferred_sband_status for S-band candidates",
            "column_provenance": {
                "archive_api": API_FIELDS,
                "derived_*": "deterministic parse of URL/filename/MJD",
                "inferred_*, cadence_*": "rule-based inference (see cadence_inference.rules)",
                "hdr_*": "read from HDF5 header via HTTP range requests (sampled files only)",
            },
        },
    }
    return to_jsonable(summary)


def print_report(s: dict) -> None:
    c, q, h, f = s["counts"], s["quality_field"], s["header_verification"], \
        s["split_feasibility_60_15_15"]
    size = s["file_size_bytes_sband_candidates"]
    line = "=" * 72
    print(f"\n{line}\nDATASET V2 ARCHIVE AUDIT - GBT / Fine / HDF5 / S-band (provisional)\n{line}")
    print(f"API requests: {s['archive_source']['n_api_requests']}  leaf windows: "
          f"{len(s['query_definition']['leaf_windows'])}  runtime: {s['runtime_s']} s")
    print(f"Fine GBT HDF5 records (unique):     {c['fine_gbt_hdf5_records_unique']:,}")
    print(f"Filter violations:                  {c['filter_violations'] or 'none'}")
    print(f"S-band status:                      {c['sband_status']}")
    print(f"S-band candidate records:           {c['sband_candidate_records']:,}")
    print(f"  unique targets:                   {c['unique_targets']:,}")
    print(f"  unique observing days (UTC):      {c['unique_observing_days_utc']:,}")
    print(f"  unique session folders:           {c['unique_session_folders']:,}")
    print(f"  by era:                           {c['by_archive_era']}")
    print(f"  by pipeline:                      {c['by_reduction_pipeline']}")
    print(f"  by year:                          {c['by_year']}")
    print(f"Quality field (all records):        {q['distribution_all_records']}")
    print(f"tempX availability:                 {q['tempX_tempY_availability']['tempX']}")
    if size.get("count"):
        print(f"File size: min {gib(size['min'])} / median {gib(size['median'])} / "
              f"max {gib(size['max'])} GiB; total {size['total_tib']} TiB")
    print(f"Centre freq (100 MHz bins):         "
          f"{s['center_freq_mhz_sband_candidates']['histogram_100mhz_bins']}")
    print(f"\nHeader verification: sample {h['sample_size']}, ok {h['succeeded']}, "
          f"failed {h['failed']} {h['failure_types'] or ''}, unavailable (HTTP 404/410) "
          f"{h.get('unavailable_archive_files', 0)}, bytes {h['total_bytes_fetched']:,}")
    if h.get("aborted"):
        print(f"  ABORTED: {h['aborted']}")
    print(f"  channel width (Hz):  {h['channel_width_hz_distribution']}")
    print(f"  tsamp (s):           {h['tsamp_s_distribution']}")
    print(f"  n_ints:              {h['n_ints_distribution']}")
    print(f"  bandwidth (MHz):     {h['bandwidth_mhz_distribution']}")
    print(f"  coverage (MHz):      {dict(list(h['frequency_coverage_mhz_distribution'].items())[:8])}")
    if h["fine_geometry"]:
        g = h["fine_geometry"]
        print(f"  ~2.79 Hz & ~18.25 s & 16 ints: {g['all_three_match']}/{g['verified_files']} "
              f"(by era {g['all_three_match_by_era']}; "
              f"by pipeline {g['all_three_match_by_pipeline']})")
    p = s["problem_region_2300_2360_mhz"]
    print(f"\n2300-2360 MHz overlap (reported, not removed): verified "
          f"{p['verified_header_files_overlapping']}/{p['verified_header_files']}; "
          f"catalog estimate {p['catalog_estimate'] and p['catalog_estimate']['files_estimated_to_overlap']}"
          f"/{p['catalog_estimate'] and p['catalog_estimate']['files_with_estimated_coverage']} (inferred)")
    ci = s["cadence_inference"]
    print("\nCadence inference (INFERRED, not archive-native):")
    print(f"  file status:            {ci['status_counts_sband_candidate_files']}")
    print(f"  ABACAD-like cadences:   {ci['inferred_abacad_like_cadences']}")
    print(f"  ambiguous groups:       {ci['ambiguous_groups_not_overlapping_confirmed']} "
          f"{ci['ambiguous_reason_counts']}")
    print(f"\n60/15/15 feasibility (needs {f['needed_groups']} independent groups):")
    if f.get("inferred_cadences"):
        print(f"  sessions with a cadence:            {f['sessions_with_inferred_cadence']}")
        print(f"  unique A targets:                   {f['unique_a_targets']}")
        print(f"  independent units (session~A):      "
              f"{f['independence_components_session_x_A_target']} "
              f"(largest {f['largest_components_session_x_A_target_sessions'][:5]})")
        print(f"  independent units (strict, all):    "
              f"{f['independence_components_session_x_all_targets']} "
              f"(largest {f['largest_components_session_x_all_targets_sessions'][:5]})")
        print(f"  target overlap risk:                {f['target_overlap_risk']}")
        print(f"  storage / cadence, all slices:      "
              f"{f['storage_per_cadence_all_frequency_slices_gib']} GiB")
        print(f"  storage / cadence, one slice:       "
              f"{f['storage_per_cadence_single_frequency_slice_gib']} GiB")
        print(f"  est. storage for 90 cadences:       "
              f"{f['estimated_storage_90_cadences_all_slices_tib']} TiB (all slices) / "
              f"{f['estimated_storage_90_cadences_single_slice_tib']} TiB (one slice)")
    print(f"  APPEARS FEASIBLE: {f['appears_feasible']} "
          f"(strict: {f.get('appears_feasible_strict_all_targets')})")
    print(line)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--mjd-start", type=float, default=DEFAULT_MJD_START)
    p.add_argument("--mjd-end", type=float, default=None,
                   help="default: current MJD + 1 (recorded in the summary)")
    p.add_argument("--window-days", type=float, default=DEFAULT_WINDOW_DAYS)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--header-sample", type=int, default=DEFAULT_HEADER_SAMPLE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--skip-headers", action="store_true")
    p.add_argument("--raw-cache", type=Path, default=None,
                   help="JSON snapshot of raw API rows; loaded if it exists, else written")
    args = p.parse_args()
    if args.mjd_end is None:
        args.mjd_end = float(np.ceil(mjd_now() + 1.0))
    return args


def main() -> int:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_cache and args.raw_cache.exists():
        cache = json.loads(args.raw_cache.read_text(encoding="utf-8"))
        rows, leaves, n_requests = cache["rows"], cache["leaf_windows"], cache["n_requests"]
        args.mjd_start, args.mjd_end = cache["mjd_range"]
        args.retrieved_utc = cache["retrieved_utc"]
        print(f"Loaded {len(rows):,} raw API rows from cache {args.raw_cache} "
              f"(retrieved {args.retrieved_utc})")
    else:
        print(f"Enumerating {API_URL} {API_FILTERS} over MJD "
              f"[{args.mjd_start}, {args.mjd_end}]")
        try:
            rows, leaves, n_requests = enumerate_catalog(args)
        except ArchiveError as exc:
            print(f"\nAUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        args.retrieved_utc = datetime.now(UTC).isoformat(timespec="seconds")
        if args.raw_cache:
            args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
            args.raw_cache.write_text(json.dumps({
                "api_url": API_URL, "api_filters": API_FILTERS,
                "mjd_range": [args.mjd_start, args.mjd_end],
                "retrieved_utc": args.retrieved_utc, "n_requests": n_requests,
                "leaf_windows": leaves, "rows": rows,
            }), encoding="utf-8")
    args._raw_rows = len(rows)

    df = pd.DataFrame(rows, columns=API_FIELDS)
    conflicting = df.groupby("id").nunique(dropna=False).gt(1).any(axis=1).sum()
    if conflicting:
        print(f"\nAUDIT FAILED: {conflicting} archive ids returned with conflicting "
              "metadata across windows.", file=sys.stderr)
        return 2
    df = df.drop_duplicates("id").sort_values(["mjd", "id"]).reset_index(drop=True)
    for col in ("size", "mjd", "ra", "decl", "center_freq"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"\n{len(rows):,} rows returned, {len(df):,} unique archive ids")

    df = derive_metadata(df)
    if args.skip_headers:
        header_info = {"sample_size": 0, "attempted": 0, "succeeded": 0, "failed": 0,
                       "failure_types": {}, "failure_reasons": {},
                       "unavailable_archive_files": 0,
                       "unavailable_archive_file_records": [],
                       "aborted": "skipped (--skip-headers)",
                       "total_bytes_fetched": 0}
        df["header_sampled"] = False
        df["header_status"] = ""
    else:
        header_info = verify_headers(df, args)
    amb_info = infer_cadences(df)
    feas = cadence_feasibility(df)

    summary = build_summary(df, args, leaves, n_requests, started, header_info,
                            amb_info, feas)
    catalog_path = args.out_dir / CATALOG_NAME
    summary_path = args.out_dir / SUMMARY_NAME
    df.to_csv(catalog_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_report(summary)
    print(f"Wrote {catalog_path}\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
