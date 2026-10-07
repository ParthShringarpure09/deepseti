"""Quality-control the frozen DeepSeti scale-up dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from blimpy import Waterfall


MANIFEST = Path("configs/data/scaleup_observations.csv")
RAW_ROOT = Path("data/raw/scaleup_mid")

F_START = 2200.0
F_STOP = 2300.0
TIME_WINDOW = 16
FREQ_WINDOW = 4096


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    passed = 0
    failed = []
    total_frames = {"train": 0, "validation": 0, "test": 0}

    print(f"Checking {len(rows)} observations...\n")

    for i, row in enumerate(rows, 1):
        target = row["target"]
        split = row["split"]
        filename = Path(urlparse(row["url"]).path).name
        path = RAW_ROOT / split / filename

        problems = []

        try:
            wf = Waterfall(
                str(path),
                f_start=F_START,
                f_stop=F_STOP,
            )

            data = np.asarray(wf.data)

            n_time = int(data.shape[0])
            n_freq = int(data.shape[-1])

            finite = bool(np.isfinite(data).all())

            time_blocks = n_time // TIME_WINDOW
            freq_blocks = n_freq // FREQ_WINDOW
            frames = time_blocks * freq_blocks

            if n_time < TIME_WINDOW:
                problems.append(f"only {n_time} time integrations")

            if n_freq < FREQ_WINDOW:
                problems.append(f"only {n_freq} frequency channels")

            if not finite:
                problems.append("contains NaN/Inf")

            if frames < 1:
                problems.append("produces zero usable frames")

            if problems:
                status = "FAIL"
                failed.append((target, split, "; ".join(problems)))
            else:
                status = "PASS"
                passed += 1
                total_frames[split] += frames

            print(
                f"[{i:02d}/{len(rows)}] {status:4s} "
                f"{split:10s} {target:15s} "
                f"shape={tuple(data.shape)!s:18s} "
                f"frames={frames:4d} "
                f"finite={finite}"
            )

            del wf
            del data

        except Exception as exc:
            failed.append((target, split, repr(exc)))
            print(
                f"[{i:02d}/{len(rows)}] FAIL "
                f"{split:10s} {target:15s} "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 65)
    print("QC SUMMARY")
    print("=" * 65)
    print(f"Passed: {passed}/{len(rows)}")
    print(f"Failed: {len(failed)}")

    print("\nUsable frame counts")
    print("-------------------")
    print(f"Train:      {total_frames['train']}")
    print(f"Validation: {total_frames['validation']}")
    print(f"Test:       {total_frames['test']}")
    print(f"Total:      {sum(total_frames.values())}")

    if failed:
        print("\nFailures:")
        for target, split, reason in failed:
            print(f"  {target:15s} {split:10s} {reason}")
    else:
        print("\n✓ ALL 40 OBSERVATIONS PASSED QC")


if __name__ == "__main__":
    main()
