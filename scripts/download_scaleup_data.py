from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from urllib.parse import urlparse

MANIFEST = Path("configs/data/scaleup_observations.csv")
ROOT = Path("data/raw/scaleup_mid")


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    completed = 0
    failed = []

    print(f"Dataset contains {len(rows)} observations.\n")

    for i, row in enumerate(rows, 1):
        target = row["target"]
        split = row["split"]
        url = row["url"]
        expected = int(row["size_bytes"])

        filename = Path(urlparse(url).path).name
        out = ROOT / split / filename
        out.parent.mkdir(parents=True, exist_ok=True)

        actual = out.stat().st_size if out.exists() else 0

        # Already complete
        if actual == expected:
            print(
                f"[{i:02d}/{len(rows)}] ✓ COMPLETE "
                f"{split:10s} {target:15s} "
                f"{actual / 1024**2:.1f} MB"
            )
            completed += 1
            continue

        # Something impossible happened: local file bigger than archive file.
        if actual > expected:
            print(
                f"[{i:02d}/{len(rows)}] ! OVERSIZED {target} "
                f"— deleting and restarting"
            )
            out.unlink()
            actual = 0

        if actual > 0:
            print(
                f"\n[{i:02d}/{len(rows)}] ↻ RESUME "
                f"{split:10s} {target}"
            )
            print(
                f"    {actual / 1024**2:.1f} / "
                f"{expected / 1024**2:.1f} MB"
            )
        else:
            print(
                f"\n[{i:02d}/{len(rows)}] ↓ DOWNLOAD "
                f"{split:10s} {target}"
            )

        result = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry", "30",
                "--retry-delay", "5",
                "--retry-all-errors",
                "--connect-timeout", "30",
                "--continue-at", "-",
                "--progress-bar",
                url,
                "-o", str(out),
            ],
            check=False,
        )

        actual = out.stat().st_size if out.exists() else 0

        if result.returncode == 0 and actual == expected:
            print(f"    ✓ VERIFIED {actual / 1024**2:.1f} MB")
            completed += 1
        else:
            print(
                f"    ✗ INCOMPLETE "
                f"{actual / 1024**2:.1f} / "
                f"{expected / 1024**2:.1f} MB"
            )
            failed.append((target, split, actual, expected))

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Complete:          {completed}/{len(rows)}")
    print(f"Failed/incomplete: {len(failed)}")

    if failed:
        print("\nIncomplete:")
        for target, split, actual, expected in failed:
            print(
                f"{target:15s} {split:10s} "
                f"{actual / 1024**2:.1f} / "
                f"{expected / 1024**2:.1f} MB"
            )

        print("\nRun this same script again to resume.")
    else:
        print("\n✓ ALL 40 OBSERVATIONS DOWNLOADED AND SIZE-VERIFIED")


if __name__ == "__main__":
    main()
