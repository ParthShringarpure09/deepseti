from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse

from deepseti.data.manifest import load_manifest


SOURCE = Path("configs/data/scaleup_observations.csv")
OUTPUT = Path("configs/data/scaleup_training_manifest.csv")

PURPOSE = {
    "train": "training",
    "validation": "validation",
    "test": "evaluation",
}


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output_rows = []

    for row in rows:
        split = row["split"]
        target = row["target"]
        filename = Path(urlparse(row["url"]).path).name

        if split not in PURPOSE:
            raise ValueError(f"Unexpected split: {split}")

        output_rows.append(
            {
                "observation_id": f"{target.lower()}_mid_scaleup_001",
                "source_name": target,
                "filename": filename,
                "raw_subdir": f"scaleup_mid/{split}",
                "resolution": "mid",
                "split": split,
                "purpose": PURPOSE[split],
            }
        )

    fieldnames = [
        "observation_id",
        "source_name",
        "filename",
        "raw_subdir",
        "resolution",
        "split",
        "purpose",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Validate using DeepSeti's existing manifest validator.
    validated = load_manifest(OUTPUT)

    counts = {
        split: sum(row["split"] == split for row in validated)
        for split in ("train", "validation", "test")
    }

    print("Created:", OUTPUT)
    print("Total observations:", len(validated))
    print("Train:", counts["train"])
    print("Validation:", counts["validation"])
    print("Test:", counts["test"])
    print("\nManifest validation PASSED.")


if __name__ == "__main__":
    main()
