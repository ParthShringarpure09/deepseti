import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "observation_id",
    "source_name",
    "filename",
    "raw_subdir",
    "resolution",
    "split",
    "purpose",
}

ALLOWED_SPLITS = {
    "development",
    "train",
    "validation",
    "test",
}


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read and validate the observation manifest CSV."""

    path = Path(path)

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        columns = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        rows = list(reader)

    observation_ids = [
        row["observation_id"]
        for row in rows
    ]

    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError(
            "Manifest contains duplicate observation_id values."
        )

    for row in rows:
        if row["split"] not in ALLOWED_SPLITS:
            raise ValueError(
                f"Invalid split '{row['split']}' "
                f"for observation '{row['observation_id']}'."
            )

    target_splits: dict[str, set[str]] = {}

    for row in rows:
        if row["split"] == "development":
            continue

        target = row["source_name"]
        split = row["split"]

        if target not in target_splits:
            target_splits[target] = set()

        target_splits[target].add(split)

    for target, splits in target_splits.items():
        if len(splits) > 1:
            raise ValueError(
                f"Target '{target}' appears in multiple splits: "
                f"{sorted(splits)}"
            )

    return rows