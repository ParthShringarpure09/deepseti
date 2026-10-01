from pathlib import Path

import pytest

from deepseti.data.manifest import load_manifest


def test_manifest_loads_valid_file(tmp_path: Path):
    manifest_path = tmp_path / "observations.csv"

    manifest_path.write_text(
        "observation_id,source_name,filename,raw_subdir,resolution,split,purpose\n"
        "obs001,STAR_A,file_a.h5,dev_mid,mid,train,training\n"
    )

    rows = load_manifest(manifest_path)

    assert len(rows) == 1
    assert rows[0]["observation_id"] == "obs001"


def test_manifest_rejects_duplicate_ids(tmp_path: Path):
    manifest_path = tmp_path / "observations.csv"

    manifest_path.write_text(
        "observation_id,source_name,filename,raw_subdir,resolution,split,purpose\n"
        "obs001,STAR_A,file_a.h5,dev_mid,mid,train,training\n"
        "obs001,STAR_B,file_b.h5,dev_mid,mid,test,evaluation\n"
    )

    with pytest.raises(ValueError):
        load_manifest(manifest_path)


def test_manifest_rejects_invalid_split(tmp_path: Path):
    manifest_path = tmp_path / "observations.csv"

    manifest_path.write_text(
        "observation_id,source_name,filename,raw_subdir,resolution,split,purpose\n"
        "obs001,STAR_A,file_a.h5,dev_mid,mid,wrong_split,training\n"
    )

    with pytest.raises(ValueError):
        load_manifest(manifest_path)

def test_manifest_rejects_target_leakage(tmp_path: Path):
    manifest_path = tmp_path / "observations.csv"

    manifest_path.write_text(
        "observation_id,source_name,filename,raw_subdir,resolution,split,purpose\n"
        "obs001,STAR_A,file_a.h5,dev_mid,mid,train,training\n"
        "obs002,STAR_A,file_b.h5,dev_mid,mid,test,evaluation\n"
    )

    with pytest.raises(ValueError):
        load_manifest(manifest_path)