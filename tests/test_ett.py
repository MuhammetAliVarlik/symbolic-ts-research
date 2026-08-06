import hashlib
import json
from pathlib import Path

import pytest

from datasets.ett import VALID_VARIANTS, load_ett

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "datasets" / "snapshots" / "ett"
MANIFEST_PATH = REPO_ROOT / "datasets" / "snapshots" / "MANIFEST.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())["ett"]


@pytest.mark.parametrize("variant", VALID_VARIANTS)
def test_snapshot_matches_manifest_hash(variant, manifest):
    parquet_path = SNAPSHOT_DIR / f"{variant}.parquet"
    assert _sha256_file(parquet_path) == manifest[variant]["sha256"]


@pytest.mark.parametrize("variant", VALID_VARIANTS)
def test_load_ett_row_count_and_date_range_match_manifest(variant, manifest):
    df = load_ett(variant)
    entry = manifest[variant]

    assert len(df) == entry["row_count"]
    assert df["date"].min().isoformat() == entry["date_range"][0]
    assert df["date"].max().isoformat() == entry["date_range"][1]


@pytest.mark.parametrize("variant", VALID_VARIANTS)
def test_load_ett_has_expected_columns(variant):
    df = load_ett(variant)
    assert list(df.columns) == ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def test_load_ett_rejects_unknown_variant():
    with pytest.raises(ValueError):
        load_ett("ETTh3")


@pytest.mark.disable_socket
@pytest.mark.parametrize("variant", VALID_VARIANTS)
def test_load_ett_performs_no_network_io(variant):
    df = load_ett(variant)
    assert not df.empty
