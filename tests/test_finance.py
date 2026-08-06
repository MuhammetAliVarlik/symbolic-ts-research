import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from datasets.finance import VALID_TICKERS, load_finance

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "datasets" / "snapshots" / "finance"
MANIFEST_PATH = REPO_ROOT / "datasets" / "snapshots" / "MANIFEST.json"

MIN_YEARS = 10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())["finance"]


@pytest.mark.parametrize("ticker", VALID_TICKERS)
def test_snapshot_matches_manifest_hash(ticker, manifest):
    parquet_path = SNAPSHOT_DIR / f"{ticker}.parquet"
    assert _sha256_file(parquet_path) == manifest[ticker]["sha256"]


@pytest.mark.parametrize("ticker", VALID_TICKERS)
def test_load_finance_row_count_and_date_range_match_manifest(ticker, manifest):
    df = load_finance([ticker])
    entry = manifest[ticker]

    assert len(df) == entry["row_count"]
    assert df["date"].min().isoformat() == entry["date_range"][0]
    assert df["date"].max().isoformat() == entry["date_range"][1]


@pytest.mark.parametrize("ticker", VALID_TICKERS)
def test_snapshot_covers_at_least_ten_years(ticker, manifest):
    start = datetime.fromisoformat(manifest[ticker]["date_range"][0])
    end = datetime.fromisoformat(manifest[ticker]["date_range"][1])
    assert (end - start).days >= MIN_YEARS * 365


def test_load_finance_has_expected_columns():
    df = load_finance(["AAPL"])
    assert list(df.columns) == ["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def test_load_finance_combines_multiple_tickers_long_format():
    df = load_finance(["AAPL", "MSFT"])
    assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}
    assert list(df.columns[:2]) == ["date", "ticker"]


def test_load_finance_rejects_empty_list():
    with pytest.raises(ValueError):
        load_finance([])


def test_load_finance_rejects_unknown_ticker():
    with pytest.raises(ValueError):
        load_finance(["NOTATICKER"])


@pytest.mark.disable_socket
@pytest.mark.parametrize("ticker", VALID_TICKERS)
def test_load_finance_performs_no_network_io(ticker):
    df = load_finance([ticker])
    assert not df.empty
