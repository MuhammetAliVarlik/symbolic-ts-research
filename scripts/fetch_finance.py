"""One-time network fetch that freezes the finance dataset as local Parquet snapshots.

Run by hand whenever the frozen snapshot needs to be (re)created:

    python scripts/fetch_finance.py

This script is the ONLY place in the finance pipeline allowed to touch the network.
`datasets/finance.py` never imports `yfinance`/`requests` and reads exclusively from
the Parquet files this script produces.

Source: Stooq's scripted CSV export is currently behind a JS proof-of-work
challenge and `pandas_datareader`'s Stooq reader 404s against the live site, so
this freezes from Yahoo Finance via `yfinance` instead (the same fallback
FinwiseBackend's `stock_service.py` already relies on when Stooq is unavailable).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "datasets" / "snapshots" / "finance"
MANIFEST_PATH = REPO_ROOT / "datasets" / "snapshots" / "MANIFEST.json"

# Sector-diverse large-cap set plus one broad index ETF, chosen for >=10 years of
# clean daily history: tech (AAPL, MSFT), financials (JPM), energy (XOM),
# healthcare (JNJ), consumer/retail (WMT), broad market (SPY).
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "SPY"]
START_DATE = "2014-01-01"  # comfortably >= 10 years of history as of the freeze date


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [str(col[0]) for col in cleaned.columns]
    cleaned = cleaned.reset_index().rename(columns={"Date": "date"})
    expected = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cleaned = cleaned[expected]

    # Non-trading days (weekends, market holidays) are simply absent from the
    # source entirely -- there is no calendar to reindex against. Any other
    # missing values (rare provider gaps) are dropped rather than filled, so the
    # frozen snapshot never contains synthesized prices.
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    dropped = before - len(cleaned)

    return cleaned.sort_values("date").reset_index(drop=True), dropped


def fetch_ticker(ticker: str) -> dict:
    raw = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    df, dropped_rows = _normalize_ohlcv(raw)

    parquet_path = SNAPSHOT_DIR / f"{ticker}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    return {
        "sha256": _sha256_file(parquet_path),
        "row_count": int(len(df)),
        "date_range": [df["date"].min().isoformat(), df["date"].max().isoformat()],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "dropped_rows": int(dropped_rows),
    }


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    manifest.setdefault("finance", {})

    for ticker in TICKERS:
        print(f"fetching {ticker} ...")
        manifest["finance"][ticker] = fetch_ticker(ticker)
        entry = manifest["finance"][ticker]
        print(f"  -> {entry['row_count']} rows, sha256={entry['sha256'][:12]}..., "
              f"dropped={entry['dropped_rows']}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
