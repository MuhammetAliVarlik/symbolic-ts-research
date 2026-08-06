"""Loader for the frozen finance (equities) dataset snapshot.

Reads exclusively from local Parquet snapshots under
``datasets/snapshots/finance/``. Performs no network I/O — to refresh the
snapshot, run ``scripts/fetch_finance.py`` by hand.

Missing/holiday handling: non-trading days (weekends, market holidays) are
simply absent from the source data — there is no continuous calendar to
reindex against, and gaps are never forward-filled or interpolated. Any other
missing values returned by the upstream provider were dropped at freeze time;
the per-ticker drop count is recorded in ``MANIFEST.json`` under
``finance.<TICKER>.dropped_rows`` for transparency.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "finance"

VALID_TICKERS = ("AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "SPY")


def load_finance(tickers: list[str]) -> pd.DataFrame:
    """Load one or more frozen finance tickers from the local snapshot.

    Args:
        tickers: Subset of ``VALID_TICKERS`` to load.

    Returns:
        Long-format DataFrame with columns
        ``date, ticker, Open, High, Low, Close, Adj Close, Volume``,
        sorted by ticker then date.

    Raises:
        ValueError: If ``tickers`` is empty or contains an unknown ticker.
        FileNotFoundError: If a requested ticker's snapshot hasn't been frozen.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list")

    unknown = sorted(set(tickers) - set(VALID_TICKERS))
    if unknown:
        raise ValueError(f"Unknown ticker(s) {unknown}; expected one of {VALID_TICKERS}")

    frames = []
    for ticker in tickers:
        path = _SNAPSHOT_DIR / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No frozen snapshot at {path}. Run scripts/fetch_finance.py to create it."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        df.insert(1, "ticker", ticker)
        frames.append(df)

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
