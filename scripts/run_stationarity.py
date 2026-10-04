"""ADF/KPSS stationarity testing across domains (F0-03).

The tests themselves come from `symbolic_ts.stationarity` (F1-08), whose
defaults are exactly the configuration below -- this script only loads the
data and writes the table. Run with:

    python scripts/run_stationarity.py

Writes `results/stationarity.csv`. Every channel of every domain is tested on both
its raw series and its first difference (`series.diff().dropna()`); no log
transform, scaling, or seasonal adjustment is applied at this stage.

ADF (`statsmodels.tsa.stattools.adfuller`, autolag="AIC", regression="c"):
    H0 = the series has a unit root (non-stationary).
KPSS (`statsmodels.tsa.stattools.kpss`, regression="c", nlags="auto"):
    H0 = the series is (trend-)stationary.

KPSS p-values are looked up from a table bounded to [0.01, 0.1]; when the true
p-value falls outside that range, statsmodels clips it. The library flags that
(`p_value_clipped`) instead of warning; the clipped p-value is recorded as-is.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from symbolic_ts.stationarity import StationarityTestResult, adf_test, kpss_test

from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_PATH = REPO_ROOT / "results" / "stationarity.csv"

ETT_CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
FINANCE_CHANNELS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

ALPHA = 0.05


def _row(result: StationarityTestResult) -> dict:
    return {
        "test": result.test,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "lags_used": result.lags,
        "n_obs": result.n_obs,
        "reject_null_5pct": result.reject_null,
        "conclusion": "stationary" if result.is_stationary else "non-stationary",
    }


def _rows_for_series(domain: str, series_id: str, channel: str, raw: pd.Series) -> list[dict]:
    raw = raw.dropna()
    differenced = raw.diff().dropna()

    rows = []
    for transform, series in [("raw", raw), ("differenced", differenced)]:
        for result in [adf_test(series, alpha=ALPHA), kpss_test(series, alpha=ALPHA)]:
            rows.append(
                {
                    "domain": domain,
                    "series_id": series_id,
                    "channel": channel,
                    "transform": transform,
                    **_row(result),
                }
            )
    return rows


def collect_ett_rows() -> list[dict]:
    rows = []
    for variant in ETT_VARIANTS:
        df = load_ett(variant)
        for channel in ETT_CHANNELS:
            print(f"  ETT {variant} / {channel}")
            rows.extend(_rows_for_series("ett", variant, channel, df[channel]))
    return rows


def collect_finance_rows() -> list[dict]:
    rows = []
    for ticker in FINANCE_TICKERS:
        df = load_finance([ticker])
        for channel in FINANCE_CHANNELS:
            print(f"  finance {ticker} / {channel}")
            rows.extend(_rows_for_series("finance", ticker, channel, df[channel]))
    return rows


def main() -> None:
    print("running ETT stationarity tests...")
    rows = collect_ett_rows()
    print("running finance stationarity tests...")
    rows.extend(collect_finance_rows())

    df = pd.DataFrame(rows)
    df = df.sort_values(["domain", "series_id", "channel", "transform", "test"]).reset_index(drop=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"wrote {len(df)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
