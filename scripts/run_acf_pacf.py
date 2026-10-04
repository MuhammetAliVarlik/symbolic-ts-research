"""ACF/PACF of token sequences and context-window justification.

A compound token (e.g. `C2_V1`) has no single natural total ordering, so it is
mapped to TWO ordinal scales, analysed separately, rather than collapsed into
one artificial combined number: the change label (0..4, most-negative to
most-positive) and the volatility label (0..4, lowest to highest). Each is a
genuinely ordinal quantity on its own; a from combining them (e.g. c*5+v)
would not be.

ACF uses Bartlett's formula for the confidence bound (accounts for the prior
autocorrelations already estimated, not just the sample size); PACF uses the
conventional large-sample bound (+-1.96/sqrt(N)), which is the standard pairing
in the literature -- Bartlett's formula is specifically an ACF result.

Per-series ordinal sequences are concatenated within a domain before computing
ACF/PACF (not kept series-separate and averaged). This introduces a handful of
boundary artifacts (one per series -- 7 for finance, 4 for ett/ett_daily) into
sequences of thousands to hundreds of thousands of points; unlike the
transition-matrix/mutual-information work (F0-06/F0-08), where a single
mislabelled transition directly pollutes one specific matrix cell, ACF/PACF
are correlation estimates over the whole series and are far more robust to a
handful of boundary points. Documented here as a simplification, not hidden.

Lags computed out to 300 -- F0-08 found lag-10 mutual information still
clearly nonzero (0.49-0.59 bits) in every domain, so stopping at 10 would not
find where autocorrelation actually decays. (A first pass at 60 left native
`ett`'s change and volatility, and finance's volatility, never stably inside
the Bartlett bound -- extended rather than reported as "undefined".)

`ett_hourly`/`ett_15min` are kept as two SEPARATE domains here rather than
pooled into one "ett" -- an earlier pooled pass produced a curve that looked
periodic but was actually a mix of the hourly variants' slower decay and the
15-min variants' genuine ~daily oscillation, an artifact of concatenating two
different sampling frequencies (lag k means different physical time in each),
the same issue F0-07 found for cross-domain comparison.

This script reports the raw "first lag stably inside the Bartlett bound" per
series (`acf_pacf_decorrelation_lags.csv`), but does NOT try to automatically
classify a series as "genuinely decayed" vs. "periodic, coincidentally dipped
inside the bound" -- two increasingly elaborate automated heuristics for this
were tried and both failed (one flagged nothing as periodic including the
15-min series' obvious ~100-lag oscillation; a stricter version flagged
EVERY series, including finance's near-immediate clean decorrelation, since
with 300 lags tested even pure noise produces occasional short runs outside a
95% band). That classification is made in `experiments/notes/F0-09.md` from
directly reading the ACF curves (Figure A8 / `acf_pacf.csv`), which is more
reliable than a brittle automatic rule for a judgment call like this one.

Run:
    python scripts/run_acf_pacf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
from symbolic_ts.stationarity import Correlogram, acf_with_bounds, pacf_with_bounds

from prototype_tokenize import BASE_READING, project_channels, tokenize
from run_granularity_matched_comparison import resample_ett_daily
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
MAX_LAG = 300
ALPHA = 0.05  # 95% confidence bounds


def domain_ordinal_series(domain: str) -> dict[str, np.ndarray]:
    """Concatenated (change_label, volatility_label) ordinal series (0..4 each)
    for one domain, pooled across all its series.

    ETT's hourly (ETTh1/h2) and 15-min (ETTm1/m2) variants are kept SEPARATE
    here, not pooled into one "ett" -- lag k means a different amount of
    physical time in each, exactly the granularity issue F0-07 already found
    for cross-domain comparison. Pooling them for ACF specifically produced a
    misleading blended curve dominated by the (4x longer) 15-min variants.
    """
    if domain == "ett_hourly":
        readings = [load_ett(v).set_index("date")[BASE_READING["ett"]] for v in ["ETTh1", "ETTh2"]]
        proj_domain = "ett"
    elif domain == "ett_15min":
        readings = [load_ett(v).set_index("date")[BASE_READING["ett"]] for v in ["ETTm1", "ETTm2"]]
        proj_domain = "ett"
    elif domain == "ett_daily":
        readings = [resample_ett_daily(v) for v in ETT_VARIANTS]
        proj_domain = "ett"
    elif domain == "finance":
        readings = [load_finance([t]).set_index("date")[BASE_READING["finance"]] for t in FINANCE_TICKERS]
        proj_domain = "finance"
    else:
        raise ValueError(domain)

    change_labels: list[str] = []
    vol_labels: list[str] = []
    for reading in readings:
        channels = project_channels(proj_domain, reading)
        tokens, series_vol_labels, _change_fit, _vol_fit = tokenize(channels)
        change_labels.extend(t.split("_")[0][1:] for t in tokens)  # "C2_V1" -> "2"
        vol_labels.extend(series_vol_labels.tolist())

    return {
        "change": np.array(change_labels, dtype=int),
        "volatility": np.array(vol_labels, dtype=int),
    }


def acf_pacf_with_bounds(series: np.ndarray, max_lag: int) -> tuple[pd.DataFrame, Correlogram]:
    """ACF (Bartlett bound) and PACF (white-noise bound) from `symbolic_ts.stationarity`,
    side by side in one table. Lag 0 is dropped by the library -- it is always 1."""
    acf_result = acf_with_bounds(series, nlags=max_lag, alpha=ALPHA)
    pacf_result = pacf_with_bounds(series, nlags=max_lag, alpha=ALPHA)
    df = pd.DataFrame(
        {
            "lag": acf_result.lags,
            "acf": acf_result.values,
            "acf_bartlett_bound": acf_result.bounds,
            "pacf": pacf_result.values,
            "pacf_bound": pacf_result.bounds,
        }
    )
    return df, acf_result


def main() -> None:
    domains = ["ett_hourly", "ett_15min", "finance", "ett_daily"]
    all_rows = []
    decorrelation_lags = {}

    for domain in domains:
        series = domain_ordinal_series(domain)
        for channel, values in series.items():
            df, acf_result = acf_pacf_with_bounds(values, MAX_LAG)
            df.insert(0, "channel", channel)
            df.insert(0, "domain", domain)
            all_rows.append(df)

            # First lag stably (3 in a row) inside the Bartlett bound: a raw
            # signal only -- see the module docstring for why "decayed" vs.
            # "periodic, coincidentally dipped" is judged from the curve instead.
            first = acf_result.first_lag_within_bounds(consecutive=3)
            lag0 = -1 if first is None else first  # -1 = never within bounds up to MAX_LAG
            decorrelation_lags[(domain, channel)] = lag0
            print(f"{domain} / {channel}: first lag with 3-consecutive-lags inside Bartlett bound = {lag0}")

    combined = pd.concat(all_rows, ignore_index=True)
    output_path = RESULTS_DIR / "acf_pacf.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nwrote {output_path}")

    # Only clean (non-oscillating) decorrelation lags feed the window
    # Classification by direct visual/quantitative inspection of the curves
    # (see experiments/notes/F0-09.md) -- not automated, per the module
    # docstring. "oscillating" = genuine periodic re-emergence (ETT's native
    # hourly/15-min series, consistent with real daily seasonality);
    # "clean" = decays toward zero without sustained re-emergence.
    oscillating = {
        ("ett_hourly", "change"): True,
        ("ett_hourly", "volatility"): False,  # never stabilises, but no periodic re-emergence either -- slow/long-memory decay, not oscillation
        ("ett_15min", "change"): True,
        ("ett_15min", "volatility"): True,
        ("finance", "change"): False,
        ("finance", "volatility"): False,
        ("ett_daily", "change"): False,
        ("ett_daily", "volatility"): False,
    }

    clean_lags = [lag for key, lag in decorrelation_lags.items() if lag > 0 and not oscillating[key]]
    recommended_window = max(clean_lags) if clean_lags else MAX_LAG
    print(f"\ndecorrelation lags: {decorrelation_lags}")
    print(f"classification (oscillating, by inspection): {oscillating}")
    print(f"recommended context window (max clean decorrelation lag): {recommended_window}")

    summary = pd.DataFrame(
        [
            {"domain": d, "channel": c, "decorrelation_lag": lag, "oscillating": oscillating[(d, c)]}
            for (d, c), lag in decorrelation_lags.items()
        ]
    )
    summary.to_csv(RESULTS_DIR / "acf_pacf_decorrelation_lags.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'acf_pacf_decorrelation_lags.csv'}")


if __name__ == "__main__":
    main()
