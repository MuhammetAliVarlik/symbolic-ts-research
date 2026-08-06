"""Prototype: channel projection (change/volatility) + sigma-based quantisation
into a token sequence, for both domains, from the frozen snapshots.

Script-level prototype only, not the library -- the goal is to learn what the
real design should be before it gets committed to a tested module. See
experiments/notes/F0-04.md for what worked and what should change.

Two design decisions, both revised after the first pass over real data:

1. `level` is intentionally NOT a projected output channel (only two channels are
   produced: change, volatility, not three). It's still used internally as the
   base reading each change is computed from, but it is never tokenised. Reason:
   a raw level is non-stationary, so a sigma threshold fit on one region of a
   growing/drifting series doesn't describe another region of the same series --
   exactly the failure mode this script is designed to avoid. Cumulative change
   already carries the same information a coarse discretised level would.

2. Finance's `change` is a log return (`log(level).diff()`), not a raw
   difference. A raw dollar difference doesn't hold up over a 12-year window for
   a stock whose price itself grew multiple times over (see notes for the AAPL
   evidence) -- the training-fit sigma ends up calibrated to the wrong scale for
   whichever price regime dominates the training slice. ETT's `OT` reading has no
   equivalent multiplicative-growth problem, so it keeps a plain difference. Log
   returns also keep this consistent with the GARCH baseline and Diebold-Mariano,
   both of which assume log returns.

3. Volatility (a rolling std, non-negative and right-skewed) is bucketed on
   log(volatility), not on the raw value. Symmetric sigma bins on the raw value
   left the bottom bucket almost empty (~2% fill) with over half of everything
   landing in one middle bucket; log(volatility) bucket-fill is close to
   symmetric across all 5 buckets. Same sigma mechanism either way, just applied
   after a log transform -- not a switch to quantile bins.

   ETT's rolling std also hits exact zero in ~0.3-2% of windows (flat sensor
   stretches), and log(0) is undefined. The fix needed two parts, not one: a
   small epsilon before the log so classification doesn't crash, AND excluding
   those same zero-volatility windows from the sigma *fit* itself. The first
   fix alone is not enough -- on ETTm1/ETTm2, leaving the epsilon-floor values
   (log(1e-8) =~ -18.4) inside the fit inflated the fitted std by ~4.5x from
   just 1.6-2.1% of rows being that extreme. Excluding them from the fit but
   still classifying them (they land in the lowest bucket automatically, which
   is the correct place for genuinely-zero local volatility) avoids that.

Run:
    python scripts/prototype_tokenize.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

# Naive prototype split -- just the first N% of each series. No purge/embargo,
# no walk-forward folds. Good enough to prove sigma is fit train-only; the real
# splitting logic is its own separate piece of work.
TRAIN_FRACTION = 0.7

# Rolling window for the volatility channel. Picked to match the kind of
# short-window realised-volatility figure a trader would already look at.
# Left as-is for now; revisit once the ACF/PACF lag structure is known.
VOLATILITY_WINDOW = 20

# Sigma bucket edges, in units of training-fit standard deviations either side
# of the training-fit mean: [-inf, -1.5, -0.5, 0.5, 1.5, +inf] -> 5 buckets.
SIGMA_EDGES = [-1.5, -0.5, 0.5, 1.5]
LABELS = ["0", "1", "2", "3", "4"]  # neutral bucket indices, 0 = lowest .. 4 = highest

# The raw reading each domain's `change` channel is derived from. Not itself a
# projected output channel -- see module docstring, point 1.
BASE_READING = {"ett": "OT", "finance": "Adj Close"}


@dataclass
class SigmaFit:
    mean: float
    std: float

    def bucket(self, values: pd.Series) -> pd.Series:
        z = (values - self.mean) / self.std
        idx = np.digitize(z, SIGMA_EDGES)  # 0..4
        return pd.Series([LABELS[i] for i in idx], index=values.index)


def fit_sigma(train_slice: pd.Series) -> SigmaFit:
    return SigmaFit(mean=float(train_slice.mean()), std=float(train_slice.std()))


def compute_change(domain: str, base_reading: pd.Series) -> pd.Series:
    """Finance: log return. ETT: plain difference. See module docstring, point 2."""
    if domain == "finance":
        return np.log(base_reading).diff()
    return base_reading.diff()


def project_channels(domain: str, base_reading: pd.Series) -> pd.DataFrame:
    """Reduce one raw reading to the two summary channels: change, volatility."""
    change = compute_change(domain, base_reading)
    volatility = change.rolling(VOLATILITY_WINDOW).std()
    return pd.DataFrame({"change": change, "volatility": volatility}).dropna()


def log_volatility(volatility: pd.Series) -> pd.Series:
    """Volatility is non-negative and right-skewed; bucket it on the log scale
    (see module docstring, point 3). ETT's rolling std hits exact zero in
    ~0.3-2% of windows (flat sensor stretches); log(0) is undefined, so a tiny
    epsilon goes in first. Those windows correctly fall into the lowest bucket
    -- genuinely zero local volatility is the extreme low end, not an error case.
    """
    return np.log(volatility + 1e-8)


def tokenize(channels: pd.DataFrame) -> tuple[list[str], pd.Series, SigmaFit, SigmaFit]:
    split = int(len(channels) * TRAIN_FRACTION)
    train = channels.iloc[:split]

    # Zero-volatility windows are degenerate (log(0) undefined) and get an
    # epsilon floor for classification, but must NOT be part of the sigma fit
    # itself -- see module docstring, point 3. Tolerance-based rather than
    # `== 0.0` / `> 0`: a rolling std over near-constant-but-not-bitwise-identical
    # values can land on something like 1e-15 instead of exactly zero, which
    # would slip past a strict equality check and reintroduce the same
    # fit-inflation problem the exclusion is meant to prevent.
    is_zero_volatility = np.isclose(train["volatility"], 0.0, atol=1e-10)
    train_volatility_nonzero = train["volatility"][~is_zero_volatility]

    change_fit = fit_sigma(train["change"])
    vol_fit = fit_sigma(log_volatility(train_volatility_nonzero))

    change_labels = change_fit.bucket(channels["change"])
    vol_labels = vol_fit.bucket(log_volatility(channels["volatility"]))

    tokens = [f"C{c}_V{v}" for c, v in zip(change_labels, vol_labels)]
    return tokens, vol_labels, change_fit, vol_fit


def run_for_series(domain: str, series_id: str, base_reading: pd.Series, verbose: bool = True) -> list[str]:
    channels = project_channels(domain, base_reading)
    tokens, _vol_labels, change_fit, vol_fit = tokenize(channels)
    if verbose:
        print(
            f"  {domain} {series_id}: {len(tokens)} tokens, "
            f"train-fit change sigma={change_fit.std:.4f}, "
            f"train-fit log-vol sigma={vol_fit.std:.4f}"
        )
        print(f"    sample: {' '.join(tokens[:15])} ...")
    return tokens


def check_determinism(domain: str, series_id: str, base_reading: pd.Series) -> None:
    tokens_a = run_for_series(domain, series_id, base_reading, verbose=False)
    tokens_b = run_for_series(domain, series_id, base_reading, verbose=False)
    assert tokens_a == tokens_b, f"non-deterministic tokenisation for {domain}/{series_id}"
    print(f"  {domain} {series_id}: identical on repeat run ({len(tokens_a)} tokens)")


def report_bucket_fill(domain: str, series_id: str, base_reading: pd.Series) -> pd.Series:
    """Volatility-channel bucket-fill distribution (proportion per label 0..4)."""
    channels = project_channels(domain, base_reading)
    _tokens, vol_labels, _change_fit, _vol_fit = tokenize(channels)
    return vol_labels.value_counts(normalize=True).reindex(LABELS, fill_value=0.0)


def main() -> None:
    print("ETT domain:")
    for variant in ETT_VARIANTS:
        df = load_ett(variant)
        base_reading = df.set_index("date")[BASE_READING["ett"]]
        run_for_series("ett", variant, base_reading)

    print("\nfinance domain:")
    for ticker in FINANCE_TICKERS:
        df = load_finance([ticker])
        base_reading = df.set_index("date")[BASE_READING["finance"]]
        run_for_series("finance", ticker, base_reading)

    print("\ndeterminism check (run each series' tokenisation twice, compare):")
    check_determinism("ett", "ETTh1", load_ett("ETTh1").set_index("date")[BASE_READING["ett"]])
    check_determinism("finance", "AAPL", load_finance(["AAPL"]).set_index("date")[BASE_READING["finance"]])

    print("\nvolatility bucket-fill distribution, pooled across all 11 series (log scale):")
    fills = []
    for variant in ETT_VARIANTS:
        base_reading = load_ett(variant).set_index("date")[BASE_READING["ett"]]
        fills.append(report_bucket_fill("ett", variant, base_reading))
    for ticker in FINANCE_TICKERS:
        base_reading = load_finance([ticker]).set_index("date")[BASE_READING["finance"]]
        fills.append(report_bucket_fill("finance", ticker, base_reading))
    pooled = pd.concat(fills, axis=1).mean(axis=1)
    for label, frac in pooled.items():
        print(f"  bucket {label}: {frac:.1%}")


if __name__ == "__main__":
    main()
