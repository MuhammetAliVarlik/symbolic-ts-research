"""Prototype: channel projection (level/change/volatility) + sigma-based
quantisation into a token sequence, for both domains, from the frozen snapshots.

Script-level prototype only, not the library -- the goal is to learn what the
real design should be before it gets committed to a tested module. See
experiments/notes/F0-04.md for what worked and what should change.

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
# short-window realised-volatility figure a trader would already look at;
# untested against alternatives.
VOLATILITY_WINDOW = 20

# Sigma bucket edges, in units of training-fit standard deviations either side
# of the training-fit mean: [-inf, -1.5, -0.5, 0.5, 1.5, +inf] -> 5 buckets.
SIGMA_EDGES = [-1.5, -0.5, 0.5, 1.5]
LABELS = ["0", "1", "2", "3", "4"]  # neutral bucket indices, 0 = lowest .. 4 = highest

# The single reading each domain's forecasting task actually cares about.
# Finance: the adjusted close price. ETT: OT (oil temperature), the designated
# forecast target in the ETT benchmark. Everything else in a raw series is only
# used to get here, not tokenised directly.
LEVEL_CHANNEL = {"ett": "OT", "finance": "Adj Close"}


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


def project_channels(level: pd.Series) -> pd.DataFrame:
    """Reduce one raw reading to the three summary channels: level, change, volatility."""
    change = level.diff()
    volatility = change.rolling(VOLATILITY_WINDOW).std()
    return pd.DataFrame({"level": level, "change": change, "volatility": volatility}).dropna()


def tokenize(channels: pd.DataFrame) -> tuple[list[str], SigmaFit, SigmaFit]:
    split = int(len(channels) * TRAIN_FRACTION)
    train = channels.iloc[:split]

    change_fit = fit_sigma(train["change"])
    vol_fit = fit_sigma(train["volatility"])

    change_labels = change_fit.bucket(channels["change"])
    vol_labels = vol_fit.bucket(channels["volatility"])

    tokens = [f"C{c}_V{v}" for c, v in zip(change_labels, vol_labels)]
    return tokens, change_fit, vol_fit


def run_for_series(domain: str, series_id: str, level: pd.Series, verbose: bool = True) -> list[str]:
    channels = project_channels(level)
    tokens, change_fit, vol_fit = tokenize(channels)
    if verbose:
        print(
            f"  {domain} {series_id}: {len(tokens)} tokens, "
            f"train-fit change sigma={change_fit.std:.4f}, train-fit vol sigma={vol_fit.std:.4f}"
        )
        print(f"    sample: {' '.join(tokens[:15])} ...")
    return tokens


def check_determinism(domain: str, series_id: str, level: pd.Series) -> None:
    tokens_a = run_for_series(domain, series_id, level, verbose=False)
    tokens_b = run_for_series(domain, series_id, level, verbose=False)
    assert tokens_a == tokens_b, f"non-deterministic tokenisation for {domain}/{series_id}"
    print(f"  {domain} {series_id}: identical on repeat run ({len(tokens_a)} tokens)")


def main() -> None:
    print("ETT domain:")
    for variant in ETT_VARIANTS:
        df = load_ett(variant)
        level = df.set_index("date")[LEVEL_CHANNEL["ett"]]
        run_for_series("ett", variant, level)

    print("\nfinance domain:")
    for ticker in FINANCE_TICKERS:
        df = load_finance([ticker])
        level = df.set_index("date")[LEVEL_CHANNEL["finance"]]
        run_for_series("finance", ticker, level)

    print("\ndeterminism check (run each series' tokenisation twice, compare):")
    check_determinism("ett", "ETTh1", load_ett("ETTh1").set_index("date")[LEVEL_CHANNEL["ett"]])
    check_determinism("finance", "AAPL", load_finance(["AAPL"]).set_index("date")[LEVEL_CHANNEL["finance"]])


if __name__ == "__main__":
    main()
