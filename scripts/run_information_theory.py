"""Entropy rate and mutual information at lags 1-10.

Entropy rate estimator: plug-in (maximum-likelihood) entropy rate under a
first-order Markov assumption -- the empirical-P(from-state)-weighted average
of each row's own Shannon entropy, i.e. H = sum_x P(x) * H(Y|X=x). This is the
standard MLE entropy-rate estimator for a first-order chain and it is known to
be NEGATIVILY BIASED in finite samples: finite-sample fluctuations make an
estimated distribution look more peaked (lower entropy) than the true one,
worse for rows with fewer observations. Reported alongside the Miller-Madow
correction (H_MM = H_plugin + (K-1)/(2*n*ln2) bits per row, K=25 possible
symbols, n=row's own observation count), the standard first-order bias
correction for plug-in entropy estimates, applied per row before weighting.

Lag-k mutual information: for k=1..10, the plug-in estimate of I(X_t; X_{t+k})
from the joint co-occurrence distribution of tokens k steps apart (within a
series only -- never crossing a series boundary). No smoothing needed: MI's
log ratio is only ever evaluated where the joint count is already positive,
which makes both its own marginals positive by construction.

Lag-1 MI must reproduce F0-06's row-vs-marginal KL numbers exactly (same
quantity, two different computational paths -- row-conditional-KL here vs.
joint-count-based there) as a consistency check; checked before extending to
lags 2-10, per F0-06's own note.

Run on native-granularity ETT, finance, and granularity-matched (daily-
resampled) ETT, so the three stay directly comparable to F0-07's outputs.

Run:
    python scripts/run_information_theory.py
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

from prototype_tokenize import BASE_READING
from run_transition_matrices import ALL_TOKENS
from run_transition_similarity import counts_from_index_arrays
from run_granularity_matched_comparison import resample_ett_daily, tokens_to_index_array
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
N_TOKENS = len(ALL_TOKENS)
MAX_LAG = 10

# F0-06's row-vs-marginal weighted-mean-KL numbers, native granularity --
# mathematically the same quantity as lag-1 MI computed here, via a
# different computational path. Must match to floating-point tolerance.
F0_06_LAG1_MI = {"ett": 1.7448, "finance": 1.4964}
CROSS_CHECK_TOLERANCE = 1e-3


def entropy_rate(counts: np.ndarray) -> tuple[float, float]:
    """(plug-in, Miller-Madow-corrected) entropy rate in bits, from a 25x25
    transition count matrix. Miller-Madow applied per row (K=25 categories,
    n=that row's own observation count) before the P(from)-weighted average."""
    row_totals = counts.sum(axis=1)
    total = row_totals.sum()

    plugin_bits = 0.0
    mm_bits = 0.0
    for i in range(N_TOKENS):
        n = row_totals[i]
        if n == 0:
            continue
        p_row = counts[i] / n
        nonzero = p_row > 0
        row_entropy = -np.sum(p_row[nonzero] * np.log2(p_row[nonzero]))
        mm_correction = (N_TOKENS - 1) / (2 * n * np.log(2))

        weight = n / total
        plugin_bits += weight * row_entropy
        mm_bits += weight * (row_entropy + mm_correction)

    return float(plugin_bits), float(mm_bits)


def lag_k_mutual_information(arrays: list[np.ndarray], lag: int) -> float:
    joint_counts = np.zeros((N_TOKENS, N_TOKENS), dtype=np.int64)
    for arr in arrays:
        if len(arr) <= lag:
            continue
        x, y = arr[:-lag], arr[lag:]
        flat_idx = x * N_TOKENS + y
        joint_counts += np.bincount(flat_idx, minlength=N_TOKENS * N_TOKENS).reshape(N_TOKENS, N_TOKENS)

    total = joint_counts.sum()
    p_xy = joint_counts / total
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    nonzero = p_xy > 0
    rows, cols = np.nonzero(nonzero)
    mi = np.sum(p_xy[rows, cols] * np.log2(p_xy[rows, cols] / (p_x[rows] * p_y[cols])))
    return float(mi)


def domain_arrays(domain: str) -> list[np.ndarray]:
    if domain == "ett":
        return [
            tokens_to_index_array("ett", load_ett(v).set_index("date")[BASE_READING["ett"]]) for v in ETT_VARIANTS
        ]
    if domain == "ett_daily":
        return [tokens_to_index_array("ett", resample_ett_daily(v)) for v in ETT_VARIANTS]
    if domain == "finance":
        return [
            tokens_to_index_array("finance", load_finance([t]).set_index("date")[BASE_READING["finance"]])
            for t in FINANCE_TICKERS
        ]
    raise ValueError(domain)


def main() -> None:
    domains = ["ett", "finance", "ett_daily"]
    arrays_by_domain = {d: domain_arrays(d) for d in domains}

    rows = []

    print("entropy rate (bits):")
    for domain in domains:
        counts = counts_from_index_arrays(arrays_by_domain[domain])
        plugin, mm = entropy_rate(counts)
        bias = mm - plugin
        print(f"  {domain}: plug-in={plugin:.4f}, Miller-Madow-corrected={mm:.4f}, bias={bias:.4f}")
        rows.append({"domain": domain, "metric": "entropy_rate_plugin", "lag": None, "value_bits": plugin})
        rows.append({"domain": domain, "metric": "entropy_rate_miller_madow", "lag": None, "value_bits": mm})

    print("\nlag-1 mutual information cross-check against F0-06:")
    for domain in ["ett", "finance"]:
        lag1_mi = lag_k_mutual_information(arrays_by_domain[domain], lag=1)
        f0_06_value = F0_06_LAG1_MI[domain]
        diff = abs(lag1_mi - f0_06_value)
        status = "MATCH" if diff < CROSS_CHECK_TOLERANCE else "MISMATCH"
        print(f"  {domain}: this script={lag1_mi:.4f}, F0-06={f0_06_value:.4f}, diff={diff:.6f} -> {status}")
        if status == "MISMATCH":
            raise RuntimeError(
                f"lag-1 MI cross-check failed for {domain}: {lag1_mi:.4f} vs F0-06's {f0_06_value:.4f}. "
                "This is a bug to trace, not a discrepancy to average over -- see F0-06's note."
            )

    print("\nmutual information at lags 1-10 (bits):")
    for domain in domains:
        for lag in range(1, MAX_LAG + 1):
            mi = lag_k_mutual_information(arrays_by_domain[domain], lag)
            rows.append({"domain": domain, "metric": "mutual_information", "lag": lag, "value_bits": mi})
        lags_str = ", ".join(
            f"{lag}:{r['value_bits']:.3f}" for lag, r in zip(range(1, MAX_LAG + 1), rows[-MAX_LAG:])
        )
        print(f"  {domain}: {lags_str}")

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "information_theory.csv"
    df.to_csv(output_path, index=False)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
