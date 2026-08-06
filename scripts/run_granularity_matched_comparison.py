"""F0-07 diagnostic: is the cross-domain divergence a sampling-frequency confound?

ETT is hourly/15-min; finance is daily. A lag-1 transition therefore represents
a very different elapsed physical time in each domain, which could produce
exactly the kind of divergence F0-07 found (ETT's short sampling interval
gives adjacent readings more physical continuity) independent of anything
about the tokenisation design itself.

This resamples ETT's OT reading to one observation per calendar day -- the
LAST reading of each day (`resample("D").last()`), mirroring how finance's
"Close" is an end-of-day snapshot, not an average -- then re-runs the exact
same tokenisation (`prototype_tokenize.project_channels`/`tokenize`, domain
still "ett" so it keeps a plain difference, not a log return -- OT has no
multiplicative-growth problem regardless of sampling frequency) and the exact
same transition-matrix + permutation-null pipeline as F0-06/F0-07, now with
both domains' rolling-volatility window (20) spanning 20 DAYS on both sides
instead of 20 hours/20 quarter-hours vs. 20 days.

This is a diagnostic, not a replacement: writes to new `*_daily` result files
only. The original F0-06/F0-07 outputs are untouched.

Run:
    python scripts/run_granularity_matched_comparison.py
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

from prototype_tokenize import BASE_READING, project_channels, tokenize
from run_transition_matrices import ALL_TOKENS, MIN_OBSERVATIONS
from run_transition_similarity import (
    N_PERMUTATIONS,
    RNG_SEED,
    common_defined_rows,
    counts_from_index_arrays,
    frobenius_norm,
    jeffreys_smoothed,
    row_normalise,
    row_wise_kl,
)
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
TOKEN_TO_IDX = {token: i for i, token in enumerate(ALL_TOKENS)}
N_TOKENS = len(ALL_TOKENS)


def resample_ett_daily(variant: str) -> pd.Series:
    """One reading per calendar day: the last one, matching finance's
    end-of-day "Close" semantics rather than an intra-day average."""
    df = load_ett(variant).set_index("date")
    return df[BASE_READING["ett"]].resample("D").last().dropna()


def tokens_to_index_array(domain: str, base_reading: pd.Series) -> np.ndarray:
    channels = project_channels(domain, base_reading)
    tokens, _vol_labels, _change_fit, _vol_fit = tokenize(channels)
    return np.array([TOKEN_TO_IDX[t] for t in tokens], dtype=np.int64)


def marginal_from_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Overall token frequency (every position, including each series' last
    token) -- matches how F0-05's marginal was built."""
    counts = np.zeros(N_TOKENS, dtype=np.int64)
    for arr in arrays:
        counts += np.bincount(arr, minlength=N_TOKENS)
    return counts / counts.sum()


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("resampling ETT to daily (last reading of each calendar day)...")
    ett_daily_arrays = []
    for variant in ETT_VARIANTS:
        daily = resample_ett_daily(variant)
        arr = tokens_to_index_array("ett", daily)
        ett_daily_arrays.append(arr)
        print(f"  {variant}: {len(daily)} daily readings -> {len(arr)} tokens")

    finance_arrays = [
        tokens_to_index_array("finance", load_finance([t]).set_index("date")[BASE_READING["finance"]])
        for t in FINANCE_TICKERS
    ]
    total_ett_daily = sum(len(a) for a in ett_daily_arrays)
    total_finance = sum(len(a) for a in finance_arrays)
    print(f"\ntotal tokens: ett_daily={total_ett_daily}, finance={total_finance}")

    ett_counts = counts_from_index_arrays(ett_daily_arrays)
    finance_counts = counts_from_index_arrays(finance_arrays)
    ett_probs, ett_totals = row_normalise(ett_counts)
    finance_probs, finance_totals = row_normalise(finance_counts)

    ett_matrix_df = pd.DataFrame(ett_probs, index=ALL_TOKENS, columns=ALL_TOKENS)
    ett_matrix_df.to_csv(RESULTS_DIR / "transition_matrix_ett_daily.csv")
    print(f"wrote {RESULTS_DIR / 'transition_matrix_ett_daily.csv'}")

    n_undefined = int((ett_totals == 0).sum())
    n_sparse = int(((ett_totals > 0) & (ett_totals < MIN_OBSERVATIONS)).sum())
    n_sufficient = int((ett_totals >= MIN_OBSERVATIONS).sum())
    print(
        f"ett_daily row sufficiency: {n_undefined} undefined, {n_sparse} sparse "
        f"(1-{MIN_OBSERVATIONS - 1} obs), {n_sufficient} sufficient (>={MIN_OBSERVATIONS})"
    )

    common_rows = common_defined_rows(ett_totals, finance_totals)
    excluded = [ALL_TOKENS[i] for i in range(N_TOKENS) if not common_rows[i]]
    print(f"rows excluded (undefined in at least one domain): {excluded}")
    print(f"comparing {common_rows.sum()} / {N_TOKENS} rows\n")

    # -- observed statistics -------------------------------------------------
    observed_frobenius = frobenius_norm(ett_probs, finance_probs, common_rows)
    ett_smoothed = jeffreys_smoothed(ett_counts)
    finance_smoothed = jeffreys_smoothed(finance_counts)
    observed_kl_ef = row_wise_kl(ett_smoothed, finance_smoothed, common_rows)
    observed_kl_fe = row_wise_kl(finance_smoothed, ett_smoothed, common_rows)
    observed_kl_mean = (observed_kl_ef + observed_kl_fe) / 2

    print(f"observed Frobenius norm (granularity-matched, {common_rows.sum()} common rows): {observed_frobenius:.4f}")
    print(f"observed symmetrised KL: {observed_kl_mean:.4f} bits\n")

    # -- permutation null ------------------------------------------------------
    null_frobenius = np.empty(N_PERMUTATIONS)
    null_kl_mean = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuf_ett = [rng.permutation(arr) for arr in ett_daily_arrays]
        shuf_finance = [rng.permutation(arr) for arr in finance_arrays]

        shuf_ett_counts = counts_from_index_arrays(shuf_ett)
        shuf_finance_counts = counts_from_index_arrays(shuf_finance)
        shuf_ett_probs, shuf_ett_totals = row_normalise(shuf_ett_counts)
        shuf_finance_probs, shuf_finance_totals = row_normalise(shuf_finance_counts)

        shuf_common = common_defined_rows(shuf_ett_totals, shuf_finance_totals)
        if shuf_common.sum() < common_rows.sum():
            shuf_common = shuf_common & common_rows

        null_frobenius[i] = frobenius_norm(shuf_ett_probs, shuf_finance_probs, shuf_common)

        shuf_ett_smoothed = jeffreys_smoothed(shuf_ett_counts)
        shuf_finance_smoothed = jeffreys_smoothed(shuf_finance_counts)
        kl_ef = row_wise_kl(shuf_ett_smoothed, shuf_finance_smoothed, shuf_common)
        kl_fe = row_wise_kl(shuf_finance_smoothed, shuf_ett_smoothed, shuf_common)
        null_kl_mean[i] = (kl_ef + kl_fe) / 2

    percentile_frobenius = float((null_frobenius <= observed_frobenius).mean() * 100)
    percentile_kl = float((null_kl_mean <= observed_kl_mean).mean() * 100)

    print(f"permutation null (n={N_PERMUTATIONS}):")
    print(
        f"  Frobenius norm: null mean={null_frobenius.mean():.4f} (sd {null_frobenius.std():.4f}), "
        f"observed={observed_frobenius:.4f}, percentile={percentile_frobenius:.2f}"
    )
    print(
        f"  symmetrised KL: null mean={null_kl_mean.mean():.4f} (sd {null_kl_mean.std():.4f}), "
        f"observed={observed_kl_mean:.4f}, percentile={percentile_kl:.2f}"
    )

    summary = pd.DataFrame(
        {
            "statistic": ["frobenius_norm", "symmetrised_kl_bits"],
            "observed": [observed_frobenius, observed_kl_mean],
            "null_mean": [null_frobenius.mean(), null_kl_mean.mean()],
            "null_std": [null_frobenius.std(), null_kl_mean.std()],
            "observed_percentile": [percentile_frobenius, percentile_kl],
            "n_permutations": [N_PERMUTATIONS, N_PERMUTATIONS],
        }
    )
    summary.to_csv(RESULTS_DIR / "transition_similarity_summary_daily.csv", index=False)
    pd.DataFrame({"frobenius_norm": null_frobenius, "symmetrised_kl_bits": null_kl_mean}).to_csv(
        RESULTS_DIR / "transition_similarity_null_daily.csv", index=False
    )
    print(f"\nwrote transition_similarity_summary_daily.csv and transition_similarity_null_daily.csv")

    # -- marginal-deviation correlation check (was r=0.855 at native granularity) --
    ett_marginal = marginal_from_arrays(ett_daily_arrays)
    finance_marginal = marginal_from_arrays(finance_arrays)

    row_idx = np.where(common_rows)[0]
    ett_dev = (ett_probs[row_idx] - ett_marginal[None, :]).flatten()
    finance_dev = (finance_probs[row_idx] - finance_marginal[None, :]).flatten()
    correlation = float(np.corrcoef(ett_dev, finance_dev)[0, 1])
    print(f"\nmarginal-deviation correlation (granularity-matched): r = {correlation:.4f}")
    print("(native-granularity comparison from F0-07 was r = 0.8554)")


if __name__ == "__main__":
    main()
