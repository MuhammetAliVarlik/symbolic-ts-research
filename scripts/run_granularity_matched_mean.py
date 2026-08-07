"""F0-11: mean-based daily-ETT resampling robustness check.

F0-07's granularity-matched comparison resampled ETT to daily using
`resample("D").last()` -- the reading at end of each calendar day, matching
finance's "Close" semantics (a snapshot, not an average). A `.mean()`
alternative (average within each day) was deferred at the time: different
aggregation, different noise properties (averaging smooths out intra-day
extremes a last-reading snapshot would preserve), which could shift the
residual 96.8th/80.4th-percentile gap either direction.

This re-runs the exact same tokenisation and Frobenius/KL/permutation-null
pipeline as F0-07's granularity-matched comparison, with ETT resampled by
`.mean()` instead of `.last()`, against the same unchanged real finance data,
and compares explicitly against the `.last()`-based figures. Writes to NEW
result files -- does not touch or replace F0-07's originals.

Run:
    python scripts/run_granularity_matched_mean.py
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
from run_granularity_matched_comparison import tokens_to_index_array
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"

# The .last()-based figures from F0-07, for direct comparison.
LAST_BASED_RESULT = {
    "frobenius_norm": {"observed": 1.0647, "null_mean": 0.9511, "null_std": 0.0581, "percentile": 96.8},
    "symmetrised_kl_bits": {"observed": 0.3605, "null_mean": 0.3422, "null_std": 0.0216, "percentile": 80.4},
}


def resample_ett_daily_mean(variant: str) -> pd.Series:
    """One reading per calendar day: the mean of that day's raw readings,
    as opposed to F0-07's `.last()` (end-of-day snapshot)."""
    df = load_ett(variant).set_index("date")
    return df[BASE_READING["ett"]].resample("D").mean().dropna()


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("resampling ETT to daily using .mean() (F0-07 used .last())...")
    ett_daily_mean_arrays = []
    for variant in ETT_VARIANTS:
        daily = resample_ett_daily_mean(variant)
        arr = tokens_to_index_array("ett", daily)
        ett_daily_mean_arrays.append(arr)
        print(f"  {variant}: {len(daily)} daily readings -> {len(arr)} tokens")

    finance_arrays = [
        tokens_to_index_array("finance", load_finance([t]).set_index("date")[BASE_READING["finance"]])
        for t in FINANCE_TICKERS
    ]
    total_ett_daily_mean = sum(len(a) for a in ett_daily_mean_arrays)
    total_finance = sum(len(a) for a in finance_arrays)
    print(f"\ntotal tokens: ett_daily(mean)={total_ett_daily_mean}, finance={total_finance}")

    ett_counts = counts_from_index_arrays(ett_daily_mean_arrays)
    finance_counts = counts_from_index_arrays(finance_arrays)
    ett_probs, ett_totals = row_normalise(ett_counts)
    finance_probs, finance_totals = row_normalise(finance_counts)

    from run_transition_matrices import ALL_TOKENS
    pd.DataFrame(ett_probs, index=ALL_TOKENS, columns=ALL_TOKENS).to_csv(
        RESULTS_DIR / "transition_matrix_ett_daily_mean.csv"
    )

    common_rows = common_defined_rows(ett_totals, finance_totals)
    excluded = [ALL_TOKENS[i] for i in range(len(ALL_TOKENS)) if not common_rows[i]]
    print(f"rows excluded (undefined in at least one domain): {excluded}")
    print(f"comparing {common_rows.sum()} / {len(ALL_TOKENS)} rows\n")

    # -- observed statistics -------------------------------------------------
    observed_frobenius = frobenius_norm(ett_probs, finance_probs, common_rows)
    ett_smoothed = jeffreys_smoothed(ett_counts)
    finance_smoothed = jeffreys_smoothed(finance_counts)
    observed_kl_ef = row_wise_kl(ett_smoothed, finance_smoothed, common_rows)
    observed_kl_fe = row_wise_kl(finance_smoothed, ett_smoothed, common_rows)
    observed_kl_mean = (observed_kl_ef + observed_kl_fe) / 2

    print(f"observed Frobenius norm (mean-based, {common_rows.sum()} common rows): {observed_frobenius:.4f}")
    print(f"observed symmetrised KL: {observed_kl_mean:.4f} bits\n")

    # -- marginal-deviation correlation (F0-07 reported r=0.898 for .last()) ----
    ett_marginal = ett_counts.sum(axis=1) / ett_counts.sum()
    finance_marginal = finance_counts.sum(axis=1) / finance_counts.sum()
    row_idx = np.where(common_rows)[0]
    ett_dev = (ett_probs[row_idx] - ett_marginal[None, :]).flatten()
    finance_dev = (finance_probs[row_idx] - finance_marginal[None, :]).flatten()
    correlation = float(np.corrcoef(ett_dev, finance_dev)[0, 1])
    print(f"marginal-deviation correlation (mean-based): r = {correlation:.4f} (F0-07's .last()-based value: r = 0.898)\n")

    # -- permutation null ------------------------------------------------------
    null_frobenius = np.empty(N_PERMUTATIONS)
    null_kl_mean = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled_ett = [rng.permutation(arr) for arr in ett_daily_mean_arrays]
        shuffled_finance = [rng.permutation(arr) for arr in finance_arrays]

        shuf_ett_counts = counts_from_index_arrays(shuffled_ett)
        shuf_finance_counts = counts_from_index_arrays(shuffled_finance)
        shuf_ett_probs, shuf_ett_totals = row_normalise(shuf_ett_counts)
        shuf_finance_probs, shuf_finance_totals = row_normalise(shuf_finance_counts)

        shuf_common_rows = common_defined_rows(shuf_ett_totals, shuf_finance_totals)
        if shuf_common_rows.sum() < common_rows.sum():
            shuf_common_rows = shuf_common_rows & common_rows

        null_frobenius[i] = frobenius_norm(shuf_ett_probs, shuf_finance_probs, shuf_common_rows)

        shuf_ett_smoothed = jeffreys_smoothed(shuf_ett_counts)
        shuf_finance_smoothed = jeffreys_smoothed(shuf_finance_counts)
        kl_ef = row_wise_kl(shuf_ett_smoothed, shuf_finance_smoothed, shuf_common_rows)
        kl_fe = row_wise_kl(shuf_finance_smoothed, shuf_ett_smoothed, shuf_common_rows)
        null_kl_mean[i] = (kl_ef + kl_fe) / 2

    percentile_frobenius = float((null_frobenius <= observed_frobenius).mean() * 100)
    percentile_kl = float((null_kl_mean <= observed_kl_mean).mean() * 100)
    gap_frobenius = (observed_frobenius - null_frobenius.mean()) / null_frobenius.std()
    gap_kl = (observed_kl_mean - null_kl_mean.mean()) / null_kl_mean.std()

    print(f"permutation null (n={N_PERMUTATIONS}):")
    print(
        f"  Frobenius norm: null mean={null_frobenius.mean():.4f}, null std={null_frobenius.std():.4f}, "
        f"observed={observed_frobenius:.4f}, percentile={percentile_frobenius:.2f}, gap={gap_frobenius:.2f} null sd"
    )
    print(
        f"  symmetrised KL: null mean={null_kl_mean.mean():.4f}, null std={null_kl_mean.std():.4f}, "
        f"observed={observed_kl_mean:.4f}, percentile={percentile_kl:.2f}, gap={gap_kl:.2f} null sd"
    )

    summary = pd.DataFrame(
        {
            "statistic": ["frobenius_norm", "symmetrised_kl_bits"],
            "observed": [observed_frobenius, observed_kl_mean],
            "null_mean": [null_frobenius.mean(), null_kl_mean.mean()],
            "null_std": [null_frobenius.std(), null_kl_mean.std()],
            "observed_percentile": [percentile_frobenius, percentile_kl],
            "gap_null_sd": [gap_frobenius, gap_kl],
            "n_permutations": [N_PERMUTATIONS, N_PERMUTATIONS],
        }
    )
    summary_path = RESULTS_DIR / "transition_similarity_summary_daily_mean.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")

    null_df = pd.DataFrame({"frobenius_norm": null_frobenius, "symmetrised_kl_bits": null_kl_mean})
    null_path = RESULTS_DIR / "transition_similarity_null_daily_mean.csv"
    null_df.to_csv(null_path, index=False)
    print(f"wrote {null_path}")

    # -- explicit comparison against .last()-based figures ------------------
    print("\n=== .last() vs .mean() comparison ===")
    for stat, label in [("frobenius_norm", "Frobenius norm"), ("symmetrised_kl_bits", "symmetrised KL")]:
        last_r = LAST_BASED_RESULT[stat]
        mean_r = summary[summary.statistic == stat].iloc[0]
        last_gap = (last_r["observed"] - last_r["null_mean"]) / last_r["null_std"]
        print(
            f"{label}: .last() = {last_r['observed']:.4f} (percentile {last_r['percentile']:.1f}, gap {last_gap:.2f} sd) "
            f"vs .mean() = {mean_r['observed']:.4f} (percentile {mean_r['observed_percentile']:.1f}, gap {mean_r['gap_null_sd']:.2f} sd)"
        )


if __name__ == "__main__":
    main()
