"""White-noise sanity check: is F0-07's cross-domain similarity a real shared
pattern, or an artifact of sigma-binning's central-tendency construction?

Generates synthetic white noise (iid Gaussian innovations, no real temporal
structure) matched per-series to each real domain's sample size AND its
estimated change-variance, runs it through the exact same tokenisation
(prototype_tokenize.tokenize), transition-matrix (run_transition_matrices),
and cross-domain similarity (run_transition_similarity) pipeline as F0-06/
F0-07, and compares:

  (a) noise-vs-noise (matched to finance vs matched to ett_daily) -- what the
      pipeline reports as "similarity" when there is genuinely nothing shared.
  (b) noise-vs-real, both directions -- does structureless noise look
      artificially similar to a real domain under this binning scheme.

against F0-07's real finance-vs-ett_daily (granularity-matched) result:
Frobenius 1.0647 (96.8th percentile, 1.95 null sd), symmetrised KL 0.3605
bits (80.4th percentile, 0.85 null sd).

Side check: because sigma-binning always compares a value against its OWN
train-fit mean/std, the resulting token distribution should be invariant to
the raw variance of the input noise -- verified directly (not just argued)
before relying on "variance-matched" as if it were a meaningful knob distinct
from "unit-variance, sample-size-matched only".

Run:
    python scripts/run_white_noise_sanity_check.py
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
from scipy.spatial.distance import jensenshannon

from prototype_tokenize import BASE_READING, VOLATILITY_WINDOW, project_channels, tokenize
from run_transition_matrices import ALL_TOKENS
from run_transition_similarity import (
    N_PERMUTATIONS,
    TOKEN_TO_IDX,
    common_defined_rows,
    counts_from_index_arrays,
    frobenius_norm,
    jeffreys_smoothed,
    row_normalise,
    row_wise_kl,
)
from run_granularity_matched_comparison import resample_ett_daily, tokens_to_index_array
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
N_TOKENS = len(ALL_TOKENS)
RNG_SEED = 20260807  # distinct from run_transition_similarity's seed -- independent draw

# F0-07's real (granularity-matched) result, for direct comparison.
REAL_RESULT = {
    "frobenius_norm": {"observed": 1.0647, "null_mean": 0.9511, "null_std": 0.0581, "percentile": 96.8},
    "symmetrised_kl_bits": {"observed": 0.3605, "null_mean": 0.3422, "null_std": 0.0216, "percentile": 80.4},
}


def real_series_stats(domain: str, base_reading: pd.Series) -> tuple[int, float]:
    """(n_channel_rows, change_variance) for one real series, pre-tokenisation."""
    channels = project_channels(domain, base_reading)
    return len(channels), float(channels["change"].var())


def synthetic_token_array(rng: np.random.Generator, n_channel_rows: int, change_std: float) -> np.ndarray:
    """iid Gaussian noise, injected directly as the `change` channel (skipping
    the level -> diff step entirely, since the point is zero real temporal
    structure) -- same volatility/sigma-binning/tokenize as the real pipeline."""
    n_raw = n_channel_rows + VOLATILITY_WINDOW - 1
    change = pd.Series(rng.normal(0.0, change_std, size=n_raw))
    volatility = change.rolling(VOLATILITY_WINDOW).std()
    channels = pd.DataFrame({"change": change, "volatility": volatility}).dropna().reset_index(drop=True)
    assert len(channels) == n_channel_rows, f"{len(channels)} != {n_channel_rows}"
    tokens, _vol_labels, _change_fit, _vol_fit = tokenize(channels)
    return np.array([TOKEN_TO_IDX[t] for t in tokens], dtype=np.int64)


def matched_noise_arrays(rng: np.random.Generator, series_stats: list[tuple[int, float]]) -> list[np.ndarray]:
    return [synthetic_token_array(rng, n, np.sqrt(var)) for n, var in series_stats]


def run_similarity(arrays_a: list[np.ndarray], arrays_b: list[np.ndarray], rng: np.random.Generator) -> dict:
    counts_a = counts_from_index_arrays(arrays_a)
    counts_b = counts_from_index_arrays(arrays_b)
    probs_a, totals_a = row_normalise(counts_a)
    probs_b, totals_b = row_normalise(counts_b)

    common_rows = common_defined_rows(totals_a, totals_b)
    observed_frobenius = frobenius_norm(probs_a, probs_b, common_rows)

    smoothed_a = jeffreys_smoothed(counts_a)
    smoothed_b = jeffreys_smoothed(counts_b)
    kl_ab = row_wise_kl(smoothed_a, smoothed_b, common_rows)
    kl_ba = row_wise_kl(smoothed_b, smoothed_a, common_rows)
    observed_kl = (kl_ab + kl_ba) / 2

    marginal_a = counts_a.sum(axis=1) / counts_a.sum()
    marginal_b = counts_b.sum(axis=1) / counts_b.sum()
    row_idx = np.where(common_rows)[0]
    dev_a = (probs_a[row_idx] - marginal_a[None, :]).flatten()
    dev_b = (probs_b[row_idx] - marginal_b[None, :]).flatten()
    correlation = float(np.corrcoef(dev_a, dev_b)[0, 1])

    null_frobenius = np.empty(N_PERMUTATIONS)
    null_kl = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuf_a = [rng.permutation(arr) for arr in arrays_a]
        shuf_b = [rng.permutation(arr) for arr in arrays_b]
        sc_a, sc_b = counts_from_index_arrays(shuf_a), counts_from_index_arrays(shuf_b)
        sp_a, st_a = row_normalise(sc_a)
        sp_b, st_b = row_normalise(sc_b)
        shuf_common = common_defined_rows(st_a, st_b)
        if shuf_common.sum() < common_rows.sum():
            shuf_common = shuf_common & common_rows
        null_frobenius[i] = frobenius_norm(sp_a, sp_b, shuf_common)
        ssa, ssb = jeffreys_smoothed(sc_a), jeffreys_smoothed(sc_b)
        null_kl[i] = (row_wise_kl(ssa, ssb, shuf_common) + row_wise_kl(ssb, ssa, shuf_common)) / 2

    pct_frob = float((null_frobenius <= observed_frobenius).mean() * 100)
    pct_kl = float((null_kl <= observed_kl).mean() * 100)
    return {
        "frobenius_norm": {
            "observed": observed_frobenius, "null_mean": null_frobenius.mean(),
            "null_std": null_frobenius.std(), "percentile": pct_frob,
        },
        "symmetrised_kl_bits": {
            "observed": observed_kl, "null_mean": null_kl.mean(),
            "null_std": null_kl.std(), "percentile": pct_kl,
        },
        "marginal_deviation_correlation": correlation,
        "n_common_rows": int(common_rows.sum()),
    }


def print_comparison(name: str, result: dict) -> None:
    print(f"\n{name}:")
    for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
        r = result[stat]
        gap_sd = (r["observed"] - r["null_mean"]) / r["null_std"] if r["null_std"] > 0 else float("nan")
        print(
            f"  {stat}: observed={r['observed']:.4f}, null_mean={r['null_mean']:.4f} "
            f"(sd {r['null_std']:.4f}), percentile={r['percentile']:.1f}, gap={gap_sd:.2f} null sd"
        )
    print(f"  marginal-deviation correlation: r={result['marginal_deviation_correlation']:.4f}")
    print(f"  common rows compared: {result['n_common_rows']}/{N_TOKENS}")


N_NOISE_REPLICATIONS = 10  # independent noise realisations -- a single draw was found
# to be unstable (Frobenius norm std=0.149 across 15 exploratory draws, comparable to
# the effect size itself), so every noise-involving comparison is repeated and reported
# as a distribution, not a single number.


def main() -> None:
    seed_rng = np.random.default_rng(RNG_SEED)

    finance_base_readings = [load_finance([t]).set_index("date")[BASE_READING["finance"]] for t in FINANCE_TICKERS]
    ett_daily_base_readings = [resample_ett_daily(v) for v in ETT_VARIANTS]

    finance_stats = [real_series_stats("finance", s) for s in finance_base_readings]
    ett_daily_stats = [real_series_stats("ett", s) for s in ett_daily_base_readings]
    print("finance per-series (n, change_var):", [(n, round(v, 6)) for n, v in finance_stats])
    print("ett_daily per-series (n, change_var):", [(n, round(v, 6)) for n, v in ett_daily_stats])

    # -- side check: does raw noise variance matter for the tokenised output? ----
    print("\n--- side check: variance-matched vs unit-variance noise, same n (finance-sized) ---")
    n_ref = finance_stats[0][0]
    var_ref = finance_stats[0][1]
    variance_matched = synthetic_token_array(seed_rng, n_ref, np.sqrt(var_ref))
    unit_variance = synthetic_token_array(seed_rng, n_ref, 1.0)
    p_vm = np.bincount(variance_matched, minlength=N_TOKENS) / len(variance_matched)
    p_uv = np.bincount(unit_variance, minlength=N_TOKENS) / len(unit_variance)
    js = jensenshannon(p_vm, p_uv, base=2) ** 2
    print(f"  Jensen-Shannon divergence between token distributions: {js:.5f} (bounded [0,1])")
    print("  (near zero would confirm sigma-binning normalises away raw input variance, as expected)")

    real_finance_arrays = [tokens_to_index_array("finance", s) for s in finance_base_readings]
    real_ett_daily_arrays = [tokens_to_index_array("ett", s) for s in ett_daily_base_readings]

    print(f"\nreal F0-07 result (granularity-matched, for reference):")
    print_comparison("real finance vs real ett_daily (F0-07)", REAL_RESULT | {
        "marginal_deviation_correlation": 0.898, "n_common_rows": 23,
    })

    # -- N independent noise replications per comparison -------------------------
    replication_rows = []
    for replication in range(N_NOISE_REPLICATIONS):
        rep_rng = np.random.default_rng(RNG_SEED + 1000 * (replication + 1))
        noise_finance = matched_noise_arrays(rep_rng, finance_stats)
        noise_ett_daily = matched_noise_arrays(rep_rng, ett_daily_stats)

        comparisons = {
            "noise_vs_noise": (noise_finance, noise_ett_daily),
            "noise_finance_vs_real_ett_daily": (noise_finance, real_ett_daily_arrays),
            "noise_ett_daily_vs_real_finance": (noise_ett_daily, real_finance_arrays),
        }
        for name, (arrays_a, arrays_b) in comparisons.items():
            # Cheap per-replication: observed statistic only, no 1000-permutation
            # null per draw (that would be 10x1000x3 permutations). The null's
            # purpose -- "what does chance look like" -- is instead served by the
            # spread ACROSS the 10 independent noise replications themselves.
            counts_a = counts_from_index_arrays(arrays_a)
            counts_b = counts_from_index_arrays(arrays_b)
            probs_a, totals_a = row_normalise(counts_a)
            probs_b, totals_b = row_normalise(counts_b)
            common_rows = common_defined_rows(totals_a, totals_b)
            frob = frobenius_norm(probs_a, probs_b, common_rows)
            sa, sb = jeffreys_smoothed(counts_a), jeffreys_smoothed(counts_b)
            kl = (row_wise_kl(sa, sb, common_rows) + row_wise_kl(sb, sa, common_rows)) / 2
            replication_rows.append({"comparison": name, "replication": replication, "frobenius_norm": frob, "symmetrised_kl_bits": kl})

    rep_df = pd.DataFrame(replication_rows)

    # -- full permutation-null pipeline, once, on the FIRST replication (as literally
    # specified: same F0-06/F0-07 pipeline including its own null) -----------------
    rng = np.random.default_rng(RNG_SEED)
    noise_finance_0 = matched_noise_arrays(np.random.default_rng(RNG_SEED + 1000), finance_stats)
    noise_ett_daily_0 = matched_noise_arrays(np.random.default_rng(RNG_SEED + 1000), ett_daily_stats)
    comparisons_0 = {
        "noise_vs_noise": (noise_finance_0, noise_ett_daily_0),
        "noise_finance_vs_real_ett_daily": (noise_finance_0, real_ett_daily_arrays),
        "noise_ett_daily_vs_real_finance": (noise_ett_daily_0, real_finance_arrays),
    }

    summary_rows = []
    for name, (arrays_a, arrays_b) in comparisons_0.items():
        result = run_similarity(arrays_a, arrays_b, rng)
        print_comparison(f"{name} (single replication, own permutation null)", result)
        for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
            r = result[stat]
            replication_spread = rep_df[rep_df["comparison"] == name][stat]
            summary_rows.append(
                {
                    "comparison": name,
                    "statistic": stat,
                    "observed_single_draw": r["observed"],
                    "own_permutation_null_mean": r["null_mean"],
                    "own_permutation_null_std": r["null_std"],
                    "observed_percentile_vs_own_null": r["percentile"],
                    "gap_own_null_sd": (r["observed"] - r["null_mean"]) / r["null_std"] if r["null_std"] > 0 else np.nan,
                    f"mean_across_{N_NOISE_REPLICATIONS}_noise_replications": replication_spread.mean(),
                    "std_across_replications": replication_spread.std(),
                    "min_across_replications": replication_spread.min(),
                    "max_across_replications": replication_spread.max(),
                    "marginal_deviation_correlation": result["marginal_deviation_correlation"],
                    "n_common_rows": result["n_common_rows"],
                }
            )

    print(f"\nspread across {N_NOISE_REPLICATIONS} independent noise replications (observed statistic only, no per-replication null):")
    for name in comparisons_0:
        for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
            s = rep_df[rep_df["comparison"] == name][stat]
            print(f"  {name} / {stat}: mean={s.mean():.4f}, std={s.std():.4f}, range=[{s.min():.4f}, {s.max():.4f}]")
    real_val = {"frobenius_norm": REAL_RESULT["frobenius_norm"]["observed"], "symmetrised_kl_bits": REAL_RESULT["symmetrised_kl_bits"]["observed"]}
    for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
        print(f"  (real finance-vs-ett_daily {stat}: {real_val[stat]:.4f})")

    df = pd.DataFrame(summary_rows)
    output_path = RESULTS_DIR / "white_noise_sanity_check.csv"
    df.to_csv(output_path, index=False)
    print(f"\nwrote {output_path}")

    rep_output_path = RESULTS_DIR / "white_noise_sanity_check_replications.csv"
    rep_df.to_csv(rep_output_path, index=False)
    print(f"wrote {rep_output_path}")


if __name__ == "__main__":
    main()
