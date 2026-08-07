"""Change-only re-check of the white-noise sanity check (5 symbols, not 25).

The volatility channel's 20-period rolling std mechanically induces
autocorrelation in ANY input, including pure iid noise -- verified directly:
raw iid noise has ~zero ACF at every lag, but rolling(20).std() of that same
noise shows ACF decaying from 0.94 (lag 1) through 0.48 (lag 10) to ~0 right
at lag 19-20, exactly where 20-period windows stop overlapping. This is a
textbook property of moving-window statistics, not something specific to real
data -- see the note for the verification numbers.

This re-runs the white-noise sanity check (run_white_noise_sanity_check.py)
using ONLY the change label (0-4, 5 symbols) as the token, dropping the
volatility digit entirely, to isolate whether the apparent cross-domain
similarity found in F0-06/F0-07/the original white-noise check survives
without the windowing-induced artificial memory the volatility channel
carries.

NOTE: local, correctly-sized reimplementations of counts/normalise/smoothing/
KL/Frobenius are used here rather than importing run_transition_similarity's
versions directly -- those hardcode a 25-token matrix via module-level
closure, and naively feeding 5-symbol data through them would apply Jeffreys
smoothing calibrated for 25 possible categories (dividing by row_total + 12.5)
instead of 5 (row_total + 2.5), silently over-smoothing every row.

Run:
    python scripts/run_white_noise_change_only.py
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

from prototype_tokenize import BASE_READING, TRAIN_FRACTION, fit_sigma, project_channels
from run_granularity_matched_comparison import resample_ett_daily
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
N_TOKENS = 5  # change label only: 0..4
LABELS = ["0", "1", "2", "3", "4"]
N_PERMUTATIONS = 1000
JEFFREYS_ALPHA = 0.5
N_NOISE_REPLICATIONS = 10
RNG_SEED = 20260807


def change_only_tokens(change_series: pd.Series) -> np.ndarray:
    """Sigma-bucket a `change` series alone (no volatility), fit on its own
    first 70% -- identical bucketing rule to prototype_tokenize.tokenize(),
    just dropping the volatility half of the compound token entirely."""
    split = int(len(change_series) * TRAIN_FRACTION)
    fit = fit_sigma(change_series.iloc[:split])
    labels = fit.bucket(change_series)
    return np.array([int(x) for x in labels], dtype=np.int64)


def real_change_series_and_tokens(domain: str, base_reading: pd.Series) -> tuple[int, float, np.ndarray]:
    channels = project_channels(domain, base_reading)
    change = channels["change"]
    return len(change), float(change.var()), change_only_tokens(change)


def synthetic_change_only_tokens(rng: np.random.Generator, n: int, change_std: float) -> np.ndarray:
    change = pd.Series(rng.normal(0.0, change_std, size=n))
    return change_only_tokens(change)


def counts_from_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    counts = np.zeros(N_TOKENS * N_TOKENS, dtype=np.int64)
    for arr in arrays:
        if len(arr) < 2:
            continue
        flat_idx = arr[:-1] * N_TOKENS + arr[1:]
        counts += np.bincount(flat_idx, minlength=N_TOKENS * N_TOKENS)
    return counts.reshape(N_TOKENS, N_TOKENS)


def row_normalise(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_totals = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = counts / row_totals[:, None]
    return probabilities, row_totals


def jeffreys_smoothed(counts: np.ndarray) -> np.ndarray:
    smoothed = counts + JEFFREYS_ALPHA
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def common_defined_rows(totals_a: np.ndarray, totals_b: np.ndarray) -> np.ndarray:
    return (totals_a > 0) & (totals_b > 0)


def row_wise_kl(p: np.ndarray, q: np.ndarray, common_rows: np.ndarray) -> float:
    kls = np.sum(p[common_rows] * np.log2(p[common_rows] / q[common_rows]), axis=1)
    return float(kls.mean())


def frobenius_norm(p: np.ndarray, q: np.ndarray, common_rows: np.ndarray) -> float:
    diff = p[common_rows] - q[common_rows]
    return float(np.sqrt(np.sum(diff**2)))


def observed_stats(arrays_a: list[np.ndarray], arrays_b: list[np.ndarray]) -> dict:
    ca, cb = counts_from_arrays(arrays_a), counts_from_arrays(arrays_b)
    pa, ta = row_normalise(ca)
    pb, tb = row_normalise(cb)
    common = common_defined_rows(ta, tb)
    frob = frobenius_norm(pa, pb, common)
    sa, sb = jeffreys_smoothed(ca), jeffreys_smoothed(cb)
    kl = (row_wise_kl(sa, sb, common) + row_wise_kl(sb, sa, common)) / 2
    return {"frobenius_norm": frob, "symmetrised_kl_bits": kl, "n_common_rows": int(common.sum())}


def full_comparison_with_null(arrays_a: list[np.ndarray], arrays_b: list[np.ndarray], rng: np.random.Generator) -> dict:
    obs = observed_stats(arrays_a, arrays_b)
    null_frob = np.empty(N_PERMUTATIONS)
    null_kl = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuf_a = [rng.permutation(arr) for arr in arrays_a]
        shuf_b = [rng.permutation(arr) for arr in arrays_b]
        s = observed_stats(shuf_a, shuf_b)
        null_frob[i], null_kl[i] = s["frobenius_norm"], s["symmetrised_kl_bits"]
    return {
        "frobenius_norm": {
            "observed": obs["frobenius_norm"], "null_mean": null_frob.mean(), "null_std": null_frob.std(),
            "percentile": float((null_frob <= obs["frobenius_norm"]).mean() * 100),
        },
        "symmetrised_kl_bits": {
            "observed": obs["symmetrised_kl_bits"], "null_mean": null_kl.mean(), "null_std": null_kl.std(),
            "percentile": float((null_kl <= obs["symmetrised_kl_bits"]).mean() * 100),
        },
        "n_common_rows": obs["n_common_rows"],
    }


def print_result(name: str, result: dict) -> None:
    print(f"\n{name}:")
    for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
        r = result[stat]
        gap = (r["observed"] - r["null_mean"]) / r["null_std"] if r["null_std"] > 0 else float("nan")
        print(f"  {stat}: observed={r['observed']:.4f}, null_mean={r['null_mean']:.4f} (sd {r['null_std']:.4f}), percentile={r['percentile']:.1f}, gap={gap:.2f} sd")
    print(f"  common rows: {result['n_common_rows']}/{N_TOKENS}")


def main() -> None:
    finance_base = [load_finance([t]).set_index("date")[BASE_READING["finance"]] for t in FINANCE_TICKERS]
    ett_daily_base = [resample_ett_daily(v) for v in ETT_VARIANTS]

    finance_stats = [real_change_series_and_tokens("finance", s) for s in finance_base]
    ett_daily_stats = [real_change_series_and_tokens("ett", s) for s in ett_daily_base]
    real_finance_tokens = [t for _n, _v, t in finance_stats]
    real_ett_daily_tokens = [t for _n, _v, t in ett_daily_stats]

    print("=== fresh change-only real finance vs real ett_daily (own permutation null) ===")
    rng = np.random.default_rng(RNG_SEED)
    real_result = full_comparison_with_null(real_finance_tokens, real_ett_daily_tokens, rng)
    print_result("real finance vs real ett_daily (change-only)", real_result)

    print("\n=== noise, matched per-series to n and change-variance (change-only tokens) ===")
    rows = []
    for replication in range(N_NOISE_REPLICATIONS):
        rep_rng = np.random.default_rng(RNG_SEED + 1000 * (replication + 1))
        noise_finance = [synthetic_change_only_tokens(rep_rng, n, np.sqrt(v)) for n, v, _t in finance_stats]
        noise_ett_daily = [synthetic_change_only_tokens(rep_rng, n, np.sqrt(v)) for n, v, _t in ett_daily_stats]

        for name, (a, b) in {
            "noise_vs_noise": (noise_finance, noise_ett_daily),
            "noise_finance_vs_real_ett_daily": (noise_finance, real_ett_daily_tokens),
            "noise_ett_daily_vs_real_finance": (noise_ett_daily, real_finance_tokens),
        }.items():
            s = observed_stats(a, b)
            rows.append({"comparison": name, "replication": replication, **s})

    rep_df = pd.DataFrame(rows)
    rep_path = RESULTS_DIR / "white_noise_change_only_replications.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"wrote {rep_path}")

    print("\nspread across 10 independent noise replications (change-only, 5 symbols):")
    summary_rows = []
    for name in ["noise_vs_noise", "noise_finance_vs_real_ett_daily", "noise_ett_daily_vs_real_finance"]:
        for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
            vals = rep_df[rep_df.comparison == name][stat]
            real_val = real_result[stat]["observed"]
            pct = float((vals <= real_val).mean() * 100)
            print(f"  {name} / {stat}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.4f}, {vals.max():.4f}] -- real value {real_val:.4f} at {pct:.0f}th percentile of this")
            summary_rows.append(
                {
                    "comparison": name, "statistic": stat, "replication_mean": vals.mean(),
                    "replication_std": vals.std(), "replication_min": vals.min(), "replication_max": vals.max(),
                    "real_value": real_val, "real_value_percentile_in_replications": pct,
                }
            )

    for stat in ["frobenius_norm", "symmetrised_kl_bits"]:
        summary_rows.append(
            {
                "comparison": "real_finance_vs_real_ett_daily", "statistic": stat,
                "replication_mean": np.nan, "replication_std": np.nan, "replication_min": np.nan, "replication_max": np.nan,
                "real_value": real_result[stat]["observed"], "real_value_percentile_in_replications": np.nan,
            }
        )
    summary_path = RESULTS_DIR / "white_noise_change_only_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
