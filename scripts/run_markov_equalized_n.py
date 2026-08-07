"""Equalized-N control for F0-10's cross-domain asymmetry.

F0-10 found `ett_daily->finance` clearly worse than finance's own in-domain
baseline at orders 2-5 (non-overlapping accuracy CIs), while `finance->
ett_daily` showed no such drop at any order. That asymmetry has two very
different possible explanations that look identical in the original numbers:

  (a) a real, directional domain-transfer effect, or
  (b) ett_daily's model being handicapped by ~8x less training data at every
      order in a way that happens to look directionally consistent, since
      finance's own in-domain baseline benefited from ~15,400 training
      positions while the model being transferred (trained on ett_daily) only
      had ~1,976.

This control removes explanation (b): finance's training set is subsampled
down to ett_daily's size, its order 1-5 models rebuilt from that smaller set,
and both cross-domain directions re-evaluated. The critical comparison is now
`finance(equalized)->finance` (in-domain, ~1,976 training positions) against
`ett_daily->finance` (cross-domain, unchanged, also ~1,976 training
positions) -- both models now trained on the same amount of data, so any
remaining "clearly worse" gap can no longer be attributed to a training-size
mismatch.

Subsampling method: a contiguous prefix of each ticker's own training split
(not a random scatter of individual points -- an n-gram model needs real
contiguous sequence to count transitions from), sized proportionally so the
total across all 7 tickers matches ett_daily's total training count. A fixed
prefix (not a random window) was used for simplicity and exact
reproducibility; this means the equalized model sees only the earliest ~13%
of each ticker's training history, a secondary limitation noted in the
report rather than engineered around.

Writes to NEW result files -- does not touch or replace F0-10's original
`results/markov_baseline.csv`.

Run:
    python scripts/run_markov_equalized_n.py
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
from statsmodels.stats.proportion import proportion_confint

from run_markov_baseline import (
    MAX_ORDER,
    N_TOKENS,
    bootstrap_ci,
    build_backoff_models,
    domain_arrays,
    evaluate_order,
    macro_f1,
    perplexity,
    split_index,
)

RESULTS_DIR = REPO_ROOT / "results"
RNG_SEED = 20260810


def equalize_finance_training(finance_arrays: list[np.ndarray], ett_daily_arrays: list[np.ndarray]) -> list[np.ndarray]:
    finance_train = [arr[: split_index(arr)] for arr in finance_arrays]
    ett_daily_train_total = sum(split_index(arr) for arr in ett_daily_arrays)
    finance_train_total = sum(len(arr) for arr in finance_train)

    fraction = ett_daily_train_total / finance_train_total
    print(f"finance train total = {finance_train_total}, ett_daily train total = {ett_daily_train_total}, fraction = {fraction:.4f}")

    equalized = [arr[: max(1, int(len(arr) * fraction))] for arr in finance_train]
    print(f"equalized finance per-ticker training sizes: {[len(a) for a in equalized]}, total = {sum(len(a) for a in equalized)}")
    return equalized


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    finance_arrays = domain_arrays("finance")
    ett_daily_arrays = domain_arrays("ett_daily")
    ett_daily_train = [arr[: split_index(arr)] for arr in ett_daily_arrays]

    finance_train_equalized = equalize_finance_training(finance_arrays, ett_daily_arrays)
    models_equalized = build_backoff_models(finance_train_equalized, MAX_ORDER)

    rows = []
    for eval_domain, eval_full_arrays in [("finance", finance_arrays), ("ett_daily", ett_daily_arrays)]:
        cell_type = "in-domain (equalized)" if eval_domain == "finance" else "cross-domain (equalized)"
        for order in range(1, MAX_ORDER + 1):
            result = evaluate_order(finance_train_equalized, eval_full_arrays, order, models_equalized)
            actual, predicted, prob_actual = result["actual"], result["predicted"], result["prob_actual"]
            n = len(actual)

            acc = float(np.mean(predicted == actual))
            acc_lo, acc_hi = proportion_confint(int(np.sum(predicted == actual)), n, alpha=0.05, method="wilson")
            f1 = macro_f1(actual, predicted, N_TOKENS)
            f1_lo, f1_hi = bootstrap_ci(actual, predicted, prob_actual, lambda a, p, pr: macro_f1(a, p, N_TOKENS), rng)
            ppl = perplexity(prob_actual)
            ppl_lo, ppl_hi = bootstrap_ci(actual, predicted, prob_actual, lambda a, p, pr: perplexity(pr), rng)

            print(
                f"finance(equalized)->{eval_domain} ({cell_type}), order={order}: "
                f"acc={acc:.4f} [{acc_lo:.4f},{acc_hi:.4f}], macroF1={f1:.4f} [{f1_lo:.4f},{f1_hi:.4f}], "
                f"ppl={ppl:.3f} [{ppl_lo:.3f},{ppl_hi:.3f}], n={n}"
            )
            rows.append(
                {
                    "train_domain": "finance_equalized", "eval_domain": eval_domain, "cell_type": cell_type,
                    "order": order, "n_eval": n, "n_train": sum(len(a) for a in finance_train_equalized),
                    "accuracy": acc, "accuracy_wilson_lo": acc_lo, "accuracy_wilson_hi": acc_hi,
                    "macro_f1": f1, "macro_f1_boot_lo": f1_lo, "macro_f1_boot_hi": f1_hi,
                    "perplexity": ppl, "perplexity_boot_lo": ppl_lo, "perplexity_boot_hi": ppl_hi,
                }
            )

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "markov_baseline_equalized_n.csv"
    df.to_csv(output_path, index=False)
    print(f"\nwrote {output_path}")

    # -- the critical comparison: does the order 2-5 asymmetry survive? ----------
    original = pd.read_csv(RESULTS_DIR / "markov_baseline.csv")
    ett_to_finance = original[(original.train_domain == "ett_daily") & (original.eval_domain == "finance")].set_index("order")
    finance_eq_indomain = df[(df.eval_domain == "finance")].set_index("order")

    def overlap(a_lo, a_hi, b_lo, b_hi):
        return not (a_hi < b_lo or b_hi < a_lo)

    print("\nequalized-N critical comparison: finance(equalized) in-domain vs ett_daily->finance (cross), unchanged:")
    comparison_rows = []
    for order in range(1, MAX_ORDER + 1):
        a = finance_eq_indomain.loc[order]
        b = ett_to_finance.loc[order]
        ov = overlap(a.accuracy_wilson_lo, a.accuracy_wilson_hi, b.accuracy_wilson_lo, b.accuracy_wilson_hi)
        print(
            f"  order {order}: finance(equalized, n_train={a.n_train:.0f})=[{a.accuracy_wilson_lo:.4f},{a.accuracy_wilson_hi:.4f}] "
            f"ett_daily->finance(n_train~1976)=[{b.accuracy_wilson_lo:.4f},{b.accuracy_wilson_hi:.4f}] "
            f"overlap={ov} ({'asymmetry NOT explained by sample size alone' if not ov else 'gap closes -- consistent with sample-size confound'})"
        )
        comparison_rows.append(
            {
                "order": order,
                "finance_equalized_indomain_acc": a.accuracy, "finance_equalized_lo": a.accuracy_wilson_lo, "finance_equalized_hi": a.accuracy_wilson_hi,
                "ett_daily_to_finance_acc": b.accuracy, "ett_daily_to_finance_lo": b.accuracy_wilson_lo, "ett_daily_to_finance_hi": b.accuracy_wilson_hi,
                "cis_overlap": ov,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "markov_equalized_n_comparison.csv", index=False)
    print(f"\nwrote {RESULTS_DIR / 'markov_equalized_n_comparison.csv'}")


if __name__ == "__main__":
    main()
