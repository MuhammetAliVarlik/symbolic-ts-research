"""Markov-k baseline, in-domain and cross-domain -- the second half of the
Phase 0 gate (the first half was F0-07's transition-matrix similarity test).

Answers "why do you need 500M parameters for next-symbol prediction?" by
setting the bar a transferred SLM must clear: orders 1-5, with backoff to
lower orders (down to a Laplace-smoothed order-0 base) on unseen contexts,
evaluated in-domain and cross-domain between finance and `ett_daily` --
F0-07's granularity-matched daily-resampled ETT, used here for the same
reason it was primary there: lag-for-lag comparability between domains.

Train/test split: first 70% / last 30% of each series, same convention as
`prototype_tokenize.TRAIN_FRACTION`. This is a Phase-0 diagnostic holdout, NOT
the locked test set from decision D9 -- that lock applies to the actual SLM
evaluation from Phase 4 onward, not this exploratory baseline.

Evaluation positions are always the TARGET domain's own test-split positions.
Context for those positions is always drawn from the target domain's own full
series (so the first few test positions can legitimately use the tail of that
same domain's training portion as context -- this is not leakage, it's how a
real deployed predictor would see a continuous stream; the domain boundary
between train/test is an evaluation-time artifact, not a real discontinuity).
What changes between in-domain and cross-domain cells is only which model
does the predicting.

Metrics: accuracy (Wilson CI -- the correct fit, since accuracy is a binomial
proportion), macro-F1 and perplexity (bootstrap CI instead -- Wilson's
formula is specifically for proportions and doesn't apply to either of these,
so a literal "Wilson CI" for them would be a misapplication rather than a
substitution).

Run:
    python scripts/run_markov_baseline.py
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

from run_transition_matrices import ALL_TOKENS
from run_granularity_matched_comparison import resample_ett_daily, tokens_to_index_array
from prototype_tokenize import BASE_READING
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
N_TOKENS = len(ALL_TOKENS)
MAX_ORDER = 5
TRAIN_FRACTION = 0.7
LAPLACE_ALPHA = 0.5  # order-0 base smoothing -- guarantees finite perplexity even
# for tokens the training domain never emitted at all (finance never emits
# C0_V0/C4_V0 at any order, including order-0, without this)
N_BOOTSTRAP = 1000
RNG_SEED = 20260810


def domain_arrays(domain: str) -> list[np.ndarray]:
    if domain == "finance":
        return [tokens_to_index_array("finance", load_finance([t]).set_index("date")[BASE_READING["finance"]]) for t in FINANCE_TICKERS]
    if domain == "ett_daily":
        return [tokens_to_index_array("ett", resample_ett_daily(v)) for v in ETT_VARIANTS]
    raise ValueError(domain)


def split_index(arr: np.ndarray) -> int:
    return int(len(arr) * TRAIN_FRACTION)


def build_order0(train_arrays: list[np.ndarray]) -> np.ndarray:
    counts = np.zeros(N_TOKENS)
    for arr in train_arrays:
        counts += np.bincount(arr, minlength=N_TOKENS)
    smoothed = counts + LAPLACE_ALPHA
    return smoothed / smoothed.sum()


def build_ngram_counts(train_arrays: list[np.ndarray], order: int) -> dict[tuple, np.ndarray]:
    model: dict[tuple, np.ndarray] = {}
    for arr in train_arrays:
        for i in range(len(arr) - order):
            context = tuple(arr[i : i + order].tolist())
            nxt = arr[i + order]
            if context not in model:
                model[context] = np.zeros(N_TOKENS)
            model[context][nxt] += 1
    return model


def build_backoff_models(train_arrays: list[np.ndarray], max_order: int) -> dict:
    models: dict = {0: build_order0(train_arrays)}
    for k in range(1, max_order + 1):
        models[k] = build_ngram_counts(train_arrays, k)
    return models


def predict_distribution(context_tail: np.ndarray, order: int, models: dict) -> np.ndarray:
    """Try `order`, back off through order-1, ..., 1 on unseen contexts (a
    context with NO training occurrences at all), and fall back to the
    Laplace-smoothed order-0 base (always available).

    Backoff only triggers on an unseen CONTEXT, per the task spec -- but a
    SEEN context with few observations still needs smoothing of its own: a
    context observed once or twice will otherwise assign 100% probability to
    whichever single token happened to follow and exactly 0% to every other
    token, which is not "the model is confident", it's "the model has no way
    to represent uncertainty" -- and makes perplexity infinite the moment a
    test continuation doesn't match what was seen in training. The same
    Laplace pseudocount used for order-0 is applied here for consistency."""
    for k in range(order, 0, -1):
        if len(context_tail) < k:
            continue
        ctx = tuple(context_tail[-k:].tolist())
        counts = models[k].get(ctx)
        if counts is not None and counts.sum() > 0:
            smoothed = counts + LAPLACE_ALPHA
            return smoothed / smoothed.sum()
    return models[0]


def evaluate_order(train_arrays: list[np.ndarray], eval_full_arrays: list[np.ndarray], order: int, models: dict) -> dict:
    """eval_full_arrays: FULL series (train+test) for the TARGET domain -- only
    the test-split positions (last 30%) are scored, but context may reach back
    into that same series' own training portion."""
    actual, predicted, prob_actual = [], [], []
    for arr in eval_full_arrays:
        split = split_index(arr)
        for i in range(max(split, 0), len(arr)):
            if i == 0:
                continue
            context_tail = arr[:i]
            dist = predict_distribution(context_tail, order, models)
            actual_token = int(arr[i])
            actual.append(actual_token)
            predicted.append(int(np.argmax(dist)))
            prob_actual.append(float(dist[actual_token]))
    return {"actual": np.array(actual), "predicted": np.array(predicted), "prob_actual": np.array(prob_actual)}


def macro_f1(actual: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((predicted == c) & (actual == c)))
        fp = int(np.sum((predicted == c) & (actual != c)))
        fn = int(np.sum((predicted != c) & (actual == c)))
        if tp + fp + fn == 0:
            continue  # class never appears as actual or predicted -- excluded, not counted as 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def perplexity(prob_actual: np.ndarray) -> float:
    return float(np.exp(-np.mean(np.log(prob_actual))))


def bootstrap_ci(actual: np.ndarray, predicted: np.ndarray, prob_actual: np.ndarray, stat_fn, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    n = len(actual)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = stat_fn(actual[idx], predicted[idx], prob_actual[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    domains = {"finance": domain_arrays("finance"), "ett_daily": domain_arrays("ett_daily")}
    train_splits = {d: [arr[: split_index(arr)] for arr in arrs] for d, arrs in domains.items()}

    rows = []
    for train_domain, train_arrays in train_splits.items():
        models_by_order = build_backoff_models(train_arrays, MAX_ORDER)
        for eval_domain, eval_full_arrays in domains.items():
            cell_type = "in-domain" if train_domain == eval_domain else "cross-domain"
            for order in range(1, MAX_ORDER + 1):
                result = evaluate_order(train_arrays, eval_full_arrays, order, models_by_order)
                actual, predicted, prob_actual = result["actual"], result["predicted"], result["prob_actual"]
                n = len(actual)

                acc = float(np.mean(predicted == actual))
                acc_lo, acc_hi = proportion_confint(int(np.sum(predicted == actual)), n, alpha=0.05, method="wilson")

                f1 = macro_f1(actual, predicted, N_TOKENS)
                f1_lo, f1_hi = bootstrap_ci(actual, predicted, prob_actual, lambda a, p, pr: macro_f1(a, p, N_TOKENS), rng)

                ppl = perplexity(prob_actual)
                ppl_lo, ppl_hi = bootstrap_ci(actual, predicted, prob_actual, lambda a, p, pr: perplexity(pr), rng)

                print(
                    f"{train_domain}->{eval_domain} ({cell_type}), order={order}: "
                    f"acc={acc:.4f} [{acc_lo:.4f},{acc_hi:.4f}], "
                    f"macroF1={f1:.4f} [{f1_lo:.4f},{f1_hi:.4f}], "
                    f"ppl={ppl:.3f} [{ppl_lo:.3f},{ppl_hi:.3f}], n={n}"
                )

                rows.append(
                    {
                        "train_domain": train_domain, "eval_domain": eval_domain, "cell_type": cell_type,
                        "order": order, "n_eval": n,
                        "accuracy": acc, "accuracy_wilson_lo": acc_lo, "accuracy_wilson_hi": acc_hi,
                        "macro_f1": f1, "macro_f1_boot_lo": f1_lo, "macro_f1_boot_hi": f1_hi,
                        "perplexity": ppl, "perplexity_boot_lo": ppl_lo, "perplexity_boot_hi": ppl_hi,
                    }
                )

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "markov_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
