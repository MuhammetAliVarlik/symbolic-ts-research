"""Cross-domain transition similarity -- the Phase 0 gate.

Quantifies how similar the ETT and finance transition matrices are, and
compares that against a chance baseline built from shuffled sequences. A
number without a null distribution is not evidence, per the task spec -- the
permutation baseline is the point of this script, not an afterthought.

Two statistics, both computed on the REAL (observed) matrices and on >=1000
within-series-shuffled replicates:

- Frobenius norm of the matrix difference, on the unsmoothed matrices
  (explicit zeros are fine for a plain elementwise difference; no log
  anywhere, so no smoothing is needed for this one).
- Row-wise KL divergence, which DOES need smoothing: checked directly, 22 of
  23 commonly-defined rows have KL(ett||finance) = +inf (finance has a hard
  zero somewhere ETT doesn't), and 10 of 23 have KL(finance||ett) = +inf
  (the reverse). Naive KL on the F0-06 explicit-zero matrices is unusable as
  a summary statistic. Jeffreys smoothing (alpha=0.5 pseudocount per cell) is
  applied here, for this comparison only -- it does not touch or replace the
  F0-06 matrices on disk, which stay explicit-zero for their own purpose.

Both domains' 2 (ETT) / 3 (finance) sparse-or-undefined rows from F0-06, plus
any row undefined in the OTHER domain, are excluded from both statistics --
comparing a well-sampled row against noise, or against a row with zero data,
isn't a comparison of domain structure.

Run:
    python scripts/run_transition_similarity.py
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
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
N_PERMUTATIONS = 1000
JEFFREYS_ALPHA = 0.5  # standard Jeffreys prior pseudocount for categorical smoothing
RNG_SEED = 42

TOKEN_TO_IDX = {token: i for i, token in enumerate(ALL_TOKENS)}
N_TOKENS = len(ALL_TOKENS)


def domain_token_index_arrays(series_ids: list[str], loader, domain: str) -> list[np.ndarray]:
    """Each series' token sequence, as an array of integer indices 0..24."""
    arrays = []
    for series_id in series_ids:
        base_reading = loader(series_id)
        channels = project_channels(domain, base_reading)
        tokens, _vol_labels, _change_fit, _vol_fit = tokenize(channels)
        arrays.append(np.array([TOKEN_TO_IDX[t] for t in tokens], dtype=np.int64))
    return arrays


def counts_from_index_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """25x25 transition count matrix from a list of per-series index arrays,
    transitions counted within each series only."""
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
    smoothed_counts = counts + JEFFREYS_ALPHA
    return smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)


def row_wise_kl(p: np.ndarray, q: np.ndarray, common_rows: np.ndarray) -> float:
    """Mean over common_rows of KL(p_row || q_row), both already smoothed
    (strictly positive everywhere), so this is always finite."""
    kls = np.sum(p[common_rows] * np.log2(p[common_rows] / q[common_rows]), axis=1)
    return float(kls.mean())


def frobenius_norm(p: np.ndarray, q: np.ndarray, common_rows: np.ndarray) -> float:
    diff = p[common_rows] - q[common_rows]
    return float(np.sqrt(np.sum(diff**2)))


def common_defined_rows(ett_row_totals: np.ndarray, finance_row_totals: np.ndarray) -> np.ndarray:
    return (ett_row_totals > 0) & (finance_row_totals > 0)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    ett_arrays = domain_token_index_arrays(list(ETT_VARIANTS), lambda v: load_ett(v).set_index("date")[BASE_READING["ett"]], "ett")
    finance_arrays = domain_token_index_arrays(
        list(FINANCE_TICKERS), lambda t: load_finance([t]).set_index("date")[BASE_READING["finance"]], "finance"
    )

    ett_counts = counts_from_index_arrays(ett_arrays)
    finance_counts = counts_from_index_arrays(finance_arrays)
    ett_probs, ett_totals = row_normalise(ett_counts)
    finance_probs, finance_totals = row_normalise(finance_counts)

    common_rows = common_defined_rows(ett_totals, finance_totals)
    excluded = [ALL_TOKENS[i] for i in range(N_TOKENS) if not common_rows[i]]
    print(f"rows excluded from cross-domain comparison (undefined in at least one domain): {excluded}")
    print(f"comparing {common_rows.sum()} / {N_TOKENS} rows\n")

    # -- observed statistics -------------------------------------------------
    observed_frobenius = frobenius_norm(ett_probs, finance_probs, common_rows)

    ett_smoothed = jeffreys_smoothed(ett_counts)
    finance_smoothed = jeffreys_smoothed(finance_counts)
    observed_kl_ett_finance = row_wise_kl(ett_smoothed, finance_smoothed, common_rows)
    observed_kl_finance_ett = row_wise_kl(finance_smoothed, ett_smoothed, common_rows)
    observed_kl_mean = (observed_kl_ett_finance + observed_kl_finance_ett) / 2

    print(f"observed Frobenius norm (unsmoothed, {common_rows.sum()} common rows): {observed_frobenius:.4f}")
    print(f"observed KL(ett||finance), Jeffreys-smoothed, mean over common rows: {observed_kl_ett_finance:.4f} bits")
    print(f"observed KL(finance||ett), Jeffreys-smoothed, mean over common rows: {observed_kl_finance_ett:.4f} bits")
    print(f"observed symmetrised mean KL: {observed_kl_mean:.4f} bits\n")

    # per-row detail for the record
    row_detail = []
    for i, token in enumerate(ALL_TOKENS):
        if not common_rows[i]:
            continue
        kl_ef = float(np.sum(ett_smoothed[i] * np.log2(ett_smoothed[i] / finance_smoothed[i])))
        kl_fe = float(np.sum(finance_smoothed[i] * np.log2(finance_smoothed[i] / ett_smoothed[i])))
        row_detail.append(
            {
                "from_token": token,
                "ett_row_total": int(ett_totals[i]),
                "finance_row_total": int(finance_totals[i]),
                "kl_ett_vs_finance_bits": kl_ef,
                "kl_finance_vs_ett_bits": kl_fe,
                "frobenius_row": float(np.sqrt(np.sum((ett_probs[i] - finance_probs[i]) ** 2))),
            }
        )
    row_detail_df = pd.DataFrame(row_detail)
    row_detail_path = RESULTS_DIR / "transition_similarity_row_detail.csv"
    row_detail_df.to_csv(row_detail_path, index=False)
    print(f"wrote {row_detail_path}")

    # -- permutation null ------------------------------------------------------
    null_frobenius = np.empty(N_PERMUTATIONS)
    null_kl_mean = np.empty(N_PERMUTATIONS)

    for i in range(N_PERMUTATIONS):
        shuffled_ett = [rng.permutation(arr) for arr in ett_arrays]
        shuffled_finance = [rng.permutation(arr) for arr in finance_arrays]

        shuf_ett_counts = counts_from_index_arrays(shuffled_ett)
        shuf_finance_counts = counts_from_index_arrays(shuffled_finance)
        shuf_ett_probs, shuf_ett_totals = row_normalise(shuf_ett_counts)
        shuf_finance_probs, shuf_finance_totals = row_normalise(shuf_finance_counts)

        shuf_common_rows = common_defined_rows(shuf_ett_totals, shuf_finance_totals)
        # Shuffling preserves each token's total count exactly; only which
        # token happens to land last in a series can flip a row_total by at
        # most (number of series), so this should essentially always match
        # `common_rows`. Fall back to the real mask if a permutation somehow
        # produces fewer usable rows, rather than silently comparing an
        # inconsistent row set across iterations.
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

    print(f"\npermutation null (n={N_PERMUTATIONS}):")
    print(
        f"  Frobenius norm: null mean={null_frobenius.mean():.4f}, "
        f"null std={null_frobenius.std():.4f}, observed={observed_frobenius:.4f}, "
        f"observed percentile={percentile_frobenius:.2f}"
    )
    print(
        f"  symmetrised KL: null mean={null_kl_mean.mean():.4f}, "
        f"null std={null_kl_mean.std():.4f}, observed={observed_kl_mean:.4f}, "
        f"observed percentile={percentile_kl:.2f}"
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
    summary_path = RESULTS_DIR / "transition_similarity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")

    null_df = pd.DataFrame({"frobenius_norm": null_frobenius, "symmetrised_kl_bits": null_kl_mean})
    null_path = RESULTS_DIR / "transition_similarity_null.csv"
    null_df.to_csv(null_path, index=False)
    print(f"wrote {null_path}")


if __name__ == "__main__":
    main()
