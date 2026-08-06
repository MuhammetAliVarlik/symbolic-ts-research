"""First-order transition matrices per domain.

Uses the tokenisation from `scripts/prototype_tokenize.py` as-is. Transitions
are counted WITHIN each series only -- the last token of one ETT variant (or
ticker) is never counted as transitioning into the first token of the next,
since that boundary isn't a real transition.

Zero-count cells are left as explicit zeros after row-normalising, not
smoothed -- smoothing here would hide exactly the sparse-row problem this
script is required to surface. Rows with zero total observations (a token
that never once appears as a "from" state) are left undefined (NaN), not
coerced into a fabricated distribution. Row observation counts are reported
in a companion CSV, flagged below a stated minimum, so F0-07's KL divergence
calculation can account for unreliable rows explicitly instead of silently
trusting them.

Run:
    python scripts/run_transition_matrices.py
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

from prototype_tokenize import BASE_READING, LABELS, project_channels, tokenize
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_DIR = REPO_ROOT / "results"
ALL_TOKENS = [f"C{c}_V{v}" for c in LABELS for v in LABELS]  # 25 possible symbols

# Common rule-of-thumb minimum for a trustworthy categorical/multinomial
# estimate. Rows below this are flagged, not dropped or smoothed away.
MIN_OBSERVATIONS = 30


def series_tokens(domain: str, series_id: str, loader) -> list[str]:
    base_reading = loader(series_id)
    channels = project_channels(domain, base_reading)
    tokens, _vol_labels, _change_fit, _vol_fit = tokenize(channels)
    return tokens


def transition_counts(domain: str, series_ids: list[str], loader) -> pd.DataFrame:
    counts = pd.DataFrame(0, index=ALL_TOKENS, columns=ALL_TOKENS, dtype=int)
    for series_id in series_ids:
        tokens = series_tokens(domain, series_id, loader)
        for current_token, next_token in zip(tokens[:-1], tokens[1:]):
            counts.loc[current_token, next_token] += 1
    return counts


def row_normalise(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    row_totals = counts.sum(axis=1)
    with np.errstate(invalid="ignore"):
        probabilities = counts.div(row_totals, axis=0)
    return probabilities, row_totals


def main() -> None:
    domains = {
        "ett": (list(ETT_VARIANTS), lambda v: load_ett(v).set_index("date")[BASE_READING["ett"]]),
        "finance": (list(FINANCE_TICKERS), lambda t: load_finance([t]).set_index("date")[BASE_READING["finance"]]),
    }

    row_count_rows = []
    for domain, (series_ids, loader) in domains.items():
        counts = transition_counts(domain, series_ids, loader)
        probabilities, row_totals = row_normalise(counts)

        matrix_path = RESULTS_DIR / f"transition_matrix_{domain}.csv"
        probabilities.to_csv(matrix_path)
        print(f"wrote {matrix_path}")

        n_undefined = (row_totals == 0).sum()
        n_sparse = ((row_totals > 0) & (row_totals < MIN_OBSERVATIONS)).sum()
        n_sufficient = (row_totals >= MIN_OBSERVATIONS).sum()
        print(
            f"  {domain}: {n_undefined} undefined rows (0 obs), "
            f"{n_sparse} sparse rows (1-{MIN_OBSERVATIONS - 1} obs), "
            f"{n_sufficient} sufficient rows (>={MIN_OBSERVATIONS} obs)"
        )

        for token in ALL_TOKENS:
            n = int(row_totals[token])
            if n == 0:
                flag = "undefined"
            elif n < MIN_OBSERVATIONS:
                flag = "sparse"
            else:
                flag = "sufficient"
            row_count_rows.append({"domain": domain, "from_token": token, "row_total": n, "flag": flag})

    row_counts_df = pd.DataFrame(row_count_rows)
    row_counts_path = RESULTS_DIR / "transition_matrix_row_counts.csv"
    row_counts_df.to_csv(row_counts_path, index=False)
    print(f"\nwrote {row_counts_path}")

    flagged = row_counts_df[row_counts_df["flag"] != "sufficient"].sort_values(["domain", "row_total"])
    print(f"\nrows flagged as undefined or sparse (< {MIN_OBSERVATIONS} observations):")
    print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()
