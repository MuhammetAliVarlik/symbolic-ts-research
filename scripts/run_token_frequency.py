"""Token frequency distributions across domains.

Uses the tokenisation from `scripts/prototype_tokenize.py` as-is (2-channel
change/volatility projection, sigma bins fit per series on its own first 70%).
Pools token counts across all series within a domain -- all 4 ETT variants
into one "ett" distribution, all 7 tickers into one "finance" distribution --
and compares the two. If one domain concentrates in a handful of the 25
possible tokens while the other spreads evenly, the shared vocabulary is
nominal rather than real and cross-domain transfer has little chance.

Run:
    python scripts/run_token_frequency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

from prototype_tokenize import BASE_READING, LABELS, project_channels, tokenize
from datasets.ett import VALID_VARIANTS as ETT_VARIANTS
from datasets.ett import load_ett
from datasets.finance import VALID_TICKERS as FINANCE_TICKERS
from datasets.finance import load_finance

RESULTS_PATH = REPO_ROOT / "results" / "token_frequency.csv"

ALL_TOKENS = [f"C{c}_V{v}" for c in LABELS for v in LABELS]  # 25 possible symbols


def domain_token_counts(domain: str, series_ids: list[str], loader) -> pd.Series:
    counts = pd.Series(0, index=ALL_TOKENS, dtype=int)
    for series_id in series_ids:
        base_reading = loader(series_id)
        channels = project_channels(domain, base_reading)
        tokens, _vol_labels, _change_fit, _vol_fit = tokenize(channels)
        counts = counts.add(pd.Series(tokens).value_counts(), fill_value=0)
    return counts.astype(int)


def main() -> None:
    ett_counts = domain_token_counts(
        "ett", list(ETT_VARIANTS),
        lambda v: load_ett(v).set_index("date")[BASE_READING["ett"]],
    )
    finance_counts = domain_token_counts(
        "finance", list(FINANCE_TICKERS),
        lambda t: load_finance([t]).set_index("date")[BASE_READING["finance"]],
    )

    df = pd.concat(
        [
            pd.DataFrame({"domain": "ett", "token": ett_counts.index, "count": ett_counts.values}),
            pd.DataFrame({"domain": "finance", "token": finance_counts.index, "count": finance_counts.values}),
        ],
        ignore_index=True,
    )
    df["proportion"] = df.groupby("domain")["count"].transform(lambda c: c / c.sum())

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"wrote {len(df)} rows to {RESULTS_PATH}")

    # Chi-square test of independence on the 2x25 contingency table: is token
    # choice independent of domain, or does domain predict which tokens appear?
    contingency = pd.DataFrame({"ett": ett_counts, "finance": finance_counts}).T
    chi2, p_value, dof, _expected = chi2_contingency(contingency)
    print(f"\nchi-square test of independence: chi2={chi2:.2f}, dof={dof}, p={p_value:.3e}")

    # Jensen-Shannon divergence (base 2, bounded [0, 1]) between the two
    # proportion distributions -- a symmetric, interpretable distance metric
    # complementing the hypothesis test above.
    ett_props = (ett_counts / ett_counts.sum()).values
    finance_props = (finance_counts / finance_counts.sum()).values
    js_divergence = jensenshannon(ett_props, finance_props, base=2) ** 2
    print(f"Jensen-Shannon divergence (base 2, bounded [0,1]): {js_divergence:.4f}")

    print(f"\nnumber of tokens with zero count in ETT: {(ett_counts == 0).sum()} / {len(ALL_TOKENS)}")
    print(f"number of tokens with zero count in finance: {(finance_counts == 0).sum()} / {len(ALL_TOKENS)}")

    ett_top5 = (ett_counts / ett_counts.sum()).sort_values(ascending=False).head(5)
    finance_top5 = (finance_counts / finance_counts.sum()).sort_values(ascending=False).head(5)
    print(f"\nETT top 5 tokens (share of total): \n{ett_top5.round(4)}")
    print(f"\nfinance top 5 tokens (share of total): \n{finance_top5.round(4)}")
    print(f"\nETT top-5 cumulative share: {ett_top5.sum():.1%}")
    print(f"finance top-5 cumulative share: {finance_top5.sum():.1%}")


if __name__ == "__main__":
    main()
