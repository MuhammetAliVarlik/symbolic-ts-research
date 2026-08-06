"""Figures for the raw-data and stationarity analysis.

No figure computes anything itself -- every function here reads a CSV under
`results/` and only aggregates/formats for display.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _style import OKABE_ITO, apply_style, save

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def fig_a2_stationarity() -> None:
    """A2 -- ADF/KPSS results table-figure, summarised by domain x transform x test."""
    apply_style()

    df = pd.read_csv(RESULTS_DIR / "stationarity.csv")
    df["is_stationary"] = df["conclusion"] == "stationary"

    summary = (
        df.groupby(["domain", "transform", "test"])
        .agg(pct_stationary=("is_stationary", "mean"), n=("is_stationary", "size"))
        .reset_index()
    )
    summary["pct_stationary"] = (summary["pct_stationary"] * 100).round(1)
    summary = summary.sort_values(["domain", "transform", "test"])

    col_labels = ["Domain", "Transform", "Test", "% stationary", "n"]
    cell_text = summary[["domain", "transform", "test", "pct_stationary", "n"]].astype(str).values.tolist()

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    header_color = OKABE_ITO["sky_blue"]
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")

    save(fig, OUTPUT_DIR, "A2_stationarity")
    plt.close(fig)


def fig_a4_token_frequency() -> None:
    """A4 -- token frequency distribution, finance vs ETT side by side."""
    apply_style()

    df = pd.read_csv(RESULTS_DIR / "token_frequency.csv")
    pivot = df.pivot(index="token", columns="domain", values="proportion").fillna(0.0)
    pivot = pivot.sort_index()

    x = range(len(pivot))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar([i - width / 2 for i in x], pivot["ett"], width, label="ETT", color=OKABE_ITO["blue"])
    ax.bar([i + width / 2 for i in x], pivot["finance"], width, label="finance", color=OKABE_ITO["vermillion"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=90, fontsize=7)
    ax.set_xlabel("token")
    ax.set_ylabel("proportion of domain's tokens")
    ax.legend(frameon=False)

    save(fig, OUTPUT_DIR, "A4_token_frequency")
    plt.close(fig)


def _transition_heatmap(matrix: pd.DataFrame, name: str) -> None:
    # Fixed [0, 1] bounds, not a data-driven max: a single-observation sparse
    # row can produce a "probability" of exactly 1.0 that has nothing to do
    # with real structure (see transition_matrix_row_counts.csv). A
    # data-driven max would let that artifact dominate the shared scale.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")  # undefined rows (0 observations), not zero probability

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_xlabel("to token")
    ax.set_ylabel("from token")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="transition probability", fraction=0.046, pad=0.04)

    save(fig, OUTPUT_DIR, name)
    plt.close(fig)


def fig_a5_transition_finance() -> None:
    """A5 -- finance transition matrix heatmap (25x25), shared colour scale with A6."""
    apply_style()
    matrix = pd.read_csv(RESULTS_DIR / "transition_matrix_finance.csv", index_col=0)
    _transition_heatmap(matrix, "A5_transition_finance")


def fig_a6_transition_ett() -> None:
    """A6 -- ETT transition matrix heatmap (25x25), shared colour scale with A5."""
    apply_style()
    matrix = pd.read_csv(RESULTS_DIR / "transition_matrix_ett.csv", index_col=0)
    _transition_heatmap(matrix, "A6_transition_ett")


def fig_a7_transition_similarity() -> None:
    """A7 -- transition matrix difference, annotated with the divergence against
    the permutation null. Left: null distribution (Frobenius norm) with the
    observed value marked. Right: the signed difference matrix itself,
    restricted to rows defined in both domains."""
    apply_style()

    summary = pd.read_csv(RESULTS_DIR / "transition_similarity_summary.csv").set_index("statistic")
    null_df = pd.read_csv(RESULTS_DIR / "transition_similarity_null.csv")

    ett = pd.read_csv(RESULTS_DIR / "transition_matrix_ett.csv", index_col=0)
    fin = pd.read_csv(RESULTS_DIR / "transition_matrix_finance.csv", index_col=0)
    common_rows = ett.dropna(how="all").index.intersection(fin.dropna(how="all").index)
    diff = (ett.loc[common_rows] - fin.loc[common_rows]).astype(float)

    frob = summary.loc["frobenius_norm"]
    kl = summary.loc["symmetrised_kl_bits"]

    fig, (ax_hist, ax_diff) = plt.subplots(1, 2, figsize=(13, 6))

    ax_hist.hist(null_df["frobenius_norm"], bins=40, color=OKABE_ITO["sky_blue"], label="permutation null")
    ax_hist.axvline(frob["observed"], color=OKABE_ITO["vermillion"], linewidth=2, label="observed")
    ax_hist.set_xlabel("Frobenius norm (ETT vs finance)")
    ax_hist.set_ylabel("permutations")
    ax_hist.legend(frameon=False)
    ax_hist.annotate(
        f"observed = {frob['observed']:.3f}\n"
        f"null mean = {frob['null_mean']:.3f} (sd {frob['null_std']:.3f})\n"
        f"percentile = {frob['observed_percentile']:.1f}\n"
        f"symmetrised KL = {kl['observed']:.3f} bits\n"
        f"(null mean {kl['null_mean']:.3f}, percentile {kl['observed_percentile']:.1f})",
        xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8,
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    bound = float(diff.abs().to_numpy().max())
    im = ax_diff.imshow(diff.to_numpy(), cmap=cmap, vmin=-bound, vmax=bound)
    ax_diff.set_xticks(range(len(diff.columns)))
    ax_diff.set_xticklabels(diff.columns, rotation=90, fontsize=6)
    ax_diff.set_yticks(range(len(diff.index)))
    ax_diff.set_yticklabels(diff.index, fontsize=6)
    ax_diff.set_xlabel("to token")
    ax_diff.set_ylabel("from token")
    ax_diff.grid(False)
    fig.colorbar(im, ax=ax_diff, label="P(ett) - P(finance)", fraction=0.046, pad=0.04)

    save(fig, OUTPUT_DIR, "A7_transition_similarity")
    plt.close(fig)


if __name__ == "__main__":
    fig_a2_stationarity()
    fig_a4_token_frequency()
    fig_a5_transition_finance()
    fig_a6_transition_ett()
    fig_a7_transition_similarity()
