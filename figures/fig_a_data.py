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


if __name__ == "__main__":
    fig_a2_stationarity()
    fig_a4_token_frequency()
