"""Figures for the raw-data and stationarity analysis.

No figure computes anything itself -- every function here reads a CSV under
`results/` and only aggregates/formats for display.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _style import OKABE_ITO, apply_style, save

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "stationarity.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def fig_a2_stationarity() -> None:
    """A2 -- ADF/KPSS results table-figure, summarised by domain x transform x test."""
    apply_style()

    df = pd.read_csv(RESULTS_PATH)
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


if __name__ == "__main__":
    fig_a2_stationarity()
