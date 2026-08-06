"""Shared figure style: Okabe-Ito palette + rcParams. Imported by every fig_*.py.

No figure script computes anything itself -- each one reads a CSV under
`results/` and only aggregates/formats for display.
"""
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

PALETTE = list(OKABE_ITO.values())

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Carlito", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def save(fig, output_dir, name: str) -> None:
    """Save a figure as both vector PDF (for LaTeX) and 300dpi PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
