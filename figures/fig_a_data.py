"""Figures for the raw-data and stationarity analysis.

No figure computes anything itself -- every function here reads a CSV under
`results/` and only aggregates/formats for display.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def fig_diag_granularity_matched_similarity() -> None:
    """Diagnostic (not a numbered thesis figure): native-granularity vs
    granularity-matched (daily-resampled ETT) null distributions side by
    side, same observed-value-vs-null-histogram layout as A7, to show how
    much the sampling-frequency confound check closes the gap."""
    apply_style()

    native_summary = pd.read_csv(RESULTS_DIR / "transition_similarity_summary.csv").set_index("statistic")
    native_null = pd.read_csv(RESULTS_DIR / "transition_similarity_null.csv")
    daily_summary = pd.read_csv(RESULTS_DIR / "transition_similarity_summary_daily.csv").set_index("statistic")
    daily_null = pd.read_csv(RESULTS_DIR / "transition_similarity_null_daily.csv")

    fig, (ax_native, ax_daily) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, summary, null_df, title in [
        (ax_native, native_summary, native_null, "native granularity (hourly/15-min ETT)"),
        (ax_daily, daily_summary, daily_null, "granularity-matched (daily-resampled ETT)"),
    ]:
        frob = summary.loc["frobenius_norm"]
        ax.hist(null_df["frobenius_norm"], bins=40, color=OKABE_ITO["sky_blue"], label="permutation null")
        ax.axvline(frob["observed"], color=OKABE_ITO["vermillion"], linewidth=2, label="observed")
        ax.set_xlabel("Frobenius norm (ETT vs finance)")
        ax.set_title(title, fontsize=10)
        ax.annotate(
            f"percentile = {frob['observed_percentile']:.1f}\n"
            f"gap = {(frob['observed'] - frob['null_mean']) / frob['null_std']:.2f} null sd",
            xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8,
        )
    ax_native.set_ylabel("permutations")
    ax_native.legend(frameon=False, loc="center left")

    save(fig, OUTPUT_DIR, "diag_granularity_matched_similarity")
    plt.close(fig)


def fig_a9_entropy_mi() -> None:
    """A9 -- entropy rate (plug-in vs. Miller-Madow-corrected) and lag-1..10
    mutual information, per domain."""
    apply_style()

    df = pd.read_csv(RESULTS_DIR / "information_theory.csv")
    domains = ["ett", "finance", "ett_daily"]
    colors = {"ett": OKABE_ITO["blue"], "finance": OKABE_ITO["vermillion"], "ett_daily": OKABE_ITO["bluish_green"]}

    fig, (ax_entropy, ax_mi) = plt.subplots(1, 2, figsize=(13, 5.5))

    x = np.arange(len(domains))
    width = 0.35
    plugin = [df[(df.domain == d) & (df.metric == "entropy_rate_plugin")]["value_bits"].iloc[0] for d in domains]
    mm = [df[(df.domain == d) & (df.metric == "entropy_rate_miller_madow")]["value_bits"].iloc[0] for d in domains]
    ax_entropy.bar(x - width / 2, plugin, width, label="plug-in", color=OKABE_ITO["sky_blue"])
    ax_entropy.bar(x + width / 2, mm, width, label="Miller-Madow", color=OKABE_ITO["orange"])
    ax_entropy.axhline(np.log2(25), color=OKABE_ITO["black"], linestyle="--", linewidth=1, label="max (log2 25)")
    ax_entropy.set_xticks(x)
    ax_entropy.set_xticklabels(domains)
    ax_entropy.set_ylabel("entropy rate (bits)")
    ax_entropy.legend(frameon=False)

    for domain in domains:
        subset = df[(df.domain == domain) & (df.metric == "mutual_information")].sort_values("lag")
        ax_mi.plot(subset["lag"], subset["value_bits"], marker="o", label=domain, color=colors[domain])
    ax_mi.set_xlabel("lag")
    ax_mi.set_ylabel("mutual information (bits)")
    ax_mi.set_xticks(range(1, 11))
    ax_mi.legend(frameon=False)

    save(fig, OUTPUT_DIR, "A9_entropy_mi")
    plt.close(fig)


def fig_a8_acf_pacf() -> None:
    """A8 -- token sequence ACF with Bartlett bounds, per domain x channel,
    justifying the recommended context window."""
    apply_style()

    df = pd.read_csv(RESULTS_DIR / "acf_pacf.csv")
    domains = ["ett_hourly", "ett_15min", "finance", "ett_daily"]
    channels = ["change", "volatility"]

    fig, axes = plt.subplots(len(domains), len(channels), figsize=(11, 12), sharex=True)

    for row, domain in enumerate(domains):
        for col, channel in enumerate(channels):
            ax = axes[row, col]
            sub = df[(df.domain == domain) & (df.channel == channel)]
            ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.5)
            ax.fill_between(
                sub["lag"], -sub["acf_bartlett_bound"], sub["acf_bartlett_bound"],
                color=OKABE_ITO["sky_blue"], alpha=0.3, label="Bartlett bound",
            )
            ax.plot(sub["lag"], sub["acf"], color=OKABE_ITO["vermillion"], linewidth=1)
            if row == 0:
                ax.set_title(channel, fontsize=10)
            if col == 0:
                ax.set_ylabel(domain, fontsize=9)
            if row == len(domains) - 1:
                ax.set_xlabel("lag")

    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=7)
    save(fig, OUTPUT_DIR, "A8_acf_pacf")
    plt.close(fig)


def fig_diag_markov_accuracy_by_order() -> None:
    """Diagnostic (not one of the numbered thesis figures -- this pairing,
    in-domain vs. cross-domain Markov accuracy by order, isn't in the
    Package A-H catalog in WORKING_PLAN.md; named `diag_` like
    `fig_diag_granularity_matched_similarity` above rather than an invented
    `A#` slot). Markov order (1-5) on the x-axis, accuracy on the y-axis,
    Wilson CI error bars from `results/markov_baseline.csv`'s own
    `accuracy_wilson_lo`/`accuracy_wilson_hi` columns (F1-07's Wilson score
    interval -- not recomputed here). Colour = training domain, line style =
    in-domain (solid) vs. cross-domain (dashed), so the same-colour
    solid-vs-dashed gap is a direct visual read of the transfer cost."""
    apply_style()

    df = pd.read_csv(RESULTS_DIR / "markov_baseline.csv")

    series = [
        ("finance", "finance", OKABE_ITO["blue"], "-", "finance -> finance (in-domain)"),
        ("finance", "ett_daily", OKABE_ITO["blue"], "--", "finance -> ett_daily (cross-domain)"),
        ("ett_daily", "ett_daily", OKABE_ITO["vermillion"], "-", "ett_daily -> ett_daily (in-domain)"),
        ("ett_daily", "finance", OKABE_ITO["vermillion"], "--", "ett_daily -> finance (cross-domain)"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for train_domain, eval_domain, color, linestyle, label in series:
        subset = df[(df.train_domain == train_domain) & (df.eval_domain == eval_domain)].sort_values("order")
        lower_err = subset["accuracy"] - subset["accuracy_wilson_lo"]
        upper_err = subset["accuracy_wilson_hi"] - subset["accuracy"]
        ax.errorbar(
            subset["order"], subset["accuracy"], yerr=[lower_err, upper_err],
            color=color, linestyle=linestyle, marker="o", capsize=3, label=label,
        )

    ax.set_xlabel("Markov order")
    ax.set_ylabel("accuracy")
    ax.set_xticks(sorted(df["order"].unique()))
    ax.legend(frameon=False)

    save(fig, OUTPUT_DIR, "diag_markov_accuracy_by_order")
    plt.close(fig)


if __name__ == "__main__":
    fig_a2_stationarity()
    fig_a4_token_frequency()
    fig_a5_transition_finance()
    fig_a6_transition_ett()
    fig_a7_transition_similarity()
    fig_a8_acf_pacf()
    fig_diag_granularity_matched_similarity()
    fig_diag_markov_accuracy_by_order()
    fig_a9_entropy_mi()
