#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import math
import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatter, LogFormatterMathtext, LogLocator, MultipleLocator
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

from typing import Tuple, List, Dict, Optional


data_folder = Path("data")

plot_folder = Path("poster/figures/generated/")

assert data_folder.exists(), f"Data folder {data_folder} does not exist. Wrong working directory?"
plot_folder.mkdir(parents=False, exist_ok=True)


# Shared palette from slides/config.tex.  The ordering retains the existing
# plot roles: accent/tan, primary/blue, secondary/teal, warning/red, dark
# neutral, and light neutral.
COLOR1 = "#4357AD"
COLOR2 = "#48A9A6"
COLOR3 = "#E4DFDA"
COLOR4 = "#D4B483"
COLOR5 = "#C1666B"
base_colors = [COLOR4, COLOR1, COLOR2, COLOR5, "#333333", COLOR3]

VOLUME_X = "Overhead"
VOLUME_Y = "Volume_[MB]"
VOLUME_X_LABEL_OVERRIDE: Optional[str] = "Relevant leaves / \"overhead\""
VOLUME_Y_LABEL_OVERRIDE: Optional[str] = "Access volume [MB]"
RUNTIME_Y = "Runtime_[ms]"
STORAGE_Y = "Total_Storage_Size_[GB]"
RUNTIME_X_LABEL_OVERRIDE: Optional[str] = None
RUNTIME_Y_LABEL_OVERRIDE: Optional[str] = None
STORAGE_Y_LABEL_OVERRIDE: Optional[str] = None
RUNTIME_AXIS_COLOR = "#222222"
STORAGE_AXIS_COLOR = "#222222"
COMPOSITION_GROUPS = ["QueryDescr", "b", "d", "Index"]
ExperimentPredicate = Callable[[pd.DataFrame], pd.Series]
QUERY_FLAVOR_LABELS = {
    "balanced_selective": "selective",
    "diverse": "less selective",
}
QUERY_FLAVOR_COLORS = {
    "balanced_selective": base_colors[1],
    "diverse": base_colors[2],
}
QUERY_FLAVOR_MARKERS = {
    "balanced_selective": "o",
    "diverse": "^",
}
SDSS_AUTO_INDEX = "d=5/r"
SDSS_VA_KEY = ("Any", "VA")
SDSS_REQUIRED_COLUMNS = {
    "d", "D", "s", "Index", "Runtime [s]", "Selectivity",
    "total_list_count", "total_list_size",
}
SDSSS_X_AXIS_LABEL: Optional[str] = "Table dim."
SDSS_COLORS = [base_colors[1], base_colors[2], base_colors[0], base_colors[3]]
SDSS_MARKERS = {
    ("d=1",): "^",
    ("d=5/m",): "^",
    ("d=5/r",): "^",
    ("VA",): "o",
}
SDSS_STYLE_LABELS = {
    ("d=1",): "$8^1$ Grid idx.",
    ("d=5/r",): "$8^5$ Grid idx.",
    ("VA",): "3-bit VA-file"
}
SDSS_STYLE_LABELS_2 = {
    ("d=1",): "$8^1$ Grid idx. (random)",
    ("d=5/r",): "$8^5$ Grid idx. (random)",
    ("VA",): "3-bit VA-file (sequential)",
}
FIGURE_8_COLORS = [base_colors[0], base_colors[3], base_colors[1], base_colors[4], base_colors[2]]

# The PDFs are scaled to fixed physical widths by the A1 poster. Keep their
# current aspect ratios, but use explicit visual weights for reliable print
# legibility at those final sizes.
TRADEOFF_TICK_LABEL_SIZE = 8
TRADEOFF_LEGEND_FONT_SIZE = 7.5
TRADEOFF_ANNOTATION_FONT_SIZE = 7
TRADEOFF_LEGEND_MARKER_SIZE = 6
TRADEOFF_DATA_MARKER_AREA = 42
TRADEOFF_LINE_WIDTH = 1.5
SCALING_TICK_LABEL_SIZE = 8.5
SCALING_LEGEND_FONT_SIZE = 7.5
SCALING_LEGEND_MARKER_SIZE = 6
SCALING_LINE_WIDTH = 1.25


@dataclass(frozen=True)
class VolumePlotStyle:
    """Visual settings shared by every frame in an incremental sequence."""

    palette: Dict[str, str]
    x_limits: Tuple[float, float]
    y_limits: Tuple[float, float]
    figsize: Tuple[float, float] = (3.5, 2.2)
    legend_fontsize: float = TRADEOFF_LEGEND_FONT_SIZE


@dataclass(frozen=True)
class RuntimePlotStyle:
    """Visual settings shared by runtime and storage comparison frames."""

    runtime_limits: Tuple[float, float]
    storage_limits: Tuple[float, float]
    figsize: Tuple[float, float] = (3.5, 2.2)
    legend_fontsize: float = TRADEOFF_LEGEND_FONT_SIZE


def _log_limits(values: pd.Series, padding_decades: float = 0.10) -> Tuple[float, float]:
    """Calculate fixed log-scale bounds with space around the extreme values."""
    minimum, maximum = values.min(), values.max()
    if minimum <= 0:
        raise ValueError("Log-scaled values must be positive.")
    return (
        10 ** (math.log10(minimum) - padding_decades),
        10 ** (math.log10(maximum) + padding_decades),
    )


def _decade_limits(values: pd.Series, padding_decades: float = 0.10) -> Tuple[float, float]:
    """Round log bounds out to labeled decades while retaining plot margins."""
    minimum, maximum = values.min(), values.max()
    if minimum <= 0:
        raise ValueError("Log-scaled values must be positive.")
    return (
        10 ** (math.floor(math.log10(minimum)) - padding_decades),
        10 ** (math.ceil(math.log10(maximum)) + padding_decades),
    )


def _select_experiments(
    df: pd.DataFrame,
    predicate: ExperimentPredicate,
) -> pd.DataFrame:
    """Apply a caller-supplied experiment predicate and validate its result."""
    mask = predicate(df)
    if not mask.index.equals(df.index) or mask.dtype != bool:
        raise ValueError("Experiment predicates must return a Boolean Series indexed like the input.")
    selected = df.loc[mask].copy()
    if type(selected) is not pd.DataFrame or selected.empty:
        raise ValueError("Experiment predicate selected no rows or a non-DataFrame object.")
    return selected


def load_composition_data(
    source_path: Path = data_folder / "composition_variation_experiment.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate the composition experiment data used by the paper plots."""
    results = pd.read_parquet(source_path)
    required_columns = {
        "QueryDescr", "b", "d", "Index", "Runtime_[ms]",
        "Total_Storage_Size_[GB]", "Team_Count", VOLUME_X, VOLUME_Y,
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Composition data is missing columns: {sorted(missing_columns)}")

    pareto_columns = COMPOSITION_GROUPS + ["Runtime_[ms]", "Total_Storage_Size_[GB]"]
    volume_columns = COMPOSITION_GROUPS + [VOLUME_Y, VOLUME_X, "Team_Count"]
    pareto_data = results[pareto_columns].groupby(COMPOSITION_GROUPS).mean().reset_index()
    volume_data = (
        results[volume_columns]
        .groupby(COMPOSITION_GROUPS)
        .agg({VOLUME_Y: "mean", VOLUME_X: "mean", "Team_Count": "max"})
        .reset_index()
    )
    return pareto_data, volume_data


def _make_volume_plot_style(df: pd.DataFrame) -> VolumePlotStyle:
    """Derive one immutable layout from the complete population of a sequence."""
    return VolumePlotStyle(
        palette=_make_flavor_d_palette(df),
        x_limits=_log_limits(df[VOLUME_X]),
        y_limits=_log_limits(df[VOLUME_Y]),
    )


def _make_flavor_d_palette(df: pd.DataFrame) -> Dict[str, str]:
    """Map each `(flavor, d)` pair to a stable shade of its flavor's hue."""
    d_values = sorted(df["d"].unique())
    if len(d_values) == 1:
        lightness = {d_values[0]: 0.65}
    else:
        lightness = {
            d_value: 0.42 + 0.42 * index / (len(d_values) - 1)
            for index, d_value in enumerate(d_values)
        }

    palette = {}
    for flavor, color in QUERY_FLAVOR_COLORS.items():
        hue, _, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
        for d_value in d_values:
            palette[f"{flavor},{d_value}"] = mcolors.to_hex(
                colorsys.hls_to_rgb(hue, lightness[d_value], saturation)
            )
    return palette


def _make_runtime_plot_style(df: pd.DataFrame, only_averages: bool = True) -> RuntimePlotStyle:
    """Derive limits from means or raw measurements, matching the display mode."""
    if only_averages:
        metric_columns = [RUNTIME_Y, STORAGE_Y]
        df = df[COMPOSITION_GROUPS + metric_columns].groupby(COMPOSITION_GROUPS).mean().reset_index()

    return RuntimePlotStyle(
        runtime_limits=_decade_limits(df[RUNTIME_Y], padding_decades=0.03),
        storage_limits=_log_limits(df[STORAGE_Y], padding_decades=0.03),
    )

def _add_query_flavor_legend(fig: plt.Figure, query_flavors: List[str]) -> None:
    """Add marker-only query flavor keys without duplicating the color encoding."""
    handles = [
        Line2D(
            [], [], linestyle="", color="black", marker=QUERY_FLAVOR_MARKERS[flavor],
            markersize=TRADEOFF_LEGEND_MARKER_SIZE, label=QUERY_FLAVOR_LABELS[flavor],
        )
        for flavor in query_flavors
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        handlelength=1.5,
        columnspacing=0.8,
    )


def _add_team_size_legend(
    fig: plt.Figure,
    style: VolumePlotStyle,
    d_values: List[int],
    team_counts: Dict[int, int],
    annotation_patch: Dict[int, str] = {6: ", 5, 5"}
) -> None:
    """Show paired flavor shades and the visible Team Size x Team Count values."""
    rows = []
    for d_value in d_values:
        swatches = DrawingArea(18, 8, 0, 0)
        swatches.add_artist(Line2D([4], [4], linestyle="", marker="o", markersize=TRADEOFF_LEGEND_MARKER_SIZE,
                                   color=style.palette[f"balanced_selective,{d_value}"]))
        swatches.add_artist(Line2D([13], [4], linestyle="", marker="^", markersize=TRADEOFF_LEGEND_MARKER_SIZE,
                                   color=style.palette[f"diverse,{d_value}"]))

        if d_value in annotation_patch:
            label = annotation_patch[d_value]
        else:
            label = f"  x {team_counts[d_value]}"
        rows.append(HPacker(
            children=[
                swatches,
                TextArea(str(d_value), textprops={"size": style.legend_fontsize}),
                TextArea(label, textprops={"size": style.legend_fontsize, "color": "#777777"}),
            ],
            align="center",
            pad=0,
            sep=1,
        ))
    legend = AnchoredOffsetbox(
        loc="center left",
        child=VPacker(
            children=[TextArea("Team size $d$", textprops={"size": style.legend_fontsize})] + rows,
            align="left",
            pad=0,
            sep=2,
        ),
        frameon=False,
        bbox_to_anchor=(0.78, 0.52),
        bbox_transform=fig.transFigure,
        borderpad=0,
        pad=0,
    )
    fig.add_artist(legend)


def _draw_direction_arrows(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw one red mean path per flavor, pointing toward increasing `d`."""
    for _, flavor_data in df.groupby("QueryDescr"):
        means = flavor_data.groupby("d", sort=True)[[VOLUME_X, VOLUME_Y]].mean()
        if len(means) < 2:
            continue

        ax.plot(means[VOLUME_X], means[VOLUME_Y], color=base_colors[3], linewidth=TRADEOFF_LINE_WIDTH, zorder=2)
        start, end = means.iloc[-2], means.iloc[-1]
        ax.annotate(
            "",
            xy=(end[VOLUME_X], end[VOLUME_Y]),
            xytext=(start[VOLUME_X], start[VOLUME_Y]),
            arrowprops={"arrowstyle": "->", "color": base_colors[3], "lw": TRADEOFF_LINE_WIDTH},
            zorder=4,
        )


def plot_volume_vs_overhead(
    df_plot: pd.DataFrame,
    file_path: Path,
    *,
    style: VolumePlotStyle,
    d_values: List[int],
    team_counts: Dict[int, int],
) -> None:
    """Plot a fixed-layout cumulative volume/overhead frame."""
    fig, ax = plt.subplots(figsize=style.figsize)

    plot_data = df_plot.copy()
    plot_data["(flavor,d)"] = plot_data["QueryDescr"] + "," + plot_data["d"].astype(str)
    sns.scatterplot(
        data=plot_data,
        x=VOLUME_X,
        y=VOLUME_Y,
        hue="(flavor,d)",
        style="QueryDescr",
        palette=style.palette,
        markers=QUERY_FLAVOR_MARKERS,
        s=TRADEOFF_DATA_MARKER_AREA,
        legend=False,
        ax=ax,
        zorder=3,
    )
    _draw_direction_arrows(ax, df_plot)
    ax.set(
        xscale="log",
        yscale="log",
        xlim=style.x_limits,
        ylim=style.y_limits,
    )
    ax.set_xlabel(VOLUME_X_LABEL_OVERRIDE or "Overhead", fontsize=10)
    ax.set_ylabel(VOLUME_Y_LABEL_OVERRIDE or "Volume [MB]", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=TRADEOFF_TICK_LABEL_SIZE)
    ax.annotate(
        r"$\leftarrow$ better",
        xy=(0.5, -0.40),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=TRADEOFF_ANNOTATION_FONT_SIZE,
        color="gray",
    )
    # With a 90-degree rotation, the left arrow points down toward the
    # lower-left corner, independently of the logarithmic axis limits.
    ax.annotate(
        r"$\leftarrow$ better",
        xy=(-0.43, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=TRADEOFF_ANNOTATION_FONT_SIZE,
        color="gray",
    )
    _add_query_flavor_legend(fig, sorted(df_plot["QueryDescr"].unique()))
    _add_team_size_legend(fig, style, sorted(df_plot["d"].unique()), team_counts)
    fig.subplots_adjust(left=0.25, right=0.75, top=0.84, bottom=0.29)
    fig.savefig(str(file_path), transparent=True)
    plt.close(fig)


def create_volume_vs_overhead_plot(
    source_path: Path = data_folder / "composition_variation_experiment.parquet",
    target_path: Path = plot_folder,
) -> None:
    """Create the complete access-volume versus overhead comparison at `b=10`."""
    _, volume_data = load_composition_data(source_path)
    target_path.mkdir(parents=True, exist_ok=True)
    sequence_data = _select_experiments(
        volume_data,
        lambda data: (data["b"] == 10) & data["QueryDescr"].isin(["balanced_selective", "diverse"]),
    )
    d_values = sorted(sequence_data["d"].unique().tolist())
    if len(d_values) != 4:
        raise ValueError(f"Expected four d values for b=10; found {d_values}.")

    style = _make_volume_plot_style(sequence_data)
    team_counts = {
        int(d_value): int(team_count)
        for d_value, team_count in sequence_data.groupby("d")["Team_Count"].max().items()
    }
    for query_flavor in ("balanced_selective", "diverse"):
        selected_flavor_data = _select_experiments(
            sequence_data,
            lambda data, flavor=query_flavor: data["QueryDescr"] == flavor,
        )
        missing_d = set(d_values).difference(selected_flavor_data["d"].unique())
        if missing_d:
            raise ValueError(f"{query_flavor} is missing d values: {sorted(missing_d)}")

    plot_volume_vs_overhead(
        sequence_data,
        target_path / "vol_vs_overhead_comparison.pdf",
        style=style,
        d_values=d_values,
        team_counts=team_counts,
    )


def _add_team_count_ticks(
    axis: plt.Axes,
    d_values: List[int],
    team_counts: Dict[int, int],
    patch = {6: ", 5, 5"}
) -> None:
    """Annotate each team-size tick with its maximum observed number of Teams."""
    axis.set_xticks(d_values, labels=[str(d_value) for d_value in d_values])
    for d_value in d_values:
        if d_value in patch:
            label = patch[d_value]
        else:
            label = f"x {team_counts[d_value]}"
        axis.annotate(
            label,
            xy=(d_value, 0),
            xycoords=("data", "axes fraction"),
            xytext=(12, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            color="#777777",
            fontsize=TRADEOFF_ANNOTATION_FONT_SIZE,
        )


def plot_runtime(
    df_plot: pd.DataFrame,
    file_path: Path,
    *,
    style: RuntimePlotStyle,
    query_flavors: List[str],
    d_values: List[int],
    team_counts: Dict[int, int],
    only_averages: bool = True,
) -> None:
    """Plot raw runtime measurements for the currently visible query flavors."""
    fig, axis = plt.subplots(figsize=style.figsize)
    flavor_colors = QUERY_FLAVOR_COLORS
    flavor_markers = QUERY_FLAVOR_MARKERS
    visible_data = df_plot.loc[df_plot["QueryDescr"].isin(query_flavors)]
    for query_flavor in query_flavors:
        means = (
            visible_data.loc[visible_data["QueryDescr"] == query_flavor]
            .groupby("d", sort=True)[RUNTIME_Y]
            .mean()
        )
        axis.plot(
            means.index,
            means.values,
            marker=flavor_markers[query_flavor] if only_averages else None,
            color=flavor_colors[query_flavor],
            linestyle="--",
            linewidth=TRADEOFF_LINE_WIDTH,
            zorder=2,
        )
    if not only_averages:
        sns.scatterplot(
            data=visible_data,
            x="d",
            y=RUNTIME_Y,
            hue="QueryDescr",
            style="QueryDescr",
            palette=flavor_colors,
            markers=flavor_markers,
            s=TRADEOFF_DATA_MARKER_AREA,
            legend=False,
            ax=axis,
            zorder=3,
        )
    axis.set(
        xlim=(min(d_values) - 0.25, max(d_values) + 0.25),
        yscale="log",
        ylim=style.runtime_limits,
        xlabel=RUNTIME_X_LABEL_OVERRIDE or "Team size",
        ylabel=RUNTIME_Y_LABEL_OVERRIDE or "Runtime [ms]",
    )
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    axis.tick_params(axis="both", which="major", labelsize=TRADEOFF_TICK_LABEL_SIZE)
    _add_team_count_ticks(axis, d_values, team_counts)
    metric_legend_handles = [
        Line2D([], [], color="black", linestyle="", marker=flavor_markers[flavor], markersize=TRADEOFF_LEGEND_MARKER_SIZE,
             label=QUERY_FLAVOR_LABELS[flavor])
        for flavor in query_flavors
    ]
    fig.legend(
        handles=metric_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(metric_legend_handles),
        frameon=False,
        fontsize=style.legend_fontsize,
        handlelength=1.5,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.20, right=0.95, top=0.8, bottom=0.30)
    fig.savefig(str(file_path), transparent=True)
    plt.close(fig)


def plot_storage(
    df_plot: pd.DataFrame,
    file_path: Path,
    *,
    style: RuntimePlotStyle,
    d_values: List[int],
    team_counts: Dict[int, int],
) -> None:
    """Plot the shared storage curve independently from runtime measurements."""
    fig, axis = plt.subplots(figsize=style.figsize)
    means = df_plot.groupby("d", sort=True)[STORAGE_Y].mean()
    axis.plot(
        means.index,
        means.values,
        color=STORAGE_AXIS_COLOR,
        linestyle="--",
        linewidth=TRADEOFF_LINE_WIDTH,
        marker="s",
        markersize=TRADEOFF_LEGEND_MARKER_SIZE + 1,
        markeredgewidth=1.0,
        markeredgecolor="white",
        zorder=3,
    )
    axis.set(
        xlim=(min(d_values) - 0.25, max(d_values) + 0.25),
        yscale="log",
        ylim=style.storage_limits,
        xlabel=RUNTIME_X_LABEL_OVERRIDE or "Team size",
        ylabel=STORAGE_Y_LABEL_OVERRIDE or "Storage [GB]",
        yticks=[10, 12, 14, 16, 18, 20],
        yticklabels=[str(tick) for tick in [10, 12, 14, 16, 18, 20]],
    )
    axis.minorticks_off()
    axis.tick_params(axis="both", which="major", labelsize=TRADEOFF_TICK_LABEL_SIZE)
    axis.tick_params(axis="y", colors=STORAGE_AXIS_COLOR)
    axis.yaxis.label.set_color(STORAGE_AXIS_COLOR)
    axis.spines["left"].set_color(STORAGE_AXIS_COLOR)
    _add_team_count_ticks(axis, d_values, team_counts)
    fig.subplots_adjust(left=0.20, right=0.95, top=0.88, bottom=0.30)
    fig.savefig(str(file_path), transparent=True)
    plt.close(fig)


def create_runtime_and_storage_plots(
    source_path: Path = data_folder / "composition_variation_experiment.parquet",
    target_path: Path = plot_folder,
    only_averages: bool = True
) -> None:
    """Create the two-flavor runtime comparison and shared storage-size plot."""
    results = pd.read_parquet(source_path)
    target_path.mkdir(parents=True, exist_ok=True)
    sequence_data = _select_experiments(
        results,
        lambda data: (data["b"] == 10) & data["QueryDescr"].isin(["balanced_selective", "diverse"]),
    )
    d_values = sorted(sequence_data["d"].unique().tolist())
    if len(d_values) != 4:
        raise ValueError(f"Expected four d values for b=10; found {d_values}.")

    for query_flavor in ("balanced_selective", "diverse"):
        flavor_d_values = set(sequence_data.loc[sequence_data["QueryDescr"] == query_flavor, "d"].unique())
        missing_d = set(d_values).difference(flavor_d_values)
        if missing_d:
            raise ValueError(f"{query_flavor} is missing d values: {sorted(missing_d)}")

    team_counts = {
        int(d_value): int(team_count)
        for d_value, team_count in sequence_data.groupby("d")["Team_Count"].max().items()
    }
    style = _make_runtime_plot_style(sequence_data, only_averages=only_averages)
    plot_runtime(
        sequence_data,
        target_path / "runtime_comparison.pdf",
        style=style,
        query_flavors=["balanced_selective", "diverse"],
        d_values=d_values,
        team_counts=team_counts,
        only_averages=only_averages,
    )
    plot_storage(
        sequence_data,
        target_path / "storage_comparison.pdf",
        style=style,
        d_values=d_values,
        team_counts=team_counts,
    )



def plot_selectivity_runtime(
    results: pd.DataFrame,
    *,
    line_by=list(('s', 'Index')),
    color_by=("s",),
    symbol_by=("Index",),
    order_by="Query Dim., D",
    y_attribute="Runtime [s]",
    x_attribute="Query Dim., D",
    x_axis_label: Optional[str] = None,
    y_axis_label: Optional[str] = None,
    dotted_value=("Any","VA"),
    zero_filter_attribute="Selectivity",
    markersize=5,
    log_y_axis=True,
    invert_x_axis=False,
    direction_label=None,
    x_tick_steps=(5,85,5),
    target_path=None,
    figure_size=(3.33, 1.6),
    colors: Optional[List[str]] = None,
    marker_lookup: Optional[Dict[Tuple[str, ...], str]] = None,
    color_legend_title: Optional[str] = None,
    symbol_legend_title: Optional[str] = None,
    style_label_overrides: Optional[Dict[Tuple[str, ...], str]] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    ax=None,
):
    """
    Plot runtime vs. query dimensionality.

    - Line *color* is determined by the tuple of columns in `color_by` (default: ("s",)).
    - Marker *symbol* is determined by the tuple of columns in `symbol_by` (default: ("index",)).
    - Any remaining differentiating columns (not in color_by or symbol_by) lead to distinct series
      with their own line style (same color and marker if groups match).
    - Points within a series are ordered by `order_by` (default: "team_count").
    - Selectivity = result_cardinality / N. Zero/negative selectivities (or runtimes) are skipped.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure

    needed = {y_attribute} | set(color_by) | set(symbol_by) | {order_by}
    missing = [c for c in needed if c not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = results.copy()

    # For log axes, keep only strictly positive values
    # if log_y_axis:
    #     df = df[(df[x_attribute] > 0) & (df[y_attribute] > 0)]
    if df.empty:
        raise ValueError("No positive selectivity/runtime rows to plot after filtering.")

    # Build color map over unique color groups
    color_key_df = df[list(color_by)].drop_duplicates()
    color_keys = [tuple(row[c] for c in color_by) for _, row in color_key_df.iterrows()]

    # Cycle through colors
    colors = colors or base_colors
    repeated_colors = colors * (len(color_keys) // len(colors) + 1)
    color_map = {ck: col for ck, col in zip(color_keys, repeated_colors[:len(color_keys)])}

    # Distinct marker per *symbol group*
    marker_sequence = [
        "^", "+", "s", "*", "D", "P", "X", "v", "<", ">", "h", "x", "1", "2", "3", "4",
    ]
    marker_map = {}

    def fmt_key(cols, vals, pfx=""):
        if not isinstance(vals, tuple):
            vals = (vals,)
        return " / ".join(pfx+f"{v}" for c, v in zip(cols, vals))

    # Plot each series line
    for line_key, sub in df.groupby(list(line_by), sort=False):
        is_dotted = line_key == dotted_value
        sub = sub.sort_values(order_by)

        x = sub[x_attribute].to_numpy()
        y = sub[y_attribute].to_numpy()
        sel = sub[zero_filter_attribute].to_numpy()


        # Color group
        cg_tuple = tuple(sub[c].iloc[0] for c in color_by)
        color = color_map[cg_tuple]

        # Marker symbol
        ## only add marker if attribute value is not equal to the dotted value
        sg_tuple = tuple(sub[c].iloc[0] for c in symbol_by)
        if sg_tuple not in marker_map:
            if is_dotted:
                marker_map[sg_tuple] = "o"
            else:
                marker_map[sg_tuple] = (marker_lookup or {}).get(
                    sg_tuple,
                    marker_sequence[len(marker_map) % len(marker_sequence)],
                )
        marker = marker_map[sg_tuple]


        # draw markers
        for i in range(len(x)):
            if is_dotted:
                markersize_ = markersize * 0.75
                marker = "o"
                mcolor = "gray"
            else:
                mcolor = color
                markersize_ = markersize
            ax.plot(x[i:i+1], y[i:i+1], color=mcolor, markersize=markersize_, marker=marker,
                    linestyle="None", alpha=1.0 if sel[i] != 0 else 0.33)

        # draw segments
        for i in range(len(x) - 1):
            # draw dotted line if one of the two points has the dotted value in the respective column
            if is_dotted:
                ax.plot(x[i:i+2], y[i:i+2], color="gray", linestyle=":", linewidth=SCALING_LINE_WIDTH)
            elif sel[i] == 0 or sel[i+1] == 0:
                ax.plot(x[i:i+2], y[i:i+2], color="gray", linestyle="--", linewidth=SCALING_LINE_WIDTH)
            else:
                ax.plot(x[i:i+2], y[i:i+2], color=color, linestyle="-", linewidth=SCALING_LINE_WIDTH)


    if log_y_axis:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value:.{max(0, -math.floor(math.log10(value)))}f}"
        ))

    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])
    if y_limits is not None:
        ax.set_ylim(bottom=y_limits[0], top=y_limits[1])

    # ax.set_xscale("log")
    
    ax.set_xlabel(x_attribute.replace("_", " "))
    ax.set_ylabel(y_attribute.replace("_", " "))

    xtick_start, xtick_end, tick_stepsize = x_tick_steps
    ax.xaxis.set_major_locator(MultipleLocator(base=tick_stepsize))

    def format_alternate_tick(value: float, _: int) -> str:
        tick_number = round((value - xtick_start) / tick_stepsize)
        if math.isclose(value, xtick_start + tick_number * tick_stepsize) and tick_number % 2 == 0:
            return str(int(value))
        return ""

    ax.xaxis.set_major_formatter(FuncFormatter(format_alternate_tick))
    ax.tick_params(axis="both", which="major", labelsize=SCALING_TICK_LABEL_SIZE)

    if invert_x_axis:
        ax.invert_xaxis()

    # Separate legends for colors and symbols
    dotted_color_key = tuple(
        dotted_value[list(line_by).index(column)]
        for column in color_by
    )
    color_legend_items = [
        (color_key, color)
        for color_key, color in color_map.items()
        if color_key != dotted_color_key
    ]
    if dotted_color_key in color_map:
        color_legend_items.append((dotted_color_key, "gray"))
    color_handles = [
        Line2D(
            [0], [0], color=color, lw=SCALING_LINE_WIDTH,
            linestyle=":" if color_key == dotted_color_key else "-",
        )
        for color_key, color in color_legend_items
    ]
    color_labels = [fmt_key(color_by, color_key) for color_key, _ in color_legend_items]
    legend1 = ax.legend(
        color_handles,
        color_labels,
        title=color_legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=False,
        fontsize=SCALING_LEGEND_FONT_SIZE,
        title_fontsize=SCALING_LEGEND_FONT_SIZE,
        labelspacing=0.2,
        handletextpad=0.3,
    )
    ax.add_artist(legend1)

    symbol_handles = [
        Line2D(
            [0], [0], color="black", marker=marker, linestyle="None",
            markersize=SCALING_LEGEND_MARKER_SIZE,
        )
        for marker in marker_map.values()
    ]
    symbol_labels = [
        (style_label_overrides or {}).get(style_key, fmt_key(symbol_by, style_key))
        for style_key in marker_map.keys()
    ]
    style_legend = ax.legend(
        symbol_handles,
        symbol_labels,
        title=symbol_legend_title,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        ncol=min(len(symbol_handles), 2),
        frameon=False,
        fontsize=SCALING_LEGEND_FONT_SIZE,
        title_fontsize=SCALING_LEGEND_FONT_SIZE,
        labelspacing=0.2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    
    # Add annotation pointing right from lower right corner
    if direction_label is not None:
        ax.annotate(
            direction_label,
            xy=(0.95, 0.05), xycoords="axes fraction",  # a bit inside the lower-right
            xytext=(-40, 0), textcoords="offset points",
            ha="right", va="center",
            arrowprops=dict(arrowstyle="->", lw=1)
        )

    fig.subplots_adjust(
        left=0.16,
        right=0.78,
        bottom=0.24,
        top=0.80,
    )
    # ax.tick_params(pad=2)   # default is ~4
    # ax.xaxis.labelpad = 1
    # ax.yaxis.labelpad = 1

    if x_axis_label is not None:
        ax.set_xlabel(x_axis_label)

    if y_axis_label is not None:
        ax.set_ylabel(y_axis_label)

    ## dump figure to pdf:
    if target_path:
        fig.savefig(
            target_path,
            transparent=True,
        )

    return ax

def _load_sdss_scaling_data(data_path: Path) -> pd.DataFrame:
    """Load the SDSS measurements used by both dimensional-scaling figures."""
    data = pd.read_parquet(data_path)
    missing_columns = SDSS_REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"SDSS scaling data is missing columns: {sorted(missing_columns)}")
    return data


def _aggregate_sdss_scaling_data(data: pd.DataFrame) -> pd.DataFrame:
    """Average repeated SDSS measurements for each plotted series and dimension."""
    return data.groupby(["s", "Index", "D"]).mean(numeric_only=True).reset_index()


def _plot_sdss_scaling(
    data: pd.DataFrame,
    target_path: Path,
    *,
    style_label_overrides: Optional[Dict[Tuple[str],str]] = SDSS_STYLE_LABELS,
    x_axis_label: Optional[str] = SDSSS_X_AXIS_LABEL,
    y_axis_label: Optional[str] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    """Draw one SDSS dimensional-scaling frame with the shared slide layout."""
    plot_selectivity_runtime(
        data,
        line_by=["s", "Index"],
        symbol_by=("Index",),
        color_by=("s",),
        order_by="D",
        x_attribute="D",
        y_attribute="Runtime [s]",
        dotted_value=SDSS_VA_KEY,
        figure_size=(4.0, 3.7),
        markersize=SCALING_LEGEND_MARKER_SIZE,
        colors=SDSS_COLORS,
        marker_lookup=SDSS_MARKERS,
        color_legend_title="Predicate sel.",
        style_label_overrides=style_label_overrides,
        x_limits=x_limits,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        y_limits=y_limits,
        target_path=target_path,
    )


def create_dimensional_scaling_plot(
    data_path: Path = data_folder / "SDSS_dimensional_scaling.parquet",
    target_path: Path = plot_folder,
) -> None:
    """Create the dimensional-scaling comparison for all selectivities and VA."""
    data = _load_sdss_scaling_data(data_path)

    automatic_data = data.loc[data["Index"] == SDSS_AUTO_INDEX].copy()
    assert type(automatic_data) is pd.DataFrame

    va_data = data.loc[(data["s"] == SDSS_VA_KEY[0]) & (data["Index"] == SDSS_VA_KEY[1])].copy()
    assert type(va_data) is pd.DataFrame

    if automatic_data.empty:
        raise ValueError(f"No automatic-team rows found for Index={SDSS_AUTO_INDEX!r}.")
    if va_data.empty:
        raise ValueError(f"No VA reference rows found for {SDSS_VA_KEY}.")
    invalid_d_values = automatic_data.loc[
        automatic_data["d"].notna() & ~automatic_data["d"].eq(5), "d"
    ].unique()
    if len(invalid_d_values) or not automatic_data["Index"].eq(SDSS_AUTO_INDEX).all():
        raise ValueError("Automatic-team filter returned unexpected rows.")

    selectivities = sorted(value for value in automatic_data["s"].unique() if value != SDSS_VA_KEY[0])
    if not selectivities:
        raise ValueError("No numeric selectivity groups found for automatic-team rows.")

    automatic_means = _aggregate_sdss_scaling_data(automatic_data)
    va_means = _aggregate_sdss_scaling_data(va_data)

    ## PATCH for runtime estimation. Old data had 3 bytes, not 3 bits per value in the VA files.
    # KIOXIA CM7-R 2TB sequential read (raw volume for bit-packed storage computed!)
    # VA files can use full bandwidth.
    # Overall an optimistic upper bound for their runtime
    PCIe_5_0_bandwidth = 14  # GB/s
    va_means["Runtime [s]"] = va_means["D"] * 0.434*10**9 * 3 / 8.0 / 1e9 / PCIe_5_0_bandwidth

    full_data = pd.concat([automatic_means, va_means], ignore_index=True)
    x_limits = (full_data["D"].min(), full_data["D"].max())
    y_limits = _log_limits(full_data["Runtime [s]"], padding_decades=0.03)
    target_path.mkdir(parents=True, exist_ok=True)

    frame_data = pd.concat([
        automatic_means,
        va_means,
    ], ignore_index=True)
    _plot_sdss_scaling(
        frame_data,
        x_limits=x_limits,
        y_limits=y_limits,
        target_path=target_path / "dimensional_scaling_comparison.pdf",
    )


def print_dimensional_scaling_io_statistics(
    data_path: Path = data_folder / "SDSS_dimensional_scaling.parquet",
) -> None:
    """Print the accessed-volume statistics behind the I/O ceiling plot."""

    comparison_dimension = 85
    comparison_selectivity = 0.375
    ## Note: "total_list_size" is in KiB!!!
    data = _load_sdss_scaling_data(data_path)
    automatic_data = data.loc[data["Index"] == SDSS_AUTO_INDEX].copy()
    automatic_data["s"] = pd.to_numeric(automatic_data["s"], errors="coerce")
    automatic_data = automatic_data.dropna(subset=["s"])
    if automatic_data.empty:
        raise ValueError(f"No automatic-team rows found for Index={SDSS_AUTO_INDEX!r}.")

    automatic_means = _aggregate_sdss_scaling_data(automatic_data)
    selectivities = sorted(automatic_means["s"].unique())
    dimensions = sorted(automatic_means["D"].unique())
    va_volume_gib = {
        dimension: dimension * 0.434 * 10**9 * 3 / 8.0 / 1024**3
        for dimension in dimensions
    }
    random_io_bandwidth = 4096 * 2_000_000 / 1e9
    automatic_means["I/O ceiling [s]"] = (
        automatic_means["total_list_size"] * 1024 / 1e9 / random_io_bandwidth
    )
    if (automatic_means["Runtime [s]"] <= 0).any():
        raise ValueError("Observed runtimes must be positive to calculate an I/O performance gap.")
    automatic_means["Observed / I/O ceiling"] = (
        automatic_means["Runtime [s]"] / automatic_means["I/O ceiling [s]"]
    )

    print("\nI/O ceiling statistics (SDSS dimensional scaling)")
    print(f"  Inverted index: {SDSS_AUTO_INDEX}; ideal random-read bandwidth: {random_io_bandwidth:.2f} GB/s")
    print("  Volumes are mean accessed compressed-list data per query; VA is a 3-bit/value full scan.\n")

    def format_selectivity(selectivity: object) -> str:
        try:
            return f"{float(selectivity):g}"
        except (TypeError, ValueError):
            return str(selectivity)

    header = "  D  " + "  ".join(
        f"s={format_selectivity(selectivity):<5}"
        for selectivity in selectivities
    ) + "  VA scan"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dimension in dimensions:
        volumes = []
        for selectivity in selectivities:
            row = automatic_means.loc[
                (automatic_means["D"] == dimension) & (automatic_means["s"] == selectivity)
            ]
            if len(row) != 1:
                raise ValueError(f"Expected one {SDSS_AUTO_INDEX} row for s={selectivity}, D={dimension}.")
            volume_gib = row.iloc[0]["total_list_size"] / 1024**2  # KiB to GiB
            volumes.append(f"{volume_gib:6.3f} GiB")
        print(f"  {dimension:>2}  " + "  ".join(volumes) + f"  {va_volume_gib[dimension]:6.3f} GiB")

    print("\n  Per-selectivity summary")
    for selectivity in selectivities:
        rows = automatic_means.loc[automatic_means["s"] == selectivity]
        volumes_gib = rows["total_list_size"] / 1024**2  # KiB to GiB
        mean_list_sizes_kib = rows["total_list_size"] / rows["total_list_count"]
        page_amplification_upper = 1 + 4 / mean_list_sizes_kib
        print(
            f"  s={format_selectivity(selectivity)}: "
            f"accessed volume {volumes_gib.min():.3f}-{volumes_gib.max():.3f} GiB; "
            f"mean list size {mean_list_sizes_kib.min():.1f}-{mean_list_sizes_kib.max():.1f} KiB; "
            f"4-KiB page rounding < {page_amplification_upper.max():.3f}x"
        )

    print("\n  Observed-runtime gap over the ideal random-I/O ceiling")
    print("  (Observed runtime includes compute and software overhead; the ceiling models I/O only.)")
    for selectivity in selectivities:
        rows = automatic_means.loc[automatic_means["s"] == selectivity]
        observed_seconds = rows["Runtime [s]"]
        io_ceiling_seconds = rows["I/O ceiling [s]"]
        gap = rows["Observed / I/O ceiling"]
        print(
            f"  s={format_selectivity(selectivity)}: "
            f"observed {observed_seconds.min():.3f}-{observed_seconds.max():.3f} s; "
            f"I/O ceiling {io_ceiling_seconds.min():.3f}-{io_ceiling_seconds.max():.3f} s; "
            f"gap {gap.min():.1f}-{gap.max():.1f}x"
        )

    comparison_index = automatic_means.loc[
        (automatic_means["D"] == comparison_dimension)
        & (automatic_means["s"] == comparison_selectivity)
    ]
    if len(comparison_index) != 1:
        raise ValueError(
            "Expected one inverted-index row for the "
            f"D={comparison_dimension}, s={comparison_selectivity:g} comparison."
        )

    index_row = comparison_index.iloc[0]
    va_io_ceiling_seconds = (
        comparison_dimension * 0.434 * 10**9 * 3 / 8.0 / 1e9 / 14
    )
    observed_vs_ideal_va_speedup = (
        va_io_ceiling_seconds / index_row["Runtime [s]"]
    )
    io_ceiling_speedup = va_io_ceiling_seconds / index_row["I/O ceiling [s]"]
    print("\n  D=85, s=0.375: inverted index vs. VA-file")
    print(
        f"  observed inverted runtime: {index_row['Runtime [s]']:.3f} s; "
        f"ideal VA I/O lower bound: {va_io_ceiling_seconds:.3f} s; "
        f"inverted index is at least {observed_vs_ideal_va_speedup:.1f}x faster"
    )
    print(
        f"  ideal I/O: inverted {index_row['I/O ceiling [s]']:.3f} s; "
        f"VA {va_io_ceiling_seconds:.3f} s; "
        f"inverted index {io_ceiling_speedup:.1f}x faster"
    )

def create_io_bandwidth_dimensional_scaling_plot(
    data_path: Path = data_folder / "SDSS_dimensional_scaling.parquet",
    target_path: Path = plot_folder / "dimensional_scaling_io_bandwidth.pdf",
) -> None:
    """Create the all-series SDSS plot using the estimated I/O bandwidth model."""
    data = _load_sdss_scaling_data(data_path)

    automatic_data = data.loc[data["Index"] == SDSS_AUTO_INDEX].copy()
    va_data = data.loc[(data["s"] == SDSS_VA_KEY[0]) & (data["Index"] == SDSS_VA_KEY[1])].copy()
    if automatic_data.empty:
        raise ValueError(f"No automatic-team rows found for Index={SDSS_AUTO_INDEX!r}.")
    if va_data.empty:
        raise ValueError(f"No VA reference rows found for {SDSS_VA_KEY}.")

    automatic_means = _aggregate_sdss_scaling_data(automatic_data)
    va_means = _aggregate_sdss_scaling_data(va_data)

    # KIOXIA CM7-R: ideal random-read I/O bottleneck.
    # `total_list_size` is the accessed compressed-list volume in KiB.
    # This mirrors the VA curve's ideal sequential-I/O calculation: both omit
    # computation and model only the respective storage access bottleneck.
    random_io_bandwidth = 4096 * 2_000_000 / 1e9  # 4 KiB x 2 million random reads = ~8.2 GB/s
    automatic_means["Runtime [s]"] = (
        automatic_means["total_list_size"] * 1024 / 1e9 / random_io_bandwidth
    )

    # KIOXIA CM7-R 2TB: sequential read. (Volume computed!)
    sequential_bandwidth = 14
    # VA files can use full bandwidth.
    va_means["Runtime [s]"] = va_means["D"] * 0.434 * 10**9 * 3 / 8.0 / 1e9 / sequential_bandwidth

    full_data = pd.concat([automatic_means, va_means], ignore_index=True)
    color_order = [
        *sorted(value for value in full_data["s"].unique() if value != SDSS_VA_KEY[0]),
        SDSS_VA_KEY[0],
    ]
    full_data = pd.concat(
        [
            full_data.loc[full_data["s"] == selectivity]
            for selectivity in color_order
        ],
        ignore_index=True,
    )
    x_limits = (full_data["D"].min(), full_data["D"].max())
    y_limits = _log_limits(full_data["Runtime [s]"], padding_decades=0.05)
    _plot_sdss_scaling(
        full_data,
        x_limits=x_limits,
        y_limits=y_limits,
        target_path=target_path,
        y_axis_label="Raw I/O runtime [s]",
        style_label_overrides=SDSS_STYLE_LABELS_2
    )

if __name__ == "__main__":
    print("Creating figures...")
    create_volume_vs_overhead_plot()
    create_runtime_and_storage_plots(only_averages=False)
    create_dimensional_scaling_plot()
    create_io_bandwidth_dimensional_scaling_plot()
    print("Done.")
