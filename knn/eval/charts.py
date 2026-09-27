import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.pipeline import Pipeline

from ..utils import ACTIVITY_LABELS, EvaluationMetrics

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_MARKERS = ("o", "s", "D")

SEQUENTIAL_BLUE = (
    SURFACE,
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

DPI = 160

CONFUSION_MATRIX_FILE = "confusion_matrix.png"
CLASS_METRICS_FILE = "class_metrics.png"
CV_NEIGHBOURS_FILE = "cv_neighbours.png"
LDA_PROJECTION_FILE = "lda_projection.png"


def _new_figure(
    width: float,
    height: float,
) -> Figure:
    return Figure(
        figsize=(width, height),
        dpi=DPI,
        facecolor=SURFACE,
        layout="constrained",
    )


def _add_titles(
    figure: Figure,
    title: str,
    subtitle: str,
) -> None:
    figure.suptitle(
        f"{title}\n",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=TEXT_PRIMARY,
    )
    figure.text(
        0.01,
        0.985,
        f"\n{subtitle}",
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_SECONDARY,
    )


def _style_axes(
    ax: Axes,
    grid_axis: Literal["both", "x", "y"] = "y",
) -> None:
    ax.set_facecolor(SURFACE)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(
        colors=TEXT_MUTED,
        labelcolor=TEXT_SECONDARY,
        labelsize=8,
        length=0,
    )
    ax.grid(
        axis=grid_axis,
        color=GRIDLINE,
        linewidth=0.75,
        linestyle="-",
    )
    ax.set_axisbelow(True)


def _save(
    figure: Figure,
    path: Path,
) -> Path:
    figure.savefig(
        path,
        facecolor=SURFACE,
    )
    logger.info(
        "Chart saved to %s",
        path,
    )

    return path


def plot_confusion_matrix(
    metrics: EvaluationMetrics,
    path: Path,
) -> Path:
    names = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS)]
    counts = np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )
    shares = counts / counts.sum(axis=1, keepdims=True)

    figure = _new_figure(8.5, 6.8)
    ax = figure.add_subplot()
    image = ax.imshow(
        shares,
        cmap=LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE),
        vmin=0.0,
        vmax=1.0,
    )

    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            ink = SURFACE if shares[row, column] > 0.55 else TEXT_PRIMARY
            ax.text(
                column,
                row - 0.1,
                f"{counts[row, column]}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="semibold",
                color=ink,
            )
            ax.text(
                column,
                row + 0.2,
                f"{shares[row, column]:.0%}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=ink,
            )

    ax.set_xticks(
        range(len(names)),
        names,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        range(len(names)),
        names,
    )
    ax.set_xlabel(
        "Predicted activity",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.set_ylabel(
        "True activity",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.tick_params(
        length=0,
        labelsize=8,
        labelcolor=TEXT_SECONDARY,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(
        image,
        ax=ax,
        shrink=0.8,
        format=lambda value, _: f"{value:.0%}",
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(
        length=0,
        labelsize=8,
        labelcolor=TEXT_SECONDARY,
    )
    colorbar.set_label(
        "Share of true activity",
        color=TEXT_SECONDARY,
        fontsize=9,
    )

    _add_titles(
        figure,
        "Confusion matrix on the test set",
        f"Counts and row share; accuracy {metrics['accuracy']:.2%}, "
        f"{int(counts.sum())} windows",
    )

    return _save(figure, path)


def plot_class_metrics(
    metrics: EvaluationMetrics,
    path: Path,
) -> Path:
    names = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS)]
    scores = {
        metric: [float(metrics["report"][name][metric]) for name in names]
        for metric in ("precision", "recall", "f1-score")
    }
    lowest = min(min(values) for values in scores.values())

    figure = _new_figure(8.5, 5.2)
    ax = figure.add_subplot()
    _style_axes(ax, grid_axis="x")
    ax.spines["bottom"].set_visible(False)

    rows = np.arange(len(names))[::-1]
    offsets = (0.22, 0.0, -0.22)

    for index, (metric, values) in enumerate(scores.items()):
        ax.plot(
            values,
            rows + offsets[index],
            linestyle="none",
            marker=SERIES_MARKERS[index],
            markersize=8,
            markerfacecolor=SERIES[index],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=metric.replace("-score", "").capitalize(),
        )

    ax.set_yticks(
        rows,
        names,
    )
    ax.set_xlim(
        max(0.0, np.floor(lowest * 50) / 50 - 0.02),
        1.005,
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel(
        "Score",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=3,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_SECONDARY,
        handletextpad=0.3,
        columnspacing=1.2,
    )

    _add_titles(
        figure,
        "Per-activity precision, recall and F1",
        f"Test set; macro-F1 {metrics['f1_macro']:.2%}. "
        "The axis starts near the lowest score to show the differences",
    )

    return _save(figure, path)


def plot_cv_neighbours(
    cv_results: Mapping[str, Any],
    best_neighbours: int,
    path: Path,
) -> Path:
    results = pd.DataFrame(
        {
            "k": np.asarray(cv_results["param_knn__n_neighbors"], dtype=np.int64),
            "mean": np.asarray(cv_results["mean_test_score"], dtype=np.float64),
            "std": np.asarray(cv_results["std_test_score"], dtype=np.float64),
        }
    )
    best_per_k = (
        results.sort_values("mean", ascending=False)
        .drop_duplicates("k")
        .sort_values("k")
    )
    k = best_per_k["k"].to_numpy()
    mean = best_per_k["mean"].to_numpy()
    std = best_per_k["std"].to_numpy()

    figure = _new_figure(8.5, 4.8)
    ax = figure.add_subplot()
    _style_axes(ax)

    ax.fill_between(
        k,
        mean - std,
        mean + std,
        color=SERIES[0],
        alpha=0.1,
        linewidth=0,
    )
    ax.plot(
        k,
        mean,
        color=SERIES[0],
        linewidth=2,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker="o",
        markersize=8,
        markerfacecolor=SERIES[0],
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )

    chosen = int(np.flatnonzero(k == best_neighbours)[0])
    ax.annotate(
        f"Chosen k={best_neighbours}: {mean[chosen]:.2%}",
        xy=(k[chosen], mean[chosen]),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        fontweight="semibold",
        color=TEXT_PRIMARY,
    )

    ax.set_xticks(k)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    ax.set_xlabel(
        "Number of neighbours (k)",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.set_ylabel(
        "CV macro-F1",
        color=TEXT_SECONDARY,
        fontsize=9,
    )

    _add_titles(
        figure,
        "Cross-validated macro-F1 by number of neighbours",
        "Best configuration per k on the training set; band shows ±1 std across folds",
    )

    return _save(figure, path)


def plot_lda_projection(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: "pd.Series[int]",
    path: Path,
) -> Path:
    projected: npt.NDArray[np.float64] = np.asarray(
        model.named_steps["lda"].transform(X_test),
        dtype=np.float64,
    )
    labels = y_test.to_numpy()

    figure = _new_figure(10.5, 6.8)
    axes = figure.subplots(
        2,
        3,
        sharex=True,
        sharey=True,
    )

    for ax, activity_id in zip(axes.flat, sorted(ACTIVITY_LABELS), strict=True):
        _style_axes(ax, grid_axis="both")
        selected = labels == activity_id

        ax.scatter(
            projected[~selected, 0],
            projected[~selected, 1],
            s=4,
            color=BASELINE,
            alpha=0.35,
            linewidths=0,
        )
        ax.scatter(
            projected[selected, 0],
            projected[selected, 1],
            s=9,
            color=SERIES[0],
            alpha=0.7,
            linewidths=0,
        )
        ax.set_title(
            f"{ACTIVITY_LABELS[activity_id]} (n={int(selected.sum())})",
            loc="left",
            fontsize=9,
            color=TEXT_PRIMARY,
        )

    for ax in axes[1, :]:
        ax.set_xlabel(
            "Discriminant 1",
            color=TEXT_SECONDARY,
            fontsize=8,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel(
            "Discriminant 2",
            color=TEXT_SECONDARY,
            fontsize=8,
        )

    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                color=SERIES[0],
                label="Highlighted activity",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                color=BASELINE,
                label="All other activities",
            ),
        ],
        loc="outside lower left",
        ncols=2,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_SECONDARY,
    )

    _add_titles(
        figure,
        "Test windows in the first two LDA dimensions",
        "The space KNN searches for neighbours (2 of 5 dimensions shown)",
    )

    return _save(figure, path)


def save_reports(
    model: Pipeline,
    cv_results: Mapping[str, Any],
    metrics: EvaluationMetrics,
    X_test: pd.DataFrame,
    y_test: "pd.Series[int]",
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return [
        plot_confusion_matrix(
            metrics,
            output_dir / CONFUSION_MATRIX_FILE,
        ),
        plot_class_metrics(
            metrics,
            output_dir / CLASS_METRICS_FILE,
        ),
        plot_cv_neighbours(
            cv_results,
            int(model.named_steps["knn"].n_neighbors),
            output_dir / CV_NEIGHBOURS_FILE,
        ),
        plot_lda_projection(
            model,
            X_test,
            y_test,
            output_dir / LDA_PROJECTION_FILE,
        ),
    ]
