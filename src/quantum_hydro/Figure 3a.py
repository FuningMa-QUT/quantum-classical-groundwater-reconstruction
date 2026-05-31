"""Figure 3a: current synthetic aquifer configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantum_hydro.kfields import generate_k_field


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "scenario_complex_stress.json"
OUTPUT_DIR = ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "dark": "#2B2B2B",
    "source_star": "#E60000",
    "white": "#FFFFFF",
    "grid_line": "#000000",
    "head_left": "#2166AC",
    "head_right": "#B2182B",
    "noflow": "#777777",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": COLORS["dark"],
    "axes.labelpad": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.titleweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def create_geological_cmap() -> LinearSegmentedColormap:
    """Blue-white-red conductivity colormap matching the existing optimized figure."""
    cdict = {
        "red": [
            (0.00, 0.13, 0.13),
            (0.30, 0.35, 0.35),
            (0.50, 0.98, 0.98),
            (0.70, 0.85, 0.85),
            (1.00, 0.70, 0.70),
        ],
        "green": [
            (0.00, 0.15, 0.15),
            (0.30, 0.55, 0.55),
            (0.50, 0.98, 0.98),
            (0.70, 0.60, 0.60),
            (1.00, 0.20, 0.20),
        ],
        "blue": [
            (0.00, 0.67, 0.67),
            (0.30, 0.75, 0.75),
            (0.50, 0.98, 0.98),
            (0.70, 0.45, 0.45),
            (1.00, 0.13, 0.13),
        ],
    }
    return LinearSegmentedColormap("geological_cmap", cdict)


def load_case() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_true_grid(ax: plt.Axes, nx: int, ny: int) -> None:
    """Draw the actual 96 x 96 computational grid as a fine overlay."""
    for x in range(nx + 1):
        ax.axvline(x=x - 0.5, color=COLORS["grid_line"], linewidth=0.15, alpha=0.24, zorder=4)
    for y in range(ny + 1):
        ax.axhline(y=y - 0.5, color=COLORS["grid_line"], linewidth=0.15, alpha=0.24, zorder=4)


def draw_boundary_labels(ax: plt.Axes, nx: int, ny: int, h_left: float, h_right: float) -> None:
    ax.text(
        -7.0,
        ny / 2,
        "$h = {:.0f}$".format(h_left),
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        color=COLORS["head_left"],
        fontweight="bold",
    )
    ax.text(
        nx + 6.0,
        ny / 2,
        "$h = {:.0f}$".format(h_right),
        rotation=-90,
        va="center",
        ha="center",
        fontsize=10,
        color=COLORS["head_right"],
        fontweight="bold",
    )
    ax.text(
        nx / 2,
        ny + 3.2,
        "no-flow boundary",
        va="center",
        ha="center",
        fontsize=8,
        color=COLORS["noflow"],
    )
    ax.text(
        nx / 2,
        -4.2,
        "no-flow boundary",
        va="center",
        ha="center",
        fontsize=8,
        color=COLORS["noflow"],
    )


def draw_sources(ax: plt.Axes, sources: list[dict]) -> None:
    for idx, source in enumerate(sources, start=1):
        # Model indices use i=row and j=column; map coordinates use x=column and y=row.
        x = float(source["j"])
        y = float(source["i"])
        ax.scatter(
            x,
            y,
            s=175,
            color=COLORS["source_star"],
            marker="*",
            edgecolors="none",
            zorder=10,
            alpha=0.96,
        )
        ax.text(
            x + 2.6,
            y + 2.2,
            "S{}".format(idx),
            fontsize=14,
            color=COLORS["dark"],
            ha="left",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)],
            zorder=11,
        )


def add_source_legend(ax: plt.Axes) -> None:
    source_handle = Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markerfacecolor=COLORS["source_star"],
        markeredgecolor="none",
        markersize=9.5,
        label="Solute source",
    )
    ax.legend(
        handles=[source_handle],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        frameon=False,
        fontsize=8,
        borderpad=0.35,
        handletextpad=0.45,
        borderaxespad=0.2,
    )


def make_figure_3a() -> Path:
    case = load_case()
    domain = case["domain"]
    physics = case["physics"]
    k_cfg = dict(case["k_field"])
    nx = int(domain["nx"])
    ny = int(domain["ny"])

    k_field = generate_k_field(nx, ny, **k_cfg)
    log_k = np.log10(np.maximum(k_field, 1e-30))

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.4))
    im = ax.imshow(
        log_k,
        cmap=create_geological_cmap(),
        origin="lower",
        aspect="equal",
        interpolation="bicubic",
        extent=[-0.5, nx - 0.5, -0.5, ny - 0.5],
    )

    draw_true_grid(ax, nx, ny)
    draw_boundary_labels(ax, nx, ny, float(physics["h_left"]), float(physics["h_right"]))
    draw_sources(ax, list(physics.get("sources", [])))
    add_source_legend(ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-9.0, nx + 8.0)
    ax.set_ylim(-6.5, ny + 5.5)
    ax.set_aspect("equal")
    ax.set_title("Synthetic aquifer: channelized two-source stress test", pad=13)

    ax.text(
        0.02,
        0.98,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COLORS["dark"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.035)
    cbar.set_label(r"$\log_{10} K$", fontsize=12, color=COLORS["dark"], labelpad=8)
    cbar.ax.tick_params(labelsize=11, colors=COLORS["dark"], width=0.6, length=3.0)
    cbar.outline.set_visible(False)
    cbar.ax.text(
        0.5,
        -0.04,
        "low $K$",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["head_left"],
        fontweight="bold",
    )
    cbar.ax.text(
        0.5,
        1.04,
        "high $K$",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["head_right"],
        fontweight="bold",
    )

    fig.tight_layout(pad=1.6)
    output_path = OUTPUT_DIR / "Figure_03a_synthetic_current_case.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: {}".format(output_path))
    return output_path


if __name__ == "__main__":
    make_figure_3a()
