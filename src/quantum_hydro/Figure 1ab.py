"""Optimized Figure 1a and 1b: Synthetic heterogeneous aquifer configurations."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Path setup
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    "dark": "#2B2B2B",
    "source_star": "#E60000",
    "white": "#FFFFFF",
    "grid_line": "#000000",
    "cmap_low": "#2166AC",
    "cmap_high": "#B2182B",
}

# Style settings - all text in Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
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
    "legend.title_fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def create_geological_cmap():
    """
    Create a classic geological colormap:
    Blue (low K, clay/shale) -> White (medium K, silt) -> Brown/Red (high K, sand/gravel).
    """
    cdict = {
        'red': [
            (0.00, 0.13, 0.13),
            (0.30, 0.35, 0.35),
            (0.50, 0.98, 0.98),
            (0.70, 0.85, 0.85),
            (1.00, 0.70, 0.70),
        ],
        'green': [
            (0.00, 0.15, 0.15),
            (0.30, 0.55, 0.55),
            (0.50, 0.98, 0.98),
            (0.70, 0.60, 0.60),
            (1.00, 0.20, 0.20),
        ],
        'blue': [
            (0.00, 0.67, 0.67),
            (0.30, 0.75, 0.75),
            (0.50, 0.98, 0.98),
            (0.70, 0.45, 0.45),
            (1.00, 0.13, 0.13),
        ],
    }
    return LinearSegmentedColormap('geological_cmap', cdict)


def generate_k_field(grid_size: int, heterogeneity: str = "moderate"):
    """Generate a spatially correlated log-normal hydraulic conductivity field.

    Parameters
    ----------
    grid_size : int
        Grid resolution.
    heterogeneity : str
        "moderate" or "strong".

    Returns
    -------
    field : np.ndarray
        Normalized conductivity field in [0, 1].
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    if heterogeneity == "moderate":
        field = (
            0.55
            + 0.22 * np.sin(2 * np.pi * X + 0.5)
            + 0.16 * np.cos(2 * np.pi * Y - 0.3)
            + 0.10 * np.sin(3 * np.pi * (X + 0.35 * Y))
        )
    else:  # strong
        field = (
            0.55
            + 0.30 * np.sin(4 * np.pi * X + 0.4)
            + 0.22 * np.cos(3 * np.pi * Y - 0.2)
            + 0.18 * np.sin(5 * np.pi * (X - 0.45 * Y))
        )

    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min())
    return field


def draw_grid_lines(ax, grid_size):
    """Draw thin black grid lines for each cell in the computational mesh."""
    for i in range(grid_size + 1):
        ax.axvline(x=i - 0.5, color=COLORS["grid_line"], linewidth=0.3, alpha=0.4, zorder=5)
        ax.axhline(y=i - 0.5, color=COLORS["grid_line"], linewidth=0.3, alpha=0.4, zorder=5)


def draw_boundary_labels(ax, grid_size, head_left=1.0, head_right=0.0):
    """Draw head labels at the left and right boundaries."""
    n = grid_size

    ax.text(
        -2.8, n / 2,
        "$h = {}$".format(head_left),
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        color=COLORS["cmap_low"],
        fontweight="bold",
    )

    ax.text(
        n + 2.8, n / 2,
        "$h = {}$".format(head_right),
        rotation=-90,
        va="center",
        ha="center",
        fontsize=9,
        color=COLORS["cmap_high"],
        fontweight="bold",
    )


def draw_source_markers(ax, grid_size, source_positions):
    """Draw injection source locations with pure red star markers, no white edge."""
    for px, py in source_positions:
        ax.scatter(
            px, py,
            s=250,
            color=COLORS["source_star"],
            marker="*",
            edgecolors="none",
            zorder=10,
            alpha=0.95,
        )


def add_colorbar_for_axes(fig, ax, im, label="low $K$", high_label="high $K$"):
    """Add a single vertical colorbar next to a given axes."""
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("$K / K_0$", fontsize=9, color=COLORS["dark"], labelpad=6)
    cbar.ax.tick_params(labelsize=8, colors=COLORS["dark"])
    cbar.outline.set_visible(False)
    
    # Add low/high annotations
    cbar.ax.text(0.5, -0.04, label, transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=7, color=COLORS["cmap_low"],
                 fontweight="bold")
    cbar.ax.text(0.5, 1.04, high_label, transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=7, color=COLORS["cmap_high"],
                 fontweight="bold")
    
    return cbar


def plot_synthetic_aquifers():
    """Plot optimized synthetic aquifer configuration figures (Figure 1a & 1b)."""

    custom_cmap = create_geological_cmap()

    # Create 1 row x 2 columns subplot, adjust width to accommodate individual colorbars
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))

    configs = [
        {
            "ax": axes[0],
            "heterogeneity": "moderate",
            "title": "Synthetic aquifer: moderate heterogeneity",
            "panel": "(a)",
            "grid_size": 32,
            "sources": [(5, 8), (25, 22)],
        },
        {
            "ax": axes[1],
            "heterogeneity": "strong",
            "title": "Synthetic aquifer: strong heterogeneity",
            "panel": "(b)",
            "grid_size": 32,
            "sources": [(5, 8), (25, 22)],
        },
    ]

    for cfg in configs:
        ax = cfg["ax"]
        grid_size = cfg["grid_size"]

        # Generate conductivity field
        k_field = generate_k_field(grid_size, cfg["heterogeneity"])

        # Plot conductivity field
        im = ax.imshow(
            k_field,
            cmap=custom_cmap,
            origin="lower",
            aspect="equal",
            interpolation="bicubic",
            extent=[-0.5, grid_size - 0.5, -0.5, grid_size - 0.5],
        )

        # Draw black grid lines overlay
        draw_grid_lines(ax, grid_size)

        # Draw boundary labels
        draw_boundary_labels(ax, grid_size)

        # Draw source markers (pure red, no white edge)
        draw_source_markers(ax, grid_size, cfg["sources"])

        # Add individual colorbar for this subplot
        add_colorbar_for_axes(fig, ax, im)

        # Axis cleanup
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-3.5, grid_size + 1.5)
        ax.set_ylim(-2, grid_size + 0.5)
        ax.set_aspect("equal")

        # Title in Times New Roman
        ax.set_title(cfg["title"], fontsize=11, color=COLORS["dark"], pad=12)

        # Panel label
        ax.text(
            0.02, 0.98,
            cfg["panel"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=COLORS["dark"],
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
                pad=2,
            ),
        )

    fig.tight_layout(pad=2.5, w_pad=6.0)

    # Save
    for fmt in ["png", "pdf"]:
        path = OUTPUT_DIR / "Figure_01_ab_optimized.{}".format(fmt)
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        print("  Saved: {}".format(path))

    plt.close(fig)


if __name__ == "__main__":
    plot_synthetic_aquifers()
    print("Done: Optimized Figure 1a & 1b exported.")