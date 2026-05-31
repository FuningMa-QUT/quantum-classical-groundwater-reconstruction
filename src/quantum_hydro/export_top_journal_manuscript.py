"""Export manuscript figures/tables aligned to the paper placeholder order."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantum_hydro.paper_budget_ablation import export_paper_budget_ablation


ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = ROOT / "outputs" / "sweeps_phaseprotocol_confirm" / "public_modflow_benchmark_suite"
SYNTH_DIR = ROOT / "outputs" / "sweeps" / "noise_constraints_heterogeneity_pilot"
BUDGET_DIR = ROOT / "outputs" / "sweeps_budget_ablation" / "public_modflow_p06_budget_ablation"

COLORS = {
    # Nature Publishing Group / Science-style, colorblind-aware palette.
    "navy": "#3C5488",
    "blue": "#4DBBD5",
    "teal": "#00A087",
    "green": "#00A087",
    "olive": "#91D1C2",
    "gold": "#F39B7F",
    "rust": "#E64B35",
    "red": "#E64B35",
    "purple": "#8491B4",
    "gray": "#6F6F6F",
    "light": "#F5F7FA",
    "panel": "#FBFCFE",
    "grid": "#E6E9EF",
    "dark": "#2B2B2B",
}

METHOD_COLORS = {
    "global": COLORS["gray"],
    "adaptive_eb_hlradial4": COLORS["teal"],
    "adaptive_eb_x4": COLORS["teal"],
    "regional_hlradial4": COLORS["gold"],
    "regional_radial4": COLORS["rust"],
    "regional_x4": COLORS["gold"],
}

METHOD_LABELS = {
    "global": "Global MCR",
    "adaptive_eb_radial4": "Adaptive EB-R4",
    "adaptive_eb_hlradial4": "Adaptive EB-HLR4",
    "adaptive_eb_x4": "Adaptive EB-X4",
    "regional_hlradial4": "Regional HLR4",
    "regional_radial4": "Regional R4",
    "regional_x4": "Regional X4",
}

PROTOCOL_COLORS = [COLORS["navy"], COLORS["teal"], COLORS["rust"]]
BENCHMARK_COLORS = {
    "MODFLOW 6 P05 Radial Flow": COLORS["blue"],
    "MODFLOW 6 P06 Injection-Extraction Well": COLORS["rust"],
    "MODFLOW 6 P09 Two-Dimensional Application": COLORS["teal"],
}
MONITOR_COLOR = COLORS["purple"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["dark"],
            "axes.labelpad": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name.replace("_", " "))


def protocol_label(name: str) -> str:
    labels = {
        "Uniform Reference": "Uniform",
        "Full Phase Protocol": "Full phase",
        "Budget-Matched Phase Protocol": "Budget matched",
    }
    return labels.get(name, name)


def polish_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["dark"])


def clean_colorbar(cbar) -> None:
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, colors=COLORS["dark"])


def journal_bar(ax: plt.Axes, x, height, *, color, width=0.68, label=None, **kwargs):
    return ax.bar(
        x,
        height,
        width=width,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
        label=label,
        **kwargs,
    )


def log_lollipop(ax: plt.Axes, labels, values, colors) -> None:
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))
    positive = values[np.isfinite(values) & (values > 0)]
    has_floor_values = np.any(np.isfinite(values) & (values <= 0))
    ymin = positive.min() * (0.10 if has_floor_values else 0.45) if len(positive) else 1e-6
    ymax = positive.max() * 1.8 if len(positive) else 1.0
    for xi, yi, color in zip(x, values, colors):
        if not np.isfinite(yi):
            continue
        marker_y = yi if yi > 0 else ymin * 1.08
        ax.vlines(xi, ymin, marker_y, color=color, linewidth=2.0, alpha=0.75)
        ax.scatter(xi, marker_y, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        if yi <= 0:
            ax.text(xi, marker_y * 1.18, "0", ha="center", va="bottom", fontsize=8, color=color)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COLORS["dark"],
    )


def write_table(df: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    def latex_escape(text: str) -> str:
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("~", "\\textasciitilde{}")
            .replace("^", "\\textasciicircum{}")
        )

    def format_plain(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            val = float(value)
            if val == 0:
                return "0"
            if abs(val) >= 1e4 or abs(val) < 1e-3:
                return f"{val:.3e}"
            return f"{val:.4f}"
        return str(value)

    plain = df.apply(lambda column: column.map(format_plain))
    plain.to_csv(csv_path, index=False)

    headers = [latex_escape(str(c)) for c in plain.columns]
    lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for _, row in plain.iterrows():
        values = [latex_escape(str(value)) for value in row]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def export_table_1(out_dir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "method": "Global MCR",
                "simple idea": "one scale for the whole field",
                "anchors": "boundary / source / well anchors",
                "local model": "none",
                "role": "baseline reconstruction",
            },
            {
                "method": "Regional radial MCR",
                "simple idea": "local scale varies by region",
                "anchors": "truth-independent local anchors",
                "local model": "radial basis",
                "role": "tests flexible regional scaling",
            },
            {
                "method": "Regional hierarchical radial MCR",
                "simple idea": "regional scale with low-rank structure",
                "anchors": "truth-independent local anchors",
                "local model": "hierarchical low-rank radial basis",
                "role": "stabilizes regional scaling",
            },
            {
                "method": "Adaptive EB-R4 / EB-HLR4",
                "simple idea": "global backbone plus local departures when holdout evidence supports them",
                "anchors": "shared noisy anchors + holdouts",
                "local model": "radial or hierarchical low-rank radial basis",
                "role": "main adaptive family",
            },
        ]
    )
    write_table(df, out_dir / "Table_01.csv", out_dir / "Table_01.tex")


def export_table_2(out_dir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "benchmark": "Synthetic moderate heterogeneity",
                "setup": "32x32 synthetic grid",
                "stress": "steady flow + transient transport",
                "protocol": "exact / moderate / high noise; 2, 4, 16, 32, 64 anchors",
                "purpose": "magnitude recovery check",
            },
            {
                "benchmark": "Synthetic strong heterogeneity",
                "setup": "128x128 synthetic grid",
                "stress": "extreme heterogeneity + sparse data",
                "protocol": "exact / moderate / high noise; 2, 4, 16, 32, 64 anchors",
                "purpose": "stress test",
            },
            {
                "benchmark": "MODFLOW 6 P05",
                "setup": "public radial-flow case",
                "stress": "baseline public benchmark",
                "protocol": "exact / moderate / high noise; 4, 16, 64 anchors",
                "purpose": "public baseline",
            },
            {
                "benchmark": "MODFLOW 6 P06",
                "setup": "public injection-extraction case",
                "stress": "injection phase + pumpback phase",
                "protocol": "phase-aware monitoring; 6848 full-phase or 4800 budget-matched observations",
                "purpose": "main public benchmark",
            },
            {
                "benchmark": "MODFLOW 6 P09",
                "setup": "public 2-D transport case",
                "stress": "cross-benchmark check",
                "protocol": "exact / moderate / high noise; 4, 16, 64 anchors",
                "purpose": "robustness check",
            },
        ]
    )
    write_table(df, out_dir / "Table_02.csv", out_dir / "Table_02.tex")


def export_table_3(out_dir: Path) -> None:
    df = pd.read_csv(SUITE_DIR / "manuscript_summary_table.csv")
    subset = df[
        (df["benchmark_label"].isin(["MODFLOW 6 P05 Radial Flow", "MODFLOW 6 P06 Injection-Extraction Well", "MODFLOW 6 P09 Two-Dimensional Application"]))
        & (df["constraint_name"] == "pathline_monitoring64")
        & (df["observation_noise_relative"] == 0.02)
        & (df["quantum_noise_name"] == "exact")
    ].copy()
    subset = subset.sort_values("rmse_mean").groupby(
        ["benchmark_label", "quantum_noise_name", "observation_noise_relative"], as_index=False
    ).first()
    subset = subset[
        [
            "benchmark_label",
            "mcr_method_name",
            "rmse_mean",
            "mass_rel_error_abs_mean",
            "peak_bias_abs_mean",
        ]
    ].rename(
        columns={
            "benchmark_label": "benchmark",
            "mcr_method_name": "best method",
            "rmse_mean": "RMSE",
            "mass_rel_error_abs_mean": "mass error",
            "peak_bias_abs_mean": "peak bias",
        }
    )
    subset["benchmark"] = subset["benchmark"].str.replace("MODFLOW 6 ", "", regex=False)
    subset["best method"] = subset["best method"].map(method_label)
    write_table(subset, out_dir / "Table_03.csv", out_dir / "Table_03.tex")


def export_table_4(out_dir: Path) -> None:
    df = pd.read_csv(SUITE_DIR / "capture_manuscript_table.csv")
    subset = df[
        (df["benchmark_name"] == "modflow6_mt3dms_p06")
        & (df["constraint_name"] == "pathline_monitoring64")
        & (df["observation_noise_relative"].isin([0.0, 0.02, 0.05]))
        & (df["quantum_noise_name"].isin(["exact", "high_noise"]))
        & (df["mcr_method_name"].isin(["global", "adaptive_eb_hlradial4"]))
    ].copy()
    subset = subset[
        [
            "quantum_noise_name",
            "observation_noise_relative",
            "mcr_method_name",
            "rmse_mean",
            "capture_rate_scaled_error_mean",
            "overall_regret_score",
        ]
    ].rename(
        columns={
            "quantum_noise_name": "state noise",
            "observation_noise_relative": "obs noise",
            "mcr_method_name": "method",
            "rmse_mean": "RMSE",
            "capture_rate_scaled_error_mean": "capture error",
            "overall_regret_score": "normalized regret",
        }
    )
    subset["state noise"] = subset["state noise"].replace({"exact": "exact", "high_noise": "noisy"})
    subset["obs noise"] = subset["obs noise"].map(lambda x: f"{float(x):.2f}")
    subset["method"] = subset["method"].map(method_label)
    write_table(subset, out_dir / "Table_04.csv", out_dir / "Table_04.tex")


def export_table_5(out_dir: Path) -> None:
    export_paper_budget_ablation(BUDGET_DIR)
    main = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_main.csv")
    dense = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_dense_exact.csv")
    budget = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_budget.csv")

    dense002 = dense[dense["observation_noise_relative"] == 0.02][
        ["protocol_label", "global_rmse", "adaptive_rmse"]
    ].copy()
    rows = []
    for _, row in main.iterrows():
        protocol = row["protocol_label"]
        dense_row = dense002[dense002["protocol_label"] == protocol].iloc[0]
        rows.append(
            {
                "protocol": protocol.replace("Uniform Reference", "Uniform").replace("Full Phase Protocol", "Full phase").replace("Budget-Matched Phase Protocol", "Budget matched"),
                "total observations": int(
                    budget[budget["protocol_label"] == protocol]["transport_total_observations"].iloc[0]
                ),
                "global regret": float(row["global_overall_regret_score_mean"]),
                "adaptive regret": float(row["adaptive_overall_regret_score_mean"]),
                "adaptive gain": float(row["adaptive_gain_vs_global"]),
                "dense RMSE (global -> adaptive)": f"{float(dense_row['global_rmse']):.3f} -> {float(dense_row['adaptive_rmse']):.3f}",
            }
        )
    df = pd.DataFrame(rows)
    write_table(df, out_dir / "Table_05.csv", out_dir / "Table_05.tex")


def figure_1(out_dir: Path) -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)
    field_a = (
        0.55
        + 0.22 * np.sin(2 * np.pi * X + 0.5)
        + 0.16 * np.cos(2 * np.pi * Y - 0.3)
        + 0.10 * np.sin(3 * np.pi * (X + 0.35 * Y))
    )
    field_b = (
        0.55
        + 0.30 * np.sin(4 * np.pi * X + 0.4)
        + 0.22 * np.cos(3 * np.pi * Y - 0.2)
        + 0.18 * np.sin(5 * np.pi * (X - 0.45 * Y))
    )
    field_a = (field_a - field_a.min()) / (field_a.max() - field_a.min())
    field_b = (field_b - field_b.min()) / (field_b.max() - field_b.min())

    for ax, field, title, label in [
        (axes[0, 0], field_a, "Synthetic moderate heterogeneity", "(a)"),
        (axes[0, 1], field_b, "Synthetic strong heterogeneity", "(b)"),
    ]:
        im = ax.imshow(field, cmap="cividis", origin="lower")
        ax.scatter([5, 25], [8, 22], marker="*", s=120, color=COLORS["rust"], edgecolors="white", linewidths=0.6)
        ax.annotate("head = 1.0", (1.0, 16), (1.5, 27), color=COLORS["navy"], fontsize=8)
        ax.annotate("head = 0.0", (30, 16), (21.5, 3), color=COLORS["navy"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        add_panel_label(ax, label)
    cbar = fig.colorbar(im, ax=axes[0, :], orientation="horizontal", shrink=0.58, pad=0.04)
    cbar.set_label("Relative conductivity")
    clean_colorbar(cbar)

    for ax, phase, label in [
        (axes[1, 0], "Injection phase", "(c)"),
        (axes[1, 1], "Pumpback phase", "(d)"),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.add_patch(patches.Rectangle((0.5, 0.5), 9.0, 5.0, facecolor=COLORS["panel"], edgecolor="#D9DEE8", linewidth=0.9))
        ax.scatter([2.5], [3.0], s=90, color=COLORS["rust"], marker="s", label="Injection well")
        ax.scatter([7.6], [3.0], s=90, color=COLORS["navy"], marker="o", label="Extraction well")
        if phase == "Injection phase":
            ax.arrow(2.8, 3.0, 3.6, 0.0, width=0.03, head_width=0.22, color=COLORS["teal"], length_includes_head=True)
            for y0 in [1.6, 3.0, 4.4]:
                ax.plot([4.0, 8.5], [y0, y0], color=MONITOR_COLOR, linewidth=1.4, linestyle="--")
            ax.text(4.2, 5.25, "source-forward monitors", color=MONITOR_COLOR, fontsize=8)
        else:
            ax.arrow(7.3, 3.0, -3.8, 0.0, width=0.03, head_width=0.22, color=COLORS["teal"], length_includes_head=True)
            theta = np.linspace(-1.1, 1.1, 100)
            for radius in [0.8, 1.3, 1.9]:
                ax.plot(7.6 - radius * np.cos(theta), 3.0 + radius * np.sin(theta), color=MONITOR_COLOR, linewidth=1.4, linestyle="--")
            ax.text(4.5, 5.25, "capture-oriented monitors", color=MONITOR_COLOR, fontsize=8)
        ax.set_title(f"Public P06 benchmark: {phase}")
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_label(ax, label)

    save_figure(fig, out_dir / "Figure_01")


def figure_2(out_dir: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(12.0, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Normalized\noperator solve", COLORS["navy"]),
        ("Shared noisy state\nand observations", COLORS["blue"]),
        ("Global MCR\nbackbone", COLORS["teal"]),
        ("Adaptive regional\ncorrection", COLORS["gold"]),
        ("Holdout-based\nmodel selection", COLORS["rust"]),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, ((text, color), x0) in enumerate(zip(steps, xs)):
        box = patches.FancyBboxPatch(
            (x0 - 0.08, 0.35),
            0.16,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=COLORS["panel"],
            edgecolor=color,
            linewidth=1.2,
            alpha=1.0,
        )
        ax.add_patch(box)
        ax.add_patch(
            patches.Rectangle(
                (x0 - 0.08, 0.35),
                0.012,
                0.28,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(x0 + 0.005, 0.49, text, ha="center", va="center", color=COLORS["dark"], fontsize=10)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.1, 0.49),
                xytext=(x0 + 0.1, 0.49),
                arrowprops=dict(arrowstyle="->", linewidth=1.2, color=COLORS["gray"]),
            )
    ax.text(0.08, 0.16, "Quantum-side object", color=COLORS["navy"], fontsize=9)
    ax.text(0.44, 0.16, "Physical rescaling layer", color=COLORS["navy"], fontsize=9)
    ax.text(0.78, 0.16, "Evidence-gated model choice", color=COLORS["navy"], fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_02")


def figure_3(out_dir: Path) -> None:
    set_style()
    capture = pd.read_csv(SUITE_DIR / "capture_manuscript_table.csv")
    dense = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_dense_exact.csv")
    main = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_main.csv")

    p06 = capture[
        (capture["benchmark_name"] == "modflow6_mt3dms_p06")
        & (capture["constraint_name"] == "pathline_monitoring64")
        & (capture["quantum_noise_name"] == "exact")
        & (capture["observation_noise_relative"].isin([0.0, 0.02, 0.05]))
        & (capture["mcr_method_name"].isin(["global", "adaptive_eb_hlradial4", "regional_hlradial4", "regional_radial4"]))
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    methods = ["global", "adaptive_eb_hlradial4", "regional_hlradial4", "regional_radial4"]
    method_labels = ["Global\nMCR", "Adaptive\nEB-HLR4", "Regional\nHLR4", "Regional\nR4"]
    exact002 = p06[p06["observation_noise_relative"] == 0.02].set_index("mcr_method_name")
    vals = [float(exact002.loc[m, "overall_regret_score"]) if m in exact002.index else np.nan for m in methods]
    log_lollipop(axes[0, 0], method_labels, vals, [METHOD_COLORS[m] for m in methods])
    axes[0, 0].set_ylabel("Overall regret")
    polish_axis(axes[0, 0], "y")
    axes[0, 0].set_title("Method-wise regret at 2% observation noise")
    add_panel_label(axes[0, 0], "(a)")

    for method, marker in [("global", "o"), ("adaptive_eb_hlradial4", "s")]:
        subset = p06[p06["mcr_method_name"] == method].sort_values("observation_noise_relative")
        axes[0, 1].plot(
            subset["observation_noise_relative"],
            subset["rmse_mean"],
            marker=marker,
            color=METHOD_COLORS[method],
            label=method_label(method),
        )
    axes[0, 1].set_xlabel("Observation noise")
    axes[0, 1].set_ylabel("Concentration RMSE")
    axes[0, 1].set_title("Injection-phase dominated RMSE trend")
    polish_axis(axes[0, 1], "both")
    axes[0, 1].legend()
    add_panel_label(axes[0, 1], "(b)")

    for method, marker in [("global", "o"), ("adaptive_eb_hlradial4", "s")]:
        subset = p06[p06["mcr_method_name"] == method].sort_values("observation_noise_relative")
        axes[1, 0].plot(
            subset["observation_noise_relative"],
            subset["capture_rate_scaled_error_mean"],
            marker=marker,
            color=METHOD_COLORS[method],
            label=method_label(method),
        )
    axes[1, 0].set_xlabel("Observation noise")
    axes[1, 0].set_ylabel("Capture error")
    axes[1, 0].set_title("Pumpback-phase capture sensitivity")
    polish_axis(axes[1, 0], "both")
    add_panel_label(axes[1, 0], "(c)")

    protocols = [protocol_label(value) for value in main["protocol_label"].tolist()]
    global_regret = main["global_overall_regret_score_mean"].to_numpy()
    adaptive_regret = main["adaptive_overall_regret_score_mean"].to_numpy()
    x = np.arange(len(protocols))
    w = 0.34
    journal_bar(axes[1, 1], x - w / 2, global_regret, width=w, color=COLORS["gray"], label="Global MCR")
    journal_bar(axes[1, 1], x + w / 2, adaptive_regret, width=w, color=COLORS["teal"], label="Adaptive EB-HLR4")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(protocols, rotation=18, ha="right")
    axes[1, 1].set_title("Phase-aware protocol comparison")
    polish_axis(axes[1, 1], "y")
    axes[1, 1].legend()
    add_panel_label(axes[1, 1], "(d)")

    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_03")


def figure_4(out_dir: Path) -> None:
    set_style()
    main = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_main.csv")
    dense = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_dense_exact.csv")
    budget = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_budget.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    x = np.arange(len(main))
    labels = [protocol_label(value) for value in main["protocol_label"].tolist()]

    journal_bar(axes[0, 0], x, budget["transport_total_observations"], color=PROTOCOL_COLORS)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0, 0].set_ylabel("Total operator observations")
    axes[0, 0].set_title("Evidence budget")
    polish_axis(axes[0, 0], "y")
    add_panel_label(axes[0, 0], "(a)")

    journal_bar(axes[0, 1], x - 0.17, main["global_overall_regret_score_mean"], width=0.34, color=COLORS["gray"], label="Global MCR")
    journal_bar(axes[0, 1], x + 0.17, main["adaptive_overall_regret_score_mean"], width=0.34, color=COLORS["teal"], label="Adaptive EB-HLR4")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=18, ha="right")
    axes[0, 1].set_title("Benchmark-normalized regret")
    polish_axis(axes[0, 1], "y")
    axes[0, 1].legend()
    add_panel_label(axes[0, 1], "(b)")

    dense002 = dense[dense["observation_noise_relative"] == 0.02]
    journal_bar(axes[1, 0], x - 0.17, dense002["global_rmse"], width=0.34, color=COLORS["gray"], label="Global MCR")
    journal_bar(axes[1, 0], x + 0.17, dense002["adaptive_rmse"], width=0.34, color=COLORS["teal"], label="Adaptive EB-HLR4")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[1, 0].set_ylabel("Dense-regime RMSE")
    axes[1, 0].set_title("Exact dense regime at 2% observation noise")
    polish_axis(axes[1, 0], "y")
    add_panel_label(axes[1, 0], "(c)")

    journal_bar(axes[1, 1], x, main["adaptive_gain_vs_global"], color=PROTOCOL_COLORS)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1, 1].set_ylabel("Adaptive gain vs global")
    axes[1, 1].set_title("Equal-budget method comparison")
    polish_axis(axes[1, 1], "y")
    add_panel_label(axes[1, 1], "(d)")

    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_04")


def figure_5(out_dir: Path) -> None:
    set_style()
    all_metrics = pd.read_csv(SYNTH_DIR / "all_metrics.csv")
    focus = all_metrics[
        (all_metrics["transport_quantum_mode"] == "operator_stepwise_hybrid")
        & (all_metrics["constraint_type"] == "source_control_plane_hybrid")
        & (all_metrics["quantum_noise_name"].isin(["exact", "moderate_noise"]))
        & (all_metrics["observation_noise_relative"].isin([0.0, 0.02]))
        & (all_metrics["mcr_method_name"].isin(["global", "adaptive_eb_x4"]))
    ].copy()
    focus["anchor_budget"] = focus["n_constraints"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for ax, field, title, label in [
        (axes[0, 0], "head", "Head RMSE", "(a)"),
        (axes[0, 1], "concentration", "Concentration RMSE", "(b)"),
    ]:
        metric_col = "head_rmse" if field == "head" else "concentration_rmse"
        subset = all_metrics[
            (all_metrics["field"] == field)
            & (all_metrics["constraint_type"] == "source_control_plane_hybrid")
            & (all_metrics["quantum_noise_name"] == "exact")
            & (all_metrics["observation_noise_relative"] == 0.0)
            & (all_metrics["mcr_method_name"].isin(["global", "adaptive_eb_x4"]))
        ].copy()
        for method, marker in [("global", "o"), ("adaptive_eb_x4", "s")]:
            tmp = subset[subset["mcr_method_name"] == method].sort_values("n_constraints")
            ax.plot(
                tmp["n_constraints"],
                tmp[metric_col],
                marker=marker,
                color=METHOD_COLORS[method],
                label=method_label(method),
            )
        ax.set_yscale("log")
        ax.set_xlabel("Anchor budget")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        polish_axis(ax, "both")
        add_panel_label(ax, label)
    axes[0, 0].legend()

    subset = all_metrics[
        (all_metrics["field"] == "concentration")
        & (all_metrics["constraint_type"] == "source_control_plane_hybrid")
        & (all_metrics["quantum_noise_name"] == "exact")
        & (all_metrics["observation_noise_relative"].isin([0.0, 0.01, 0.02, 0.05]))
        & (all_metrics["mcr_method_name"] == "global")
    ].copy()
    for budget, color, marker in [(2, COLORS["navy"], "o"), (16, COLORS["teal"], "s"), (64, COLORS["rust"], "^")]:
        tmp = subset[subset["n_constraints"] == budget].sort_values("observation_noise_relative")
        axes[1, 0].plot(
            tmp["observation_noise_relative"],
            tmp["concentration_mass_relative_error"],
            marker=marker,
            color=color,
            label=f"{budget} anchors",
        )
    axes[1, 0].set_xlabel("Observation noise")
    axes[1, 0].set_ylabel("Mass error")
    axes[1, 0].set_title("Mass recovery under sparse constraints")
    polish_axis(axes[1, 0], "both")
    axes[1, 0].legend()
    add_panel_label(axes[1, 0], "(c)")

    subset = all_metrics[
        (all_metrics["field"] == "concentration")
        & (all_metrics["constraint_type"] == "source_control_plane_hybrid")
        & (all_metrics["observation_noise_relative"] == 0.02)
        & (all_metrics["mcr_method_name"].isin(["global", "adaptive_eb_x4", "regional_x4"]))
    ].copy()
    methods = ["global", "adaptive_eb_x4", "regional_x4"]
    labels = ["Global", "Adaptive EB-X4", "Regional X4"]
    vals = [
        subset[subset["mcr_method_name"] == method]["concentration_peak_bias_percent"].median()
        for method in methods
    ]
    journal_bar(axes[1, 1], labels, vals, color=[METHOD_COLORS[m] for m in methods])
    axes[1, 1].set_ylabel("Median peak bias")
    axes[1, 1].set_title("Naive regional flexibility is less stable")
    axes[1, 1].tick_params(axis="x", rotation=18)
    polish_axis(axes[1, 1], "y")
    add_panel_label(axes[1, 1], "(d)")

    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_05")


def figure_6(out_dir: Path) -> None:
    set_style()
    summary = pd.read_csv(SUITE_DIR / "benchmark_normalized_summary.csv")
    winners = pd.read_csv(SUITE_DIR / "manuscript_summary_table.csv")

    pivot = summary[
        summary["benchmark_name"].isin(["modflow6_mt3dms_p05", "modflow6_mt3dms_p06", "modflow6_mt3dms_p09"])
    ].pivot_table(
        index="mcr_method_name",
        columns="benchmark_label",
        values="overall_regret_score_mean",
        aggfunc="mean",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    im = axes[0, 0].imshow(np.log10(pivot.to_numpy() + 1e-12), cmap="magma_r", aspect="auto")
    axes[0, 0].set_xticks(range(len(pivot.columns)))
    axes[0, 0].set_xticklabels(pivot.columns, rotation=20, ha="right")
    axes[0, 0].set_yticks(range(len(pivot.index)))
    axes[0, 0].set_yticklabels([method_label(idx) for idx in pivot.index])
    axes[0, 0].set_title("Mean normalized regret heatmap")
    add_panel_label(axes[0, 0], "(a)")
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar.set_label("log10(regret)")
    clean_colorbar(cbar)

    for benchmark, marker in [
        ("MODFLOW 6 P05 Radial Flow", "o"),
        ("MODFLOW 6 P06 Injection-Extraction Well", "s"),
        ("MODFLOW 6 P09 Two-Dimensional Application", "^"),
    ]:
        tmp = winners[
            (winners["benchmark_label"] == benchmark)
            & (winners["constraint_name"] == "pathline_monitoring64")
            & (winners["quantum_noise_name"] == "exact")
            & (winners["mcr_method_name"] == "global")
        ].sort_values("observation_noise_relative")
        axes[0, 1].plot(
            tmp["observation_noise_relative"],
            tmp["rmse_mean"],
            marker=marker,
            color=BENCHMARK_COLORS[benchmark],
            label=benchmark.replace("MODFLOW 6 ", ""),
        )
    axes[0, 1].set_xlabel("Observation noise")
    axes[0, 1].set_ylabel("Global MCR RMSE")
    axes[0, 1].set_title("Exact-state public benchmark sensitivity")
    polish_axis(axes[0, 1], "both")
    axes[0, 1].legend()
    add_panel_label(axes[0, 1], "(b)")

    subset = winners[
        (winners["constraint_name"] == "pathline_monitoring64")
        & (winners["observation_noise_relative"] == 0.02)
        & (winners["quantum_noise_name"].isin(["exact", "high_noise"]))
        & (winners["mcr_method_name"] == "global")
    ]
    labels = ["P05", "P06", "P09"]
    benchmark_order = [
        "MODFLOW 6 P05 Radial Flow",
        "MODFLOW 6 P06 Injection-Extraction Well",
        "MODFLOW 6 P09 Two-Dimensional Application",
    ]
    x = np.arange(len(benchmark_order))
    exact = np.array(
        [
            subset[
                (subset["quantum_noise_name"] == "exact")
                & (subset["benchmark_label"] == benchmark)
            ]["mass_rel_error_abs_mean"].min()
            for benchmark in benchmark_order
        ]
    )
    noisy = np.array(
        [
            subset[
                (subset["quantum_noise_name"] == "high_noise")
                & (subset["benchmark_label"] == benchmark)
            ]["mass_rel_error_abs_mean"].min()
            for benchmark in benchmark_order
        ]
    )
    journal_bar(axes[1, 0], x - 0.17, exact, width=0.34, color=COLORS["navy"], label="Exact")
    journal_bar(axes[1, 0], x + 0.17, noisy, width=0.34, color=COLORS["rust"], label="High noise")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel("Mass error")
    axes[1, 0].set_title("Mass-error comparison")
    polish_axis(axes[1, 0], "y")
    axes[1, 0].legend()
    add_panel_label(axes[1, 0], "(c)")

    unstable = summary[
        (summary["benchmark_name"] == "modflow6_mt3dms_p06")
        & (summary["mcr_method_name"].isin(["regional_hlradial4", "regional_radial4", "adaptive_eb_hlradial4", "global"]))
    ][["mcr_method_name", "overall_regret_score_mean"]].copy()
    unstable = unstable.groupby("mcr_method_name", as_index=False).mean(numeric_only=True)
    order = ["global", "adaptive_eb_hlradial4", "regional_hlradial4", "regional_radial4"]
    unstable["order"] = unstable["mcr_method_name"].map({name: i for i, name in enumerate(order)})
    unstable = unstable.sort_values("order")
    log_lollipop(
        axes[1, 1],
        [method_label(m) for m in unstable["mcr_method_name"]],
        unstable["overall_regret_score_mean"],
        [METHOD_COLORS.get(m, COLORS["purple"]) for m in unstable["mcr_method_name"]],
    )
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].set_title("Unconstrained regional methods are less stable")
    polish_axis(axes[1, 1], "y")
    add_panel_label(axes[1, 1], "(d)")

    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_06")


def figure_7(out_dir: Path) -> None:
    set_style()
    capture = pd.read_csv(SUITE_DIR / "capture_manuscript_table.csv")
    main = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_main.csv")
    dense = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_dense_exact.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for ax, phase, label in [
        (axes[0, 0], "Injection", "(a)"),
        (axes[0, 1], "Pumpback", "(b)"),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.add_patch(patches.Rectangle((0.6, 0.6), 8.8, 4.8, facecolor=COLORS["panel"], edgecolor="#D9DEE8", linewidth=0.9))
        ax.scatter([2.3], [3.0], s=90, color=COLORS["rust"], marker="s")
        ax.scatter([7.7], [3.0], s=90, color=COLORS["navy"], marker="o")
        if phase == "Injection":
            for yy in [1.7, 3.0, 4.3]:
                ax.plot([3.6, 8.6], [yy, yy], linestyle="--", linewidth=1.4, color=MONITOR_COLOR)
            ax.arrow(2.6, 3.0, 3.9, 0.0, width=0.03, head_width=0.22, color=COLORS["teal"], length_includes_head=True)
            ax.text(4.0, 5.05, "source-forward monitors", fontsize=8, color=MONITOR_COLOR)
        else:
            theta = np.linspace(-1.1, 1.1, 100)
            for rr in [0.9, 1.45, 2.1]:
                ax.plot(7.7 - rr * np.cos(theta), 3.0 + rr * np.sin(theta), linestyle="--", linewidth=1.4, color=MONITOR_COLOR)
            ax.arrow(7.4, 3.0, -3.8, 0.0, width=0.03, head_width=0.22, color=COLORS["teal"], length_includes_head=True)
            ax.text(4.2, 5.05, "capture-oriented monitors", fontsize=8, color=MONITOR_COLOR)
        ax.set_title(f"{phase}-phase layout")
        add_panel_label(ax, label)

    x = np.arange(len(main))
    labels = [protocol_label(value) for value in main["protocol_label"].tolist()]
    journal_bar(axes[1, 0], x - 0.17, main["global_overall_regret_score_mean"], width=0.34, color=COLORS["gray"], label="Global MCR")
    journal_bar(axes[1, 0], x + 0.17, main["adaptive_overall_regret_score_mean"], width=0.34, color=COLORS["teal"], label="Adaptive EB-HLR4")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[1, 0].set_title("Protocol-wise regret")
    polish_axis(axes[1, 0], "y")
    axes[1, 0].legend()
    add_panel_label(axes[1, 0], "(c)")

    dense002 = dense[dense["observation_noise_relative"] == 0.02]
    axes[1, 1].plot(x, dense002["global_rmse"], marker="o", color=COLORS["gray"], label="Global MCR")
    axes[1, 1].plot(x, dense002["adaptive_rmse"], marker="s", color=COLORS["teal"], label="Adaptive EB-HLR4")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1, 1].set_ylabel("Dense-regime RMSE")
    axes[1, 1].set_title("Pumpback-sensitive dense regime")
    polish_axis(axes[1, 1], "both")
    axes[1, 1].legend()
    add_panel_label(axes[1, 1], "(d)")

    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_07")


def figure_8(out_dir: Path) -> None:
    set_style()
    main = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_main.csv")
    dense = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_dense_exact.csv")
    budget = pd.read_csv(BUDGET_DIR / "paper_tables" / "table_p06_budget_ablation_budget.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    labels = [protocol_label(value) for value in main["protocol_label"].tolist()]
    x = np.arange(len(labels))

    journal_bar(axes[0], x, budget["transport_total_observations"], color=PROTOCOL_COLORS)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_title("Total observation budget")
    polish_axis(axes[0], "y")
    add_panel_label(axes[0], "(a)")

    journal_bar(axes[1], x, main["adaptive_overall_regret_score_mean"], color=PROTOCOL_COLORS)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_title("Adaptive mean overall regret")
    polish_axis(axes[1], "y")
    add_panel_label(axes[1], "(b)")

    dense002 = dense[dense["observation_noise_relative"] == 0.02]
    gain = dense002["global_rmse"].to_numpy() - dense002["adaptive_rmse"].to_numpy()
    journal_bar(axes[2], x, gain, color=PROTOCOL_COLORS)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")
    axes[2].set_title("Adaptive gain in dense RMSE")
    polish_axis(axes[2], "y")
    add_panel_label(axes[2], "(c)")

    fig.tight_layout()
    save_figure(fig, out_dir / "Figure_08")


def build_manifest(out_dir: Path) -> None:
    text = """# Top-Journal Manuscript Export

This bundle is aligned to the manuscript placeholder order.

Figures:
- Figure 1. Synthetic and public benchmark configurations.
- Figure 2. Quantum-state emulation and shared-noise protocol.
- Figure 3. P06 phase-aware reconstruction performance.
- Figure 4. Budget-matched ablation for P06.
- Figure 5. Synthetic validation of magnitude recovery from normalized quantum-state outputs.
- Figure 6. Public benchmark reconstruction under exact and noisy quantum-state emulation.
- Figure 7. Phase-aware reconstruction in the P06 injection-extraction benchmark.
- Figure 8. Budget-matched comparison of uniform and phase-aware monitoring.

Tables:
- Table 1. Reconstruction methods and monitoring protocols.
- Table 2. Benchmark cases, noise settings, and observation-budget protocols.
- Table 3. Main performance table for public benchmark reconstruction.
- Table 4. Public benchmark performance under shared noisy states and observations.
- Table 5. Budget-matched ablation for P06.

Note:
- The earlier draft reused `Table 3` in two sections. This export resolves that conflict by using a single continuous numbering scheme.
- The synthetic validation figure uses the formal synthetic sweep diagnostics available in the workspace, rather than unpublished field-map panels.
- Figure styling uses a restrained Nature/Science-like palette, Times New Roman-compatible fonts, white backgrounds, light gridlines, and publication-ready PDF/PNG export.
- Tables are compact, reader-facing summaries; internal sweep fields are intentionally omitted from the main tables.
"""
    (out_dir / "MANIFEST.md").write_text(text, encoding="utf-8")


def build_publication_manifest(out_dir: Path) -> None:
    text = """# Publication-Ready Manuscript Export

Recommended main-manuscript set:

Main figures:
- Figure 1. Synthetic and public benchmark configurations.
- Figure 2. Quantum-state emulation and shared-noise protocol.
- Figure 3. Synthetic validation of magnitude recovery from normalized quantum-state outputs.
- Figure 4. Public benchmark reconstruction under exact and noisy quantum-state emulation.
- Figure 5. P06 phase-aware reconstruction performance.
- Figure 6. Budget-matched ablation for P06.

Main tables:
- Table 1. Reconstruction methods and monitoring protocols.
- Table 2. Main public-benchmark performance at 2% observation noise and exact quantum-state emulation.
- Table 3. Budget-matched ablation for P06.

Recommended supplementary set:
- Figure S1. Detailed phase-aware layout and P06 dense-regime diagnostics.
- Figure S2. Budget-matched comparison of uniform and phase-aware monitoring.
- Table S1. Benchmark cases, noise settings, and observation-budget protocols.
- Table S2. P06 performance under shared noisy states and observations.

Rationale:
- Figures 7 and 8 from the full export overlap with Figures 3 and 4 in the main evidence chain, so they are moved to the supplement.
- Tables 2 and 4 from the full export are useful for reproducibility but too detailed for the main text.
- Extreme state-noise cases are retained in the figure and supplement rather than placed in the main table, where they would obscure the central exact-state benchmark comparison.
- The source_full_set directory preserves the complete 8-figure/5-table export for traceability.
"""
    (out_dir / "MANIFEST.md").write_text(text, encoding="utf-8")


def copy_artifacts(src_dir: Path, dst_dir: Path, src_stem: str, dst_stem: str, suffixes: tuple[str, ...]) -> None:
    ensure_dir(dst_dir)
    for suffix in suffixes:
        src = src_dir / f"{src_stem}{suffix}"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{dst_stem}{suffix}")


def export_bundle(output_dir: Path) -> None:
    export_paper_budget_ablation(BUDGET_DIR)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    export_table_1(tables_dir)
    export_table_2(tables_dir)
    export_table_3(tables_dir)
    export_table_4(tables_dir)
    export_table_5(tables_dir)

    figure_1(figures_dir)
    figure_2(figures_dir)
    figure_3(figures_dir)
    figure_4(figures_dir)
    figure_5(figures_dir)
    figure_6(figures_dir)
    figure_7(figures_dir)
    figure_8(figures_dir)

    build_manifest(output_dir)


def export_publication_bundle(output_dir: Path) -> None:
    source_dir = ensure_dir(output_dir / "source_full_set")
    export_bundle(source_dir)

    main_figures = ensure_dir(output_dir / "main_figures")
    main_tables = ensure_dir(output_dir / "main_tables")
    supp_figures = ensure_dir(output_dir / "supplementary_figures")
    supp_tables = ensure_dir(output_dir / "supplementary_tables")

    source_figures = source_dir / "figures"
    source_tables = source_dir / "tables"

    for src, dst in [
        ("Figure_01", "Figure_01"),
        ("Figure_02", "Figure_02"),
        ("Figure_05", "Figure_03"),
        ("Figure_06", "Figure_04"),
        ("Figure_03", "Figure_05"),
        ("Figure_04", "Figure_06"),
    ]:
        copy_artifacts(source_figures, main_figures, src, dst, (".png", ".pdf"))

    for src, dst in [
        ("Figure_07", "Figure_S01"),
        ("Figure_08", "Figure_S02"),
    ]:
        copy_artifacts(source_figures, supp_figures, src, dst, (".png", ".pdf"))

    for src, dst in [
        ("Table_01", "Table_01"),
        ("Table_03", "Table_02"),
        ("Table_05", "Table_03"),
    ]:
        copy_artifacts(source_tables, main_tables, src, dst, (".csv", ".tex"))

    for src, dst in [
        ("Table_02", "Table_S01"),
        ("Table_04", "Table_S02"),
    ]:
        copy_artifacts(source_tables, supp_tables, src, dst, (".csv", ".tex"))

    build_publication_manifest(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export publication-ready figures and tables.")
    parser.add_argument(
        "--output-dir",
        default="outputs/manuscript_export/top_journal_publication_set",
        help="Output directory for the publication-ready manuscript bundle.",
    )
    parser.add_argument(
        "--full-set",
        action="store_true",
        help="Export the legacy full 8-figure/5-table set instead of the recommended main/supplement split.",
    )
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    if args.full_set:
        export_bundle(output_dir)
    else:
        export_publication_bundle(output_dir)


if __name__ == "__main__":
    main()
