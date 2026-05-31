"""Publication-ready tables and figures for the P06 budget-matched ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTOCOL_ORDER = [
    "modflow6_mt3dms_p06_uniform_reference",
    "modflow6_mt3dms_p06",
    "modflow6_mt3dms_p06_phaseprotocol_budgetmatched",
]

PROTOCOL_LABELS = {
    "modflow6_mt3dms_p06_uniform_reference": "Uniform Reference",
    "modflow6_mt3dms_p06": "Full Phase Protocol",
    "modflow6_mt3dms_p06_phaseprotocol_budgetmatched": "Budget-Matched Phase Protocol",
}

METHOD_ORDER = ["global", "adaptive_eb_hlradial4"]
METHOD_LABELS = {
    "global": "Global",
    "adaptive_eb_hlradial4": "Adaptive EB-HLRadial4",
}

PROTOCOL_COLORS = {
    "modflow6_mt3dms_p06_uniform_reference": "#4C78A8",
    "modflow6_mt3dms_p06": "#F58518",
    "modflow6_mt3dms_p06_phaseprotocol_budgetmatched": "#54A24B",
}

METHOD_MARKERS = {
    "global": "o",
    "adaptive_eb_hlradial4": "s",
}


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _protocol_sort_key(name: str) -> int:
    try:
        return PROTOCOL_ORDER.index(str(name))
    except ValueError:
        return len(PROTOCOL_ORDER)


def _method_sort_key(name: str) -> int:
    try:
        return METHOD_ORDER.index(str(name))
    except ValueError:
        return len(METHOD_ORDER)


def _format_protocol_table_value(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _write_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    *,
    float_columns: list[str] | None = None,
    percent_columns: list[str] | None = None,
) -> None:
    if df.empty:
        output_path.write_text("", encoding="utf-8")
        return

    float_columns = float_columns or []
    percent_columns = percent_columns or []
    formatted = df.copy()
    for column in float_columns:
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{100.0 * float(value):.1f}"
            )
    columns = [str(column) for column in formatted.columns]
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for _, row in formatted.iterrows():
        values = []
        for column in formatted.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            else:
                text = str(value)
                text = text.replace("_", "\\_")
                values.append(text)
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    latex = "\n".join(lines)
    output_path.write_text(latex, encoding="utf-8")


def build_protocol_main_table(
    normalized_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    keep = normalized_summary[
        normalized_summary["benchmark_name"].isin(PROTOCOL_ORDER)
        & normalized_summary["mcr_method_name"].isin(METHOD_ORDER)
    ].copy()
    if keep.empty:
        output_path.write_text("", encoding="utf-8")
        return keep

    numeric_cols = [
        "overall_regret_score_mean",
        "rmse_mean_relative_regret_mean",
        "mass_rel_error_abs_mean_relative_regret_mean",
        "peak_bias_abs_mean_relative_regret_mean",
        "near_best_rmse_10_share",
        "near_best_capture_10_share",
    ]
    keep = _to_numeric(keep, numeric_cols)
    keep["protocol_label"] = keep["benchmark_name"].map(PROTOCOL_LABELS).fillna(keep["benchmark_name"])
    keep = keep.sort_values(
        by=["benchmark_name", "mcr_method_name"],
        key=lambda col: col.map(_protocol_sort_key) if col.name == "benchmark_name" else col.map(_method_sort_key),
    )

    rows: list[dict[str, Any]] = []
    for protocol_name in PROTOCOL_ORDER:
        subset = keep[keep["benchmark_name"] == protocol_name].copy()
        if subset.empty:
            continue
        row: dict[str, Any] = {
            "protocol_name": protocol_name,
            "protocol_label": PROTOCOL_LABELS.get(protocol_name, protocol_name),
        }
        for _, item in subset.iterrows():
            method_name = str(item["mcr_method_name"])
            prefix = "global" if method_name == "global" else "adaptive"
            row[f"{prefix}_overall_regret_score_mean"] = item.get("overall_regret_score_mean", np.nan)
            row[f"{prefix}_rmse_relative_regret_mean"] = item.get("rmse_mean_relative_regret_mean", np.nan)
            row[f"{prefix}_mass_relative_regret_mean"] = item.get(
                "mass_rel_error_abs_mean_relative_regret_mean",
                np.nan,
            )
            row[f"{prefix}_peak_relative_regret_mean"] = item.get(
                "peak_bias_abs_mean_relative_regret_mean",
                np.nan,
            )
            row[f"{prefix}_near_best_rmse10_share"] = item.get("near_best_rmse_10_share", np.nan)
            row[f"{prefix}_near_best_capture10_share"] = item.get("near_best_capture_10_share", np.nan)
        if (
            pd.notna(row.get("global_overall_regret_score_mean"))
            and pd.notna(row.get("adaptive_overall_regret_score_mean"))
        ):
            row["adaptive_gain_vs_global"] = (
                float(row["global_overall_regret_score_mean"])
                - float(row["adaptive_overall_regret_score_mean"])
            )
        else:
            row["adaptive_gain_vs_global"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def build_dense_exact_table(
    all_metrics: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    data = all_metrics[
        (all_metrics["field"] == "concentration")
        & (all_metrics["benchmark_name"].isin(PROTOCOL_ORDER))
        & (all_metrics["constraint_name"] == "pathline_monitoring64")
        & (all_metrics["quantum_solver"] == "exact")
        & (all_metrics["mcr_method_name"].isin(METHOD_ORDER))
    ].copy()
    if data.empty:
        output_path.write_text("", encoding="utf-8")
        return data

    numeric_cols = [
        "observation_noise_relative",
        "concentration_rmse",
        "transport_total_observations",
        "transport_budget_reference_total_observations",
    ]
    data = _to_numeric(data, numeric_cols)
    data["protocol_label"] = data["benchmark_name"].map(PROTOCOL_LABELS).fillna(data["benchmark_name"])

    rows: list[dict[str, Any]] = []
    for protocol_name in PROTOCOL_ORDER:
        protocol_subset = data[data["benchmark_name"] == protocol_name]
        if protocol_subset.empty:
            continue
        for obs_noise in sorted(protocol_subset["observation_noise_relative"].dropna().unique()):
            obs_subset = protocol_subset[protocol_subset["observation_noise_relative"] == obs_noise]
            row: dict[str, Any] = {
                "protocol_name": protocol_name,
                "protocol_label": PROTOCOL_LABELS.get(protocol_name, protocol_name),
                "observation_noise_relative": float(obs_noise),
                "transport_total_observations": obs_subset["transport_total_observations"].dropna().iloc[0]
                if obs_subset["transport_total_observations"].notna().any()
                else np.nan,
                "transport_budget_reference_total_observations": obs_subset[
                    "transport_budget_reference_total_observations"
                ].dropna().iloc[0]
                if obs_subset["transport_budget_reference_total_observations"].notna().any()
                else np.nan,
            }
            for _, item in obs_subset.iterrows():
                method_name = str(item["mcr_method_name"])
                prefix = "global" if method_name == "global" else "adaptive"
                row[f"{prefix}_rmse"] = item.get("concentration_rmse", np.nan)
            if pd.notna(row.get("global_rmse")) and pd.notna(row.get("adaptive_rmse")):
                row["adaptive_minus_global_rmse"] = float(row["adaptive_rmse"]) - float(row["global_rmse"])
                row["adaptive_over_global_rmse_ratio"] = float(row["adaptive_rmse"]) / max(
                    float(row["global_rmse"]),
                    1e-30,
                )
            else:
                row["adaptive_minus_global_rmse"] = np.nan
                row["adaptive_over_global_rmse_ratio"] = np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["protocol_name", "observation_noise_relative"], key=lambda col: col.map(_protocol_sort_key) if col.name == "protocol_name" else col)
    out.to_csv(output_path, index=False)
    return out


def build_budget_table(
    dense_exact_table: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if dense_exact_table.empty:
        output_path.write_text("", encoding="utf-8")
        return dense_exact_table

    focus = dense_exact_table[dense_exact_table["observation_noise_relative"] == 0.02].copy()
    if focus.empty:
        focus = dense_exact_table.drop_duplicates("protocol_name").copy()
    focus["protocol_label"] = focus["protocol_name"].map(PROTOCOL_LABELS).fillna(focus["protocol_name"])
    uniform_total = float(
        focus.loc[focus["protocol_name"] == "modflow6_mt3dms_p06_uniform_reference", "transport_total_observations"].iloc[0]
    )
    focus["observation_budget_ratio_vs_uniform"] = focus["transport_total_observations"] / max(uniform_total, 1e-30)
    out = focus.loc[
        :,
        [
            "protocol_name",
            "protocol_label",
            "transport_total_observations",
            "transport_budget_reference_total_observations",
            "observation_budget_ratio_vs_uniform",
        ],
    ].sort_values("protocol_name", key=lambda col: col.map(_protocol_sort_key))
    out.to_csv(output_path, index=False)
    return out


def _plot_overall_regret(protocol_main_table: pd.DataFrame, output_base: Path) -> None:
    if protocol_main_table.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(protocol_main_table), dtype=float)
    width = 0.34

    adaptive = protocol_main_table["adaptive_overall_regret_score_mean"].to_numpy(dtype=float)
    global_values = protocol_main_table["global_overall_regret_score_mean"].to_numpy(dtype=float)

    ax.bar(
        x - width / 2.0,
        global_values,
        width=width,
        color="#9D9D9D",
        label="Global",
    )
    ax.bar(
        x + width / 2.0,
        adaptive,
        width=width,
        color="#2E8B57",
        label="Adaptive EB-HLRadial4",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean overall regret score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(protocol_main_table["protocol_label"], fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("P06 protocol ablation in benchmark-normalized space", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_dense_rmse(dense_exact_table: pd.DataFrame, output_base: Path) -> None:
    if dense_exact_table.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    method_specs = [
        ("adaptive_rmse", "Adaptive EB-HLRadial4"),
        ("global_rmse", "Global"),
    ]

    for ax, (column, title) in zip(axes, method_specs):
        for protocol_name in PROTOCOL_ORDER:
            subset = dense_exact_table[dense_exact_table["protocol_name"] == protocol_name].sort_values(
                "observation_noise_relative"
            )
            if subset.empty or column not in subset.columns:
                continue
            ax.plot(
                subset["observation_noise_relative"].to_numpy(dtype=float),
                subset[column].to_numpy(dtype=float),
                marker=METHOD_MARKERS["adaptive_eb_hlradial4" if column == "adaptive_rmse" else "global"],
                linewidth=2.0,
                markersize=5.5,
                color=PROTOCOL_COLORS[protocol_name],
                label=PROTOCOL_LABELS[protocol_name],
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Observation noise (relative std)", fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel("Concentration RMSE", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Exact dense-regime (pathline_monitoring64) RMSE", y=1.09, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_budget_totals(budget_table: pd.DataFrame, output_base: Path) -> None:
    if budget_table.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(budget_table), dtype=float)
    totals = budget_table["transport_total_observations"].to_numpy(dtype=float)
    colors = [PROTOCOL_COLORS.get(name, "#4C78A8") for name in budget_table["protocol_name"]]
    bars = ax.bar(x, totals, color=colors, width=0.62)

    for idx, bar in enumerate(bars):
        value = totals[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{int(round(value))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    reference_values = budget_table["transport_budget_reference_total_observations"].dropna().unique()
    if reference_values.size > 0:
        reference = float(reference_values[0])
        ax.axhline(reference, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.text(
            len(budget_table) - 0.45,
            reference,
            f"budget-matched reference = {int(round(reference))}",
            ha="right",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Total operator observations", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(budget_table["protocol_label"], fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_title("Dense-regime evidence budget (pathline_monitoring64, exact, obs=0.02)", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_compound_overview(
    protocol_main_table: pd.DataFrame,
    dense_exact_table: pd.DataFrame,
    budget_table: pd.DataFrame,
    output_base: Path,
) -> None:
    if protocol_main_table.empty or dense_exact_table.empty or budget_table.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))

    x = np.arange(len(protocol_main_table), dtype=float)
    width = 0.32
    axes[0].bar(x - width / 2.0, protocol_main_table["global_overall_regret_score_mean"], width=width, color="#9D9D9D", label="Global")
    axes[0].bar(x + width / 2.0, protocol_main_table["adaptive_overall_regret_score_mean"], width=width, color="#2E8B57", label="Adaptive")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(protocol_main_table["protocol_label"], rotation=18, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Overall regret", fontsize=9)
    axes[0].set_title("A. Normalized overall regret", fontsize=10)
    axes[0].grid(True, axis="y", which="both", alpha=0.25, linewidth=0.6)

    for protocol_name in PROTOCOL_ORDER:
        subset = dense_exact_table[dense_exact_table["protocol_name"] == protocol_name].sort_values("observation_noise_relative")
        if subset.empty:
            continue
        axes[1].plot(
            subset["observation_noise_relative"],
            subset["adaptive_rmse"],
            color=PROTOCOL_COLORS[protocol_name],
            linewidth=2.0,
            marker="s",
            markersize=5.0,
            label=PROTOCOL_LABELS[protocol_name],
        )
    axes[1].set_xlabel("Observation noise", fontsize=9)
    axes[1].set_ylabel("Adaptive RMSE", fontsize=9)
    axes[1].set_title("B. Dense exact-regime RMSE", fontsize=10)
    axes[1].grid(True, alpha=0.25, linewidth=0.6)

    budget_x = np.arange(len(budget_table), dtype=float)
    axes[2].bar(
        budget_x,
        budget_table["transport_total_observations"],
        color=[PROTOCOL_COLORS.get(name, "#4C78A8") for name in budget_table["protocol_name"]],
        width=0.62,
    )
    axes[2].set_xticks(budget_x)
    axes[2].set_xticklabels(budget_table["protocol_label"], rotation=18, ha="right", fontsize=8.5)
    axes[2].set_ylabel("Total observations", fontsize=9)
    axes[2].set_title("C. Evidence budget", fontsize=10)
    axes[2].grid(True, axis="y", alpha=0.25, linewidth=0.6)
    reference_values = budget_table["transport_budget_reference_total_observations"].dropna().unique()
    if reference_values.size > 0:
        axes[2].axhline(float(reference_values[0]), color="black", linewidth=1.1, linestyle="--", alpha=0.8)

    handles = [
        plt.Line2D([0], [0], color="#9D9D9D", linewidth=6, label="Global"),
        plt.Line2D([0], [0], color="#2E8B57", linewidth=6, label="Adaptive"),
    ] + [
        plt.Line2D([0], [0], color=PROTOCOL_COLORS[name], linewidth=2.5, marker="s", label=PROTOCOL_LABELS[name])
        for name in PROTOCOL_ORDER
    ]
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle in handles:
        label = handle.get_label()
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    fig.legend(unique_handles, unique_labels, loc="upper center", ncol=5, frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("P06 budget-matched protocol ablation", y=1.13, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_paper_notes(
    protocol_main_table: pd.DataFrame,
    dense_exact_table: pd.DataFrame,
    budget_table: pd.DataFrame,
    output_path: Path,
) -> str:
    if protocol_main_table.empty or dense_exact_table.empty or budget_table.empty:
        output_path.write_text("", encoding="utf-8")
        return ""

    matched = protocol_main_table[
        protocol_main_table["protocol_name"] == "modflow6_mt3dms_p06_phaseprotocol_budgetmatched"
    ].iloc[0]
    uniform = protocol_main_table[
        protocol_main_table["protocol_name"] == "modflow6_mt3dms_p06_uniform_reference"
    ].iloc[0]
    full_phase = protocol_main_table[
        protocol_main_table["protocol_name"] == "modflow6_mt3dms_p06"
    ].iloc[0]
    dense_002 = dense_exact_table[
        (dense_exact_table["protocol_name"] == "modflow6_mt3dms_p06_phaseprotocol_budgetmatched")
        & (dense_exact_table["observation_noise_relative"] == 0.02)
    ].iloc[0]

    lines = [
        "# Paper Budget-Ablation Notes",
        "",
        "## Main table takeaway",
        "",
        (
            f"The budget-matched phase protocol achieves the lowest adaptive mean overall regret "
            f"({float(matched['adaptive_overall_regret_score_mean']):.6f}), improving over the "
            f"uniform reference ({float(uniform['adaptive_overall_regret_score_mean']):.6f}) "
            f"and the full higher-budget phase protocol ({float(full_phase['adaptive_overall_regret_score_mean']):.6f})."
        ),
        "",
        "## Dense regime takeaway",
        "",
        (
            f"At `pathline_monitoring64` with exact solver and observation noise 0.02, the "
            f"budget-matched phase protocol reduces adaptive RMSE to {float(dense_002['adaptive_rmse']):.6f} "
            f"while holding the total operator-observation budget at "
            f"{int(round(float(dense_002['transport_total_observations'])))}."
        ),
        "",
        "## Budget framing",
        "",
        (
            "The fairness claim is now simple to state: the budget-matched protocol uses the same "
            "dense-regime total evidence budget as the uniform reference, while the full phase "
            "protocol still represents the higher-budget upper-bound protocol."
        ),
        "",
    ]
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return text


def export_paper_budget_ablation(sweep_dir: Path) -> dict[str, str]:
    normalized_summary = _load_csv(sweep_dir / "benchmark_normalized_summary.csv")
    all_metrics = _load_csv(sweep_dir / "all_metrics.csv")

    paper_tables_dir = sweep_dir / "paper_tables"
    paper_figures_dir = sweep_dir / "paper_figures"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    protocol_main_table = build_protocol_main_table(
        normalized_summary,
        paper_tables_dir / "table_p06_budget_ablation_main.csv",
    )
    _write_latex_table(
        protocol_main_table,
        paper_tables_dir / "table_p06_budget_ablation_main.tex",
        float_columns=[
            "global_overall_regret_score_mean",
            "adaptive_overall_regret_score_mean",
            "adaptive_gain_vs_global",
            "global_rmse_relative_regret_mean",
            "adaptive_rmse_relative_regret_mean",
        ],
        percent_columns=[
            "global_near_best_rmse10_share",
            "adaptive_near_best_rmse10_share",
            "global_near_best_capture10_share",
            "adaptive_near_best_capture10_share",
        ],
    )
    dense_exact_table = build_dense_exact_table(
        all_metrics,
        paper_tables_dir / "table_p06_budget_ablation_dense_exact.csv",
    )
    _write_latex_table(
        dense_exact_table,
        paper_tables_dir / "table_p06_budget_ablation_dense_exact.tex",
        float_columns=[
            "observation_noise_relative",
            "global_rmse",
            "adaptive_rmse",
            "adaptive_minus_global_rmse",
            "adaptive_over_global_rmse_ratio",
            "transport_total_observations",
            "transport_budget_reference_total_observations",
        ],
    )
    budget_table = build_budget_table(
        dense_exact_table,
        paper_tables_dir / "table_p06_budget_ablation_budget.csv",
    )
    _write_latex_table(
        budget_table,
        paper_tables_dir / "table_p06_budget_ablation_budget.tex",
        float_columns=[
            "transport_total_observations",
            "transport_budget_reference_total_observations",
            "observation_budget_ratio_vs_uniform",
        ],
    )
    build_paper_notes(
        protocol_main_table,
        dense_exact_table,
        budget_table,
        paper_tables_dir / "paper_budget_ablation_notes.md",
    )

    _plot_overall_regret(protocol_main_table, paper_figures_dir / "fig_p06_budget_ablation_overall_regret")
    _plot_dense_rmse(dense_exact_table, paper_figures_dir / "fig_p06_budget_ablation_dense_exact_rmse")
    _plot_budget_totals(budget_table, paper_figures_dir / "fig_p06_budget_ablation_budget_totals")
    _plot_compound_overview(
        protocol_main_table,
        dense_exact_table,
        budget_table,
        paper_figures_dir / "fig_p06_budget_ablation_overview",
    )

    return {
        "protocol_main_table": str(paper_tables_dir / "table_p06_budget_ablation_main.csv"),
        "protocol_main_table_tex": str(paper_tables_dir / "table_p06_budget_ablation_main.tex"),
        "dense_exact_table": str(paper_tables_dir / "table_p06_budget_ablation_dense_exact.csv"),
        "dense_exact_table_tex": str(paper_tables_dir / "table_p06_budget_ablation_dense_exact.tex"),
        "budget_table": str(paper_tables_dir / "table_p06_budget_ablation_budget.csv"),
        "budget_table_tex": str(paper_tables_dir / "table_p06_budget_ablation_budget.tex"),
        "notes_md": str(paper_tables_dir / "paper_budget_ablation_notes.md"),
        "overall_regret_figure": str(paper_figures_dir / "fig_p06_budget_ablation_overall_regret.png"),
        "dense_exact_figure": str(paper_figures_dir / "fig_p06_budget_ablation_dense_exact_rmse.png"),
        "budget_totals_figure": str(paper_figures_dir / "fig_p06_budget_ablation_budget_totals.png"),
        "overview_figure": str(paper_figures_dir / "fig_p06_budget_ablation_overview.png"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export manuscript-ready tables and figures for the P06 budget ablation.")
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Path to the ablation sweep directory containing all_metrics.csv and benchmark_normalized_summary.csv.",
    )
    args = parser.parse_args(argv)

    outputs = export_paper_budget_ablation(Path(args.sweep_dir))
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
