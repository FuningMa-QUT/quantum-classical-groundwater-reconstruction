"""Consolidate manuscript figures and tables into a journal-style export bundle."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantum_hydro.paper_budget_ablation import export_paper_budget_ablation


@dataclass(frozen=True)
class FigureSpec:
    number: int
    stem: str
    caption: str
    width: str
    source_dir: Path


@dataclass(frozen=True)
class TableSpec:
    number: int
    source_csv: Path
    caption: str
    width: str
    source_tex: Path | None = None


def _copy_pair(source_stem: Path, destination_stem: Path) -> list[Path]:
    copied: list[Path] = []
    for suffix in (".png", ".pdf"):
        source = source_stem.with_suffix(suffix)
        if source.exists():
            destination = destination_stem.with_suffix(suffix)
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def _format_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        value = float(value)
        if value == 0.0:
            return "0"
        if abs(value) >= 1e4 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _write_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        output_path.write_text("", encoding="utf-8")
        return

    columns = [str(column) for column in df.columns]
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(column.replace("_", "\\_") for column in columns) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        values = []
        for column in df.columns:
            text = _format_scalar(row[column]).replace("_", "\\_")
            values.append(text)
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_table(spec: TableSpec, destination_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(spec.source_csv)
    csv_target = destination_dir / f"Table_{spec.number:02d}.csv"
    tex_target = destination_dir / f"Table_{spec.number:02d}.tex"
    shutil.copy2(spec.source_csv, csv_target)
    if spec.source_tex and spec.source_tex.exists():
        shutil.copy2(spec.source_tex, tex_target)
    else:
        _write_latex_table(df, tex_target)
    return csv_target, tex_target


def _build_manifest(
    output_path: Path,
    figure_specs: list[FigureSpec],
    table_specs: list[TableSpec],
) -> None:
    lines = [
        "# Manuscript Asset Manifest",
        "",
        "This directory consolidates the current paper figures and tables into a manuscript-ready bundle.",
        "The suggested layout follows a WRR-style main-text flow, with single-column figures reserved for simple comparisons and full-width figures reserved for multi-panel syntheses and heatmaps.",
        "",
        "## Figures",
        "",
    ]
    for spec in figure_specs:
        lines.extend(
            [
                f"- `Figure_{spec.number:02d}`",
                f"  Caption: {spec.caption}",
                f"  Suggested width: {spec.width}",
            ]
        )
    lines.extend(["", "## Tables", ""])
    for spec in table_specs:
        lines.extend(
            [
                f"- `Table_{spec.number:02d}`",
                f"  Caption: {spec.caption}",
                f"  Suggested width: {spec.width}",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended Main-Text Order",
            "",
            "1. Benchmark suite setup and phase-aware protocol.",
            "2. Benchmark-normalized performance across P05, P06, and P09.",
            "3. Capture-oriented diagnostics for the injection-extraction benchmark.",
            "4. Budget-matched P06 ablation and dense-regime sensitivity.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_manuscript_assets(repo_root: Path, output_dir: Path) -> dict[str, str]:
    suite_dir = repo_root / "outputs" / "sweeps_phaseprotocol_confirm" / "public_modflow_benchmark_suite"
    budget_dir = repo_root / "outputs" / "sweeps_budget_ablation" / "public_modflow_p06_budget_ablation"

    export_paper_budget_ablation(budget_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_specs = [
        FigureSpec(
            1,
            "fig_benchmark_phase_diagram",
            "Benchmark configurations and phase-aware monitoring layout for the public MODFLOW 6 suite.",
            "full-width",
            suite_dir / "figures",
        ),
        FigureSpec(
            2,
            "fig_benchmark_regret_heatmap",
            "Benchmark-normalized regret across public benchmarks, noise regimes, and reconstruction methods.",
            "full-width",
            suite_dir / "figures",
        ),
        FigureSpec(
            3,
            "fig_benchmark_rmse_sensitivity",
            "RMSE sensitivity to observation noise and anchor density across the public benchmark suite.",
            "single-column",
            suite_dir / "figures",
        ),
        FigureSpec(
            4,
            "fig_benchmark_adaptive_gain",
            "Adaptive regional gain relative to global MCR across benchmark and noise regimes.",
            "single-column",
            suite_dir / "figures",
        ),
        FigureSpec(
            5,
            "fig_capture_tradeoff",
            "Tradeoff between concentration reconstruction and extraction-capture fidelity in the injection-extraction benchmark.",
            "single-column",
            suite_dir / "figures",
        ),
        FigureSpec(
            6,
            "fig_capture_error_sensitivity",
            "Sensitivity of extraction-capture error to monitoring strategy and noise level.",
            "single-column",
            suite_dir / "figures",
        ),
        FigureSpec(
            7,
            "fig_p06_budget_ablation_overview",
            "Overview of the P06 budget-matched ablation, including normalized regret, dense-regime RMSE, and total evidence budget.",
            "full-width",
            budget_dir / "paper_figures",
        ),
        FigureSpec(
            8,
            "fig_p06_budget_ablation_overall_regret",
            "Mean overall regret for uniform, full-phase, and budget-matched P06 protocols.",
            "single-column",
            budget_dir / "paper_figures",
        ),
        FigureSpec(
            9,
            "fig_p06_budget_ablation_dense_exact_rmse",
            "Dense-regime exact-solver RMSE under shared observation noise for the P06 ablation.",
            "single-column",
            budget_dir / "paper_figures",
        ),
        FigureSpec(
            10,
            "fig_p06_budget_ablation_budget_totals",
            "Total operator-observation budget used by each P06 protocol.",
            "single-column",
            budget_dir / "paper_figures",
        ),
    ]

    table_specs = [
        TableSpec(
            1,
            suite_dir / "benchmark_winner_summary.csv",
            "Winning reconstruction method by benchmark, noise regime, and anchor density in the public MODFLOW 6 suite.",
            "full-width",
        ),
        TableSpec(
            2,
            suite_dir / "capture_manuscript_table.csv",
            "Capture-focused diagnostics for the P06 injection-extraction benchmark.",
            "full-width",
        ),
        TableSpec(
            3,
            suite_dir / "method_phase_table.csv",
            "Phase-specific method performance in the public benchmark suite.",
            "full-width",
        ),
        TableSpec(
            4,
            suite_dir / "transport_diagnostics_table.csv",
            "Transport diagnostics including centroid, spread, and control-plane metrics.",
            "full-width",
        ),
        TableSpec(
            5,
            budget_dir / "paper_tables" / "table_p06_budget_ablation_main.csv",
            "Main P06 budget-ablation summary in benchmark-normalized space.",
            "full-width",
            budget_dir / "paper_tables" / "table_p06_budget_ablation_main.tex",
        ),
        TableSpec(
            6,
            budget_dir / "paper_tables" / "table_p06_budget_ablation_dense_exact.csv",
            "Dense-regime exact-solver RMSE under shared observation noise for the P06 budget ablation.",
            "single-column",
            budget_dir / "paper_tables" / "table_p06_budget_ablation_dense_exact.tex",
        ),
        TableSpec(
            7,
            budget_dir / "paper_tables" / "table_p06_budget_ablation_budget.csv",
            "Observation-budget comparison for the uniform, full-phase, and budget-matched P06 protocols.",
            "single-column",
            budget_dir / "paper_tables" / "table_p06_budget_ablation_budget.tex",
        ),
    ]

    exported: dict[str, str] = {}
    for spec in figure_specs:
        target_stem = figures_dir / f"Figure_{spec.number:02d}"
        copied = _copy_pair(spec.source_dir / spec.stem, target_stem)
        for path in copied:
            exported[path.stem + path.suffix] = str(path)

    for spec in table_specs:
        csv_target, tex_target = _copy_table(spec, tables_dir)
        exported[csv_target.name] = str(csv_target)
        exported[tex_target.name] = str(tex_target)

    notes_source = budget_dir / "paper_tables" / "paper_budget_ablation_notes.md"
    if notes_source.exists():
        notes_target = output_dir / "P06_budget_ablation_notes.md"
        shutil.copy2(notes_source, notes_target)
        exported[notes_target.name] = str(notes_target)

    manifest_path = output_dir / "MANIFEST.md"
    _build_manifest(manifest_path, figure_specs, table_specs)
    exported[manifest_path.name] = str(manifest_path)

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manuscript-ready figures and tables into a journal-style bundle.")
    parser.add_argument(
        "--output-dir",
        default="outputs/manuscript_export/top_journal_bundle",
        help="Output directory for the consolidated manuscript assets.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()
    exported = export_manuscript_assets(repo_root, output_dir)
    for name, path in exported.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
