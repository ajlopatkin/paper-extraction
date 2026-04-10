from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from paper_analysis.config import ComparisonRule, EvaluationConfig, PocConfig
from paper_analysis.plot_type_dispatch import ExtractionModel, parse_extraction_dict
from paper_analysis.schemas import (
    BoxPlotExtraction,
    ExperimentalWorkflowExtraction,
    LineChartExtraction,
    PlasmidMapExtraction,
    TableImageExtraction,
    UnknownPlotExtraction,
    WorkflowDiagramExtraction,
)


def parse_spreadsheet_value(raw: object, strip_tilde: bool) -> float:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        raise ValueError("empty cell")
    t = str(raw).strip()
    if strip_tilde and t.startswith("~"):
        t = t[1:].strip()
    return float(t)


def within_tolerance(expected: float, actual: float, rtol: float, atol: float) -> bool:
    if math.isnan(expected) or math.isnan(actual):
        return False
    return abs(actual - expected) <= atol + rtol * abs(expected)


def load_extraction(path: Path) -> ExtractionModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_extraction_dict(data, context=str(path))


@dataclass
class RowCompare:
    label: str
    expected: float | None
    extracted: float | None
    ok: bool
    note: str = ""


@dataclass
class FigureEval:
    figure_id: str
    source_contains: str
    rows: list[RowCompare] = field(default_factory=list)
    mae: float | None = None

    def summary(self) -> str:
        lines = [
            f"=== {self.figure_id} (source contains {self.source_contains!r}) ===",
        ]
        if self.mae is not None:
            lines.append(f"MAE (matched pairs): {self.mae:.4g}")
        for r in self.rows:
            status = "OK" if r.ok else "FAIL"
            lines.append(
                f"  [{status}] {r.label!r}: expected={r.expected!r} extracted={r.extracted!r} {r.note}".strip()
            )
        return "\n".join(lines)


def filter_sheet(df: pd.DataFrame, source_column: str, contains: str) -> pd.DataFrame:
    if source_column not in df.columns:
        raise KeyError(f"Missing column {source_column!r}. Available: {list(df.columns)}")
    mask = df[source_column].astype(str).str.contains(contains, case=False, regex=False, na=False)
    return df.loc[mask]


def evaluate_comparison(
    extraction: ExtractionModel,
    df: pd.DataFrame,
    rule: ComparisonRule,
    source_column: str,
    rtol: float,
    atol: float,
) -> FigureEval:
    sub = filter_sheet(df, source_column, rule.source_contains)
    out = FigureEval(figure_id="", source_contains=rule.source_contains)

    if isinstance(extraction, UnknownPlotExtraction):
        decl = extraction.declared_plot_type or "missing"
        nk = len(extraction.raw)
        out.rows = [
            RowCompare(
                label="(unknown plot_type)",
                expected=None,
                extracted=None,
                ok=False,
                note=f"POC: plot_type {decl!r} not supported ({nk} top-level keys in raw); see JSON.",
            )
        ]
        return out

    if isinstance(extraction, PlasmidMapExtraction):
        nf = len(extraction.features)
        nl = len(extraction.other_visible_labels)
        out.rows = [
            RowCompare(
                label="(plasmid_map)",
                expected=None,
                extracted=None,
                ok=False,
                note=f"POC: no numeric compare for plasmid_map ({nf} features, {nl} other labels); see JSON.",
            )
        ]
        return out

    if isinstance(extraction, WorkflowDiagramExtraction):
        nn = len(extraction.nodes)
        ne = len(extraction.edges)
        out.rows = [
            RowCompare(
                label="(workflow_diagram)",
                expected=None,
                extracted=None,
                ok=False,
                note=f"POC: no numeric compare for workflow_diagram ({nn} nodes, {ne} edges); see JSON.",
            )
        ]
        return out

    if isinstance(extraction, ExperimentalWorkflowExtraction):
        nn = len(extraction.nodes)
        ne = len(extraction.edges)
        out.rows = [
            RowCompare(
                label="(experimental_workflow)",
                expected=None,
                extracted=None,
                ok=False,
                note=f"POC: no numeric compare for experimental_workflow ({nn} nodes, {ne} edges); see JSON.",
            )
        ]
        return out

    if isinstance(extraction, TableImageExtraction):
        nrows = len(extraction.rows)
        nh = len(extraction.column_headers)
        out.rows = [
            RowCompare(
                label="(table_image)",
                expected=None,
                extracted=None,
                ok=False,
                note=f"POC: no numeric compare for table_image ({nh} headers, {nrows} data rows); see JSON.",
            )
        ]
        return out

    if isinstance(extraction, BoxPlotExtraction):
        if not rule.label_column:
            raise ValueError(f"box_plot comparison needs label_column for {rule.source_contains!r}")
        extracted_map = {g.label.strip().lower(): g.median for g in extraction.groups if g.median is not None}
        rows: list[RowCompare] = []
        errors: list[float] = []
        for _, row in sub.iterrows():
            lab = str(row[rule.label_column]).strip()
            key = lab.lower()
            exp_raw = row[rule.value_column]
            try:
                expected = parse_spreadsheet_value(exp_raw, rule.strip_tilde)
            except (ValueError, TypeError):
                rows.append(
                    RowCompare(
                        label=lab,
                        expected=None,
                        extracted=extracted_map.get(key),
                        ok=False,
                        note="could not parse expected value",
                    )
                )
                continue
            ext = extracted_map.get(key)
            if ext is None:
                rows.append(
                    RowCompare(
                        label=lab,
                        expected=expected,
                        extracted=None,
                        ok=False,
                        note="no extracted median for label",
                    )
                )
                continue
            ok = within_tolerance(expected, ext, rtol, atol)
            rows.append(RowCompare(label=lab, expected=expected, extracted=ext, ok=ok))
            errors.append(abs(ext - expected))
        mae = sum(errors) / len(errors) if errors else None
        out.rows = rows
        out.mae = mae
        return out

    # line_chart: align by row order within filtered sheet vs first series points (y)
    if not extraction.series:
        out.rows = [
            RowCompare(
                label="(line)",
                expected=None,
                extracted=None,
                ok=False,
                note="no series in extraction",
            )
        ]
        return out
    points = extraction.series[0].points
    ys = [p.y for p in points]
    n = min(len(sub), len(ys))
    rows = []
    errs: list[float] = []
    sub_reset = sub.reset_index(drop=True)
    for i in range(n):
        row = sub_reset.iloc[i]
        lab = str(row[rule.label_column]).strip() if rule.label_column else f"row_{i}"
        try:
            expected = parse_spreadsheet_value(row[rule.value_column], rule.strip_tilde)
        except (ValueError, TypeError):
            rows.append(
                RowCompare(
                    label=lab,
                    expected=None,
                    extracted=ys[i],
                    ok=False,
                    note="could not parse expected value",
                )
            )
            continue
        ext = ys[i]
        ok = within_tolerance(expected, ext, rtol, atol)
        rows.append(RowCompare(label=lab, expected=expected, extracted=ext, ok=ok))
        errs.append(abs(ext - expected))
    if len(sub) != len(ys):
        rows.append(
            RowCompare(
                label="(count)",
                expected=float(len(sub)),
                extracted=float(len(ys)),
                ok=len(sub) == len(ys),
                note="row count vs extracted point count",
            )
        )
    out.rows = rows
    out.mae = sum(errs) / len(errs) if errs else None
    return out


def run_evaluation(cfg: PocConfig) -> list[FigureEval]:
    ev = cfg.evaluation
    path = ev.spreadsheet_path
    if not path.is_file():
        raise FileNotFoundError(f"Spreadsheet not found: {path}")
    df = pd.read_excel(path, sheet_name=ev.sheet)
    results: list[FigureEval] = []
    ext_dir = cfg.artifacts.extractions_dir
    for rule in ev.comparisons:
        jp = ext_dir / f"{rule.figure_id}.json"
        if not jp.is_file():
            raise FileNotFoundError(f"Missing extraction JSON: {jp} (run run-vision first)")
        extraction = load_extraction(jp)
        fe = evaluate_comparison(
            extraction,
            df,
            rule,
            source_column=ev.source_column,
            rtol=ev.relative_tolerance,
            atol=ev.absolute_tolerance,
        )
        fe.figure_id = rule.figure_id
        results.append(fe)
    return results


def inspect_measurements(cfg: PocConfig) -> None:
    ev = cfg.evaluation
    path = ev.spreadsheet_path
    if not path.is_file():
        raise FileNotFoundError(f"Spreadsheet not found: {path}")
    df = pd.read_excel(path, sheet_name=ev.sheet)
    print(f"Sheet: {ev.sheet!r}  rows={len(df)}  columns={list(df.columns)}")
    seen: set[str] = set()
    for rule in ev.comparisons:
        key = rule.source_contains
        if key in seen:
            continue
        seen.add(key)
        sub = filter_sheet(df, ev.source_column, key)
        print(f"\n--- Filter {ev.source_column!r} contains {key!r} -> {len(sub)} rows ---")
        if len(sub):
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(sub.head(20).to_string())


def format_report(results: list[FigureEval]) -> str:
    return "\n\n".join(r.summary() for r in results)
