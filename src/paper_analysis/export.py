from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paper_analysis.config import (
    ExportConfig,
    PocConfig,
    effective_export_config,
    effective_text_config,
)
from paper_analysis.evaluate import load_extraction
from paper_analysis.runs_schemas import RunExtractionBatch
from paper_analysis.schemas import (
    BoxPlotExtraction,
    ExperimentalWorkflowExtraction,
    LineChartExtraction,
    PlasmidMapExtraction,
    TableImageExtraction,
    UnknownPlotExtraction,
    WorkflowDiagramExtraction,
)
from paper_analysis.text_schemas import TextExtractionBatch

_MAX_UNKNOWN_RAW_NOTE_CHARS = 8000

# Core columns align with typical ground-truth Measurements; trailing columns are extraction metadata.
MEASUREMENT_COLUMNS: list[str] = [
    "Run_ID",
    "Field_name",
    "Raw_value",
    "Raw_units",
    "Source_in_paper",
    "Notes",
    "Extraction_pipeline",
    "Extraction_method",
    "Source_artifact",
    "Source_id",
    "Plot_type",
    "Page",
    "Axis_or_context",
    "Confidence",
    "Supporting_evidence",
]

_PAGE_PREFIX_RE = re.compile(r"^p(\d+)_", re.IGNORECASE)


def _rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _page_from_source_id(source_id: str) -> str:
    m = _PAGE_PREFIX_RE.match(source_id)
    return str(int(m.group(1))) if m else ""


def _axis_context(
    y_label: str | None,
    y_units: str | None,
    x_label: str | None = None,
    x_units: str | None = None,
) -> str:
    parts: list[str] = []
    if y_label or y_units:
        yu = f" ({y_units})" if y_units else ""
        parts.append(f"y: {(y_label or '')}{yu}".strip())
    if x_label or x_units:
        xu = f" ({x_units})" if x_units else ""
        parts.append(f"x: {(x_label or '')}{xu}".strip())
    return " | ".join(parts)


def _base_row(
    *,
    field_name: str,
    raw_value: str,
    raw_units: str,
    source_in_paper: str,
    notes: str,
    pipeline: str,
    method: str,
    artifact: str,
    source_id: str,
    plot_type: str,
    page: str,
    axis_or_context: str,
    confidence: str,
    supporting_evidence: str,
    run_id: str = "",
) -> dict[str, str]:
    return {
        "Run_ID": run_id,
        "Field_name": field_name,
        "Raw_value": raw_value,
        "Raw_units": raw_units,
        "Source_in_paper": source_in_paper,
        "Notes": notes,
        "Extraction_pipeline": pipeline,
        "Extraction_method": method,
        "Source_artifact": artifact,
        "Source_id": source_id,
        "Plot_type": plot_type,
        "Page": page,
        "Axis_or_context": axis_or_context,
        "Confidence": confidence,
        "Supporting_evidence": supporting_evidence,
    }


def _rows_workflow_like(
    ext: WorkflowDiagramExtraction | ExperimentalWorkflowExtraction,
    *,
    stem: str,
    rel: str,
    page: str,
    plot_type: str,
    human_name: str,
    method_prefix: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    meta_parts: list[str] = []
    if ext.title_or_caption:
        meta_parts.append(f"caption={ext.title_or_caption}")
    if ext.notes:
        meta_parts.append(ext.notes)
    meta = "; ".join(meta_parts)
    for ni, node in enumerate(ext.nodes):
        nn = node.notes or ""
        if node.node_type:
            nn = f"type={node.node_type}; {nn}".strip("; ")
        rows.append(
            _base_row(
                field_name=node.label,
                raw_value="",
                raw_units="",
                source_in_paper=f"{human_name} (vision): {stem}",
                notes=nn,
                pipeline="figure_vision",
                method=f"{method_prefix}_node",
                artifact=rel,
                source_id=stem,
                plot_type=plot_type,
                page=page,
                axis_or_context=meta,
                confidence="",
                supporting_evidence=f"node_index={ni}",
            )
        )
    for ei, edge in enumerate(ext.edges):
        elab = edge.edge_label or ""
        rows.append(
            _base_row(
                field_name=f"{edge.from_label} → {edge.to_label}",
                raw_value="",
                raw_units="",
                source_in_paper=f"{human_name} (vision): {stem}",
                notes=f"edge_label={elab}" if elab else "",
                pipeline="figure_vision",
                method=f"{method_prefix}_edge",
                artifact=rel,
                source_id=stem,
                plot_type=plot_type,
                page=page,
                axis_or_context=meta,
                confidence="",
                supporting_evidence=f"edge_index={ei}",
            )
        )
    for li, lab in enumerate(ext.other_visible_labels):
        if not str(lab).strip():
            continue
        rows.append(
            _base_row(
                field_name=f"[label] {lab}",
                raw_value="",
                raw_units="",
                source_in_paper=f"{human_name} (vision): {stem}",
                notes="other_visible_labels",
                pipeline="figure_vision",
                method=f"{method_prefix}_label",
                artifact=rel,
                source_id=stem,
                plot_type=plot_type,
                page=page,
                axis_or_context=meta,
                confidence="",
                supporting_evidence=f"label_index={li}",
            )
        )
    if not rows and ext.notes:
        rows.append(
            _base_row(
                field_name=f"({plot_type} summary)",
                raw_value="",
                raw_units="",
                source_in_paper=f"{human_name} (vision): {stem}",
                notes=ext.notes,
                pipeline="figure_vision",
                method=f"{method_prefix}_empty",
                artifact=rel,
                source_id=stem,
                plot_type=plot_type,
                page=page,
                axis_or_context=meta,
                confidence="",
                supporting_evidence="",
            )
        )
    return rows


def rows_from_figure_json(path: Path) -> list[dict[str, str]]:
    rel = _rel_path(path)
    stem = path.stem
    page = _page_from_source_id(stem)
    try:
        ext = load_extraction(path)
    except Exception as e:  # noqa: BLE001 — export should not fail entire workbook
        return [
            _base_row(
                field_name="(parse_error)",
                raw_value="",
                raw_units="",
                source_in_paper=f"Figure vision ({stem})",
                notes=str(e),
                pipeline="figure_vision",
                method="error",
                artifact=rel,
                source_id=stem,
                plot_type="",
                page=page,
                axis_or_context="",
                confidence="",
                supporting_evidence="",
            )
        ]

    if isinstance(ext, BoxPlotExtraction):
        rows: list[dict[str, str]] = []
        axis_ctx = _axis_context(ext.axis_y_label, ext.axis_y_units, ext.axis_x_label, None)
        for g in ext.groups:
            stat_bits: list[str] = []
            if g.q1 is not None:
                stat_bits.append(f"q1={g.q1}")
            if g.q3 is not None:
                stat_bits.append(f"q3={g.q3}")
            if g.whisker_low is not None:
                stat_bits.append(f"whisker_low={g.whisker_low}")
            if g.whisker_high is not None:
                stat_bits.append(f"whisker_high={g.whisker_high}")
            if g.significance:
                stat_bits.append(f"sig={g.significance}")
            notes = "; ".join(stat_bits)
            rows.append(
                _base_row(
                    field_name=g.label,
                    raw_value="" if g.median is None else str(g.median),
                    raw_units=ext.axis_y_units or "",
                    source_in_paper=f"Figure crop (vision): {stem}",
                    notes=notes,
                    pipeline="figure_vision",
                    method="box_plot_median",
                    artifact=rel,
                    source_id=stem,
                    plot_type="box_plot",
                    page=page,
                    axis_or_context=axis_ctx,
                    confidence="",
                    supporting_evidence="",
                )
            )
        return rows

    if isinstance(ext, LineChartExtraction):
        rows = []
        axis_ctx = _axis_context(
            ext.axis_y_label,
            ext.axis_y_units,
            ext.axis_x_label,
            ext.axis_x_units,
        )
        for si, series in enumerate(ext.series):
            sname = series.name or f"series_{si + 1}"
            for pi, pt in enumerate(series.points):
                err_bits: list[str] = []
                if pt.error_low is not None or pt.error_high is not None:
                    err_bits.append(f"err=[{pt.error_low},{pt.error_high}]")
                note = "; ".join(err_bits)
                rows.append(
                    _base_row(
                        field_name=f"{sname} | x={pt.x}",
                        raw_value=str(pt.y),
                        raw_units=ext.axis_y_units or "",
                        source_in_paper=f"Figure crop (vision): {stem}",
                        notes=note,
                        pipeline="figure_vision",
                        method=f"{ext.plot_type}_point",
                        artifact=rel,
                        source_id=stem,
                        plot_type=ext.plot_type,
                        page=page,
                        axis_or_context=axis_ctx,
                        confidence="",
                        supporting_evidence=f"series_index={si} point_index={pi}",
                    )
                )
        return rows

    if isinstance(ext, TableImageExtraction):
        rows = []
        cap = ext.title_or_caption or ""
        caveats = ext.notes or ""
        headers = ext.column_headers
        for ri, row_cells in enumerate(ext.rows):
            for ci, cell in enumerate(row_cells):
                if not str(cell).strip():
                    continue
                hdr = headers[ci] if ci < len(headers) else f"col{ci + 1}"
                note_parts = [f"row={ri + 1} col={ci + 1}"]
                if cap:
                    note_parts.append(f"caption={cap}")
                if caveats:
                    note_parts.append(f"caveats={caveats}")
                rows.append(
                    _base_row(
                        field_name=f"{hdr} (row {ri + 1})",
                        raw_value=str(cell),
                        raw_units="",
                        source_in_paper=f"Table-as-image (vision): {stem}",
                        notes="; ".join(note_parts),
                        pipeline="figure_vision",
                        method="table_image_cell",
                        artifact=rel,
                        source_id=stem,
                        plot_type="table_image",
                        page=page,
                        axis_or_context="",
                        confidence="",
                        supporting_evidence="",
                    )
                )
        return rows

    if isinstance(ext, PlasmidMapExtraction):
        rows = []
        circ = (
            "circular"
            if ext.is_circular is True
            else "linear"
            if ext.is_circular is False
            else "unknown"
        )
        meta = f"map={ext.map_name_or_identifier or ''}; layout={circ}"
        if ext.title_or_caption:
            meta = f"{meta}; caption={ext.title_or_caption}"
        if ext.notes:
            meta = f"{meta}; {ext.notes}"
        for fi, feat in enumerate(ext.features):
            fnote = feat.notes or ""
            if feat.feature_type:
                fnote = f"type={feat.feature_type}; {fnote}".strip("; ")
            rows.append(
                _base_row(
                    field_name=feat.label,
                    raw_value="",
                    raw_units="",
                    source_in_paper=f"Plasmid map (vision): {stem}",
                    notes=fnote,
                    pipeline="figure_vision",
                    method="plasmid_map_feature",
                    artifact=rel,
                    source_id=stem,
                    plot_type="plasmid_map",
                    page=page,
                    axis_or_context=meta,
                    confidence="",
                    supporting_evidence=f"feature_index={fi}",
                )
            )
        for li, lab in enumerate(ext.other_visible_labels):
            if not str(lab).strip():
                continue
            rows.append(
                _base_row(
                    field_name=f"[label] {lab}",
                    raw_value="",
                    raw_units="",
                    source_in_paper=f"Plasmid map (vision): {stem}",
                    notes="other_visible_labels",
                    pipeline="figure_vision",
                    method="plasmid_map_label",
                    artifact=rel,
                    source_id=stem,
                    plot_type="plasmid_map",
                    page=page,
                    axis_or_context=meta,
                    confidence="",
                    supporting_evidence=f"label_index={li}",
                )
            )
        if not rows and (ext.notes or ext.map_name_or_identifier):
            rows.append(
                _base_row(
                    field_name="(plasmid_map summary)",
                    raw_value="",
                    raw_units="",
                    source_in_paper=f"Plasmid map (vision): {stem}",
                    notes=ext.notes or meta,
                    pipeline="figure_vision",
                    method="plasmid_map_empty_features",
                    artifact=rel,
                    source_id=stem,
                    plot_type="plasmid_map",
                    page=page,
                    axis_or_context=meta,
                    confidence="",
                    supporting_evidence="",
                )
            )
        return rows

    if isinstance(ext, WorkflowDiagramExtraction):
        return _rows_workflow_like(
            ext,
            stem=stem,
            rel=rel,
            page=page,
            plot_type="workflow_diagram",
            human_name="Workflow diagram",
            method_prefix="workflow_diagram",
        )

    if isinstance(ext, ExperimentalWorkflowExtraction):
        return _rows_workflow_like(
            ext,
            stem=stem,
            rel=rel,
            page=page,
            plot_type="experimental_workflow",
            human_name="Experimental workflow",
            method_prefix="experimental_workflow",
        )

    if isinstance(ext, UnknownPlotExtraction):
        decl = ext.declared_plot_type or "missing"
        raw_json = json.dumps(ext.raw, ensure_ascii=False, indent=2)
        if len(raw_json) > _MAX_UNKNOWN_RAW_NOTE_CHARS:
            raw_json = raw_json[: _MAX_UNKNOWN_RAW_NOTE_CHARS - 3] + "..."
        return [
            _base_row(
                field_name=f"(unknown plot_type: {decl})",
                raw_value="",
                raw_units="",
                source_in_paper=f"Figure vision (unregistered type): {stem}",
                notes=raw_json,
                pipeline="figure_vision",
                method="unknown_plot_type",
                artifact=rel,
                source_id=stem,
                plot_type="unknown",
                page=page,
                axis_or_context=f"declared_plot_type={decl}",
                confidence="",
                supporting_evidence="see Notes for raw JSON; add schema in paper_analysis",
            )
        ]

    return []


def rows_from_text_candidates(path: Path) -> list[dict[str, str]]:
    rel = _rel_path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        batch = TextExtractionBatch.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        return [
            _base_row(
                field_name="(parse_error)",
                raw_value="",
                raw_units="",
                source_in_paper="text LLM batch",
                notes=str(e),
                pipeline="text_llm",
                method="error",
                artifact=rel,
                source_id="candidate_measurements",
                plot_type="",
                page="",
                axis_or_context="",
                confidence="",
                supporting_evidence="",
            )
        ]

    rows: list[dict[str, str]] = []
    for i, c in enumerate(batch.candidates):
        rows.append(
            _base_row(
                field_name=c.field_name,
                raw_value=c.raw_value,
                raw_units=c.raw_units or "",
                source_in_paper=c.source_in_paper,
                notes=f"source_type={c.source_type}",
                pipeline="text_llm",
                method="text_llm_candidate",
                artifact=rel,
                source_id=f"candidate_{i}",
                plot_type="",
                page="",
                axis_or_context="",
                confidence=c.confidence or "",
                supporting_evidence=c.supporting_quote,
            )
        )
    if batch.notes:
        rows.append(
            _base_row(
                field_name="(batch_notes)",
                raw_value="",
                raw_units="",
                source_in_paper="text LLM batch",
                notes=batch.notes,
                pipeline="text_llm",
                method="text_llm_batch_notes",
                artifact=rel,
                source_id="candidate_measurements",
                plot_type="",
                page="",
                axis_or_context="",
                confidence="",
                supporting_evidence="",
            )
        )
    return rows


def rows_from_native_tables(path: Path) -> list[dict[str, str]]:
    rel = _rel_path(path)
    try:
        tables: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [
            _base_row(
                field_name="(parse_error)",
                raw_value="",
                raw_units="",
                source_in_paper="native PDF table",
                notes=str(e),
                pipeline="native_pdf_table",
                method="error",
                artifact=rel,
                source_id="tables.json",
                plot_type="",
                page="",
                axis_or_context="",
                confidence="",
                supporting_evidence="",
            )
        ]

    rows: list[dict[str, str]] = []
    for tab in tables:
        tid = str(tab.get("id", ""))
        page = str(tab.get("page", ""))
        table_rows = tab.get("rows") or []
        header_row: list[str] | None = None
        if table_rows:
            header_row = [str(c) if c is not None else "" for c in table_rows[0]]

        for ri, row_cells in enumerate(table_rows):
            for ci, cell in enumerate(row_cells):
                if cell is None or str(cell).strip() == "":
                    continue
                col_name = ""
                if header_row and ci < len(header_row) and ri > 0:
                    col_name = header_row[ci] or f"col{ci + 1}"
                elif header_row and ci < len(header_row):
                    col_name = f"[header] {header_row[ci] or f'col{ci + 1}'}"
                else:
                    col_name = f"col{ci + 1}"
                bbox = tab.get("bbox_pdf")
                bbox_s = json.dumps(bbox) if bbox is not None else ""
                rows.append(
                    _base_row(
                        field_name=f"{tid} {col_name} (R{ri + 1}C{ci + 1})",
                        raw_value=str(cell),
                        raw_units="",
                        source_in_paper=f"Native PDF table {tid} (page {page})",
                        notes=f"index_on_page={tab.get('index_on_page', '')}; bbox_pdf={bbox_s}",
                        pipeline="native_pdf_table",
                        method="native_table_cell",
                        artifact=rel,
                        source_id=tid,
                        plot_type="",
                        page=page,
                        axis_or_context="",
                        confidence="",
                        supporting_evidence="",
                    )
                )
    return rows


RUN_COLUMNS: list[str] = [
    "Run_ID",
    "Run_description",
    "Paper_ID",
    "experiment_type",
    "temperature",
    "media",
    "culture_format",
    "shaking_speed_rpm",
    "duration_h",
    "replicates_biological",
    "selection_antibiotic",
    "selection_concentration",
    "initial_dilution",
    "species",
    "strain_id",
    "sequence_type",
    "isolation_source",
    "plasmid_name",
    "plasmid_family",
    "plasmid_size_kb",
    "conjugative",
    "resistance_genes",
    "plasmid_accession",
    "measured_outcomes",
    "supporting_evidence",
    "confidence",
]


def rows_from_run_candidates(path: Path) -> list[dict[str, str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        batch = RunExtractionBatch.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        return [{"Run_ID": "(parse_error)", "Run_description": str(e)}]

    rows: list[dict[str, str]] = []
    for c in batch.candidates:
        row: dict[str, str] = {}
        for col in RUN_COLUMNS:
            field = col.lower() if col not in ("Run_ID", "Run_description", "Paper_ID") else {
                "Run_ID": "run_id",
                "Run_description": "run_description",
                "Paper_ID": "paper_id",
            }[col]
            val = getattr(c, field, None)
            row[col] = "" if val is None else str(val)
        rows.append(row)
    return rows


def list_extraction_jsons(dir_path: Path, allowed_ids: set[str] | None) -> list[Path]:
    paths = sorted(dir_path.glob("*.json"))
    if allowed_ids is None:
        return paths
    return [p for p in paths if p.stem in allowed_ids]


def build_export_rows(
    cfg: PocConfig,
    *,
    allowed_figure_ids: set[str] | None = None,
    export_cfg: ExportConfig | None = None,
) -> list[dict[str, str]]:
    ec = export_cfg or effective_export_config(cfg)
    rows: list[dict[str, str]] = []
    ext_dir = cfg.artifacts.extractions_dir
    if ext_dir.is_dir():
        for p in list_extraction_jsons(ext_dir, allowed_figure_ids):
            rows.extend(rows_from_figure_json(p))

    tc = effective_text_config(cfg)
    cand_path = tc.output_dir / "candidate_measurements.json"
    if cand_path.is_file():
        rows.extend(rows_from_text_candidates(cand_path))

    if ec.include_native_pdf_tables:
        tables_path = tc.output_dir / "tables.json"
        if tables_path.is_file():
            rows.extend(rows_from_native_tables(tables_path))

    return rows


def export_meta_dataframe(
    cfg: PocConfig, row_count: int, export_cfg: ExportConfig, *, run_count: int = 0
) -> pd.DataFrame:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items: list[tuple[str, str]] = [
        ("exported_at_utc", now),
        ("pdf_path", _rel_path(cfg.pdf.path)),
        ("extractions_dir", _rel_path(cfg.artifacts.extractions_dir)),
        ("include_native_pdf_tables", str(export_cfg.include_native_pdf_tables)),
        ("measurement_row_count", str(row_count)),
        ("run_row_count", str(run_count)),
    ]
    return pd.DataFrame(items, columns=["Key", "Value"])


def build_run_rows(cfg: PocConfig) -> list[dict[str, str]]:
    tc = effective_text_config(cfg)
    runs_path = tc.output_dir / "candidate_runs.json"
    if runs_path.is_file():
        return rows_from_run_candidates(runs_path)
    return []


def export_workbook(
    cfg: PocConfig,
    *,
    allowed_figure_ids: set[str] | None = None,
    export_cfg: ExportConfig | None = None,
) -> Path:
    ec = export_cfg or effective_export_config(cfg)
    rows = build_export_rows(cfg, allowed_figure_ids=allowed_figure_ids, export_cfg=ec)
    df = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
    for col in MEASUREMENT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[MEASUREMENT_COLUMNS]

    run_rows = build_run_rows(cfg)
    df_runs = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    for col in RUN_COLUMNS:
        if col not in df_runs.columns:
            df_runs[col] = ""
    df_runs = df_runs[RUN_COLUMNS]

    meta = export_meta_dataframe(cfg, len(rows), ec, run_count=len(run_rows))
    out = ec.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Extracted_Measurements", index=False)
        if not df_runs.empty:
            df_runs.to_excel(writer, sheet_name="Extracted_Runs", index=False)
        meta.to_excel(writer, sheet_name="Export_Meta", index=False)
    return out
