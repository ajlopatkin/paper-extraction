from __future__ import annotations

import warnings
from typing import Any

from paper_analysis.schemas import (
    BoxPlotExtraction,
    ExperimentalWorkflowExtraction,
    LineChartExtraction,
    PlasmidMapExtraction,
    TableImageExtraction,
    UnknownPlotExtraction,
    WorkflowDiagramExtraction,
)

ExtractionModel = (
    BoxPlotExtraction
    | LineChartExtraction
    | TableImageExtraction
    | PlasmidMapExtraction
    | WorkflowDiagramExtraction
    | ExperimentalWorkflowExtraction
    | UnknownPlotExtraction
)


def warn_unknown_plot_type(declared: object, *, context: str) -> None:
    label = repr(declared) if declared is not None else "missing"
    warnings.warn(
        f"Unknown plot_type {label} ({context}). "
        "Add a schema in paper_analysis/schemas.py, prompts in prompts.py, "
        "and register the type in plot_type_dispatch.parse_extraction_dict.",
        UserWarning,
        stacklevel=3,
    )


def _coerce_unknown(raw: dict[str, Any], declared: object) -> UnknownPlotExtraction:
    dt = declared if isinstance(declared, str) else None
    return UnknownPlotExtraction(declared_plot_type=dt, raw=dict(raw))


def parse_extraction_dict(data: dict[str, Any], *, context: str) -> ExtractionModel:
    """Route parsed JSON to the correct Pydantic model; unknown types become UnknownPlotExtraction with a warning."""
    pt = data.get("plot_type")
    if pt == "unknown":
        return UnknownPlotExtraction.model_validate(data)
    if pt == "box_plot":
        return BoxPlotExtraction.model_validate(data)
    if pt in ("line_chart", "line_plot"):
        return LineChartExtraction.model_validate(data)
    if pt == "table_image":
        return TableImageExtraction.model_validate(data)
    if pt == "plasmid_map":
        return PlasmidMapExtraction.model_validate(data)
    if pt == "workflow_diagram":
        return WorkflowDiagramExtraction.model_validate(data)
    if pt == "experimental_workflow":
        return ExperimentalWorkflowExtraction.model_validate(data)
    warn_unknown_plot_type(pt, context=context)
    return _coerce_unknown(data, pt)
