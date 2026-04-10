from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class BoxPlotGroup(BaseModel):
    label: str
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    whisker_low: float | None = None
    whisker_high: float | None = None
    significance: str | None = None


class BoxPlotExtraction(BaseModel):
    plot_type: Literal["box_plot"] = "box_plot"
    axis_x_label: str | None = None
    axis_y_label: str | None = None
    axis_y_units: str | None = None
    groups: list[BoxPlotGroup] = Field(default_factory=list)


class LinePoint(BaseModel):
    x: float | str
    y: float
    error_low: float | None = None
    error_high: float | None = None


class LineSeries(BaseModel):
    name: str | None = None
    points: list[LinePoint] = Field(default_factory=list)


class LineChartExtraction(BaseModel):
    """Line or scatter panels; models sometimes emit plot_type line_plot instead of line_chart."""

    plot_type: Literal["line_chart", "line_plot"] = "line_chart"
    axis_x_label: str | None = None
    axis_y_label: str | None = None
    axis_x_units: str | None = None
    axis_y_units: str | None = None
    series: list[LineSeries] = Field(default_factory=list)


class PlasmidFeature(BaseModel):
    """One annotated element on a plasmid / cloning vector diagram."""

    label: str
    feature_type: str | None = Field(
        default=None,
        description="e.g. gene, promoter, ori, resistance_marker, MCS, primer_site",
    )
    notes: str | None = None


class PlasmidMapExtraction(BaseModel):
    """Circular or linear plasmid / vector map (not a numeric plot)."""

    plot_type: Literal["plasmid_map"] = "plasmid_map"
    title_or_caption: str | None = None
    map_name_or_identifier: str | None = None
    is_circular: bool | None = None
    features: list[PlasmidFeature] = Field(default_factory=list)
    other_visible_labels: list[str] = Field(
        default_factory=list,
        description="Readable text on the map not assigned to a feature",
    )
    notes: str | None = None

    @field_validator("other_visible_labels", mode="before")
    @classmethod
    def _stringify_other_labels(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return ["" if x is None else str(x) for x in v]


class WorkflowNode(BaseModel):
    """One box or shape in a flowchart / pipeline diagram."""

    label: str
    node_type: str | None = Field(
        default=None,
        description="e.g. start_end, process, decision, data, document, subprocess",
    )
    notes: str | None = None


class WorkflowEdge(BaseModel):
    """Directed link between nodes (arrow, flow line)."""

    model_config = ConfigDict(populate_by_name=True)

    from_label: str = Field(validation_alias=AliasChoices("from_label", "from"))
    to_label: str = Field(validation_alias=AliasChoices("to_label", "to"))
    edge_label: str | None = Field(
        default=None,
        description="Text on the arrow (e.g. Yes/No, condition)",
        validation_alias=AliasChoices("edge_label", "label"),
    )


class WorkflowDiagramExtraction(BaseModel):
    """Workflow, flowchart, or pipeline schematic (not a data chart)."""

    plot_type: Literal["workflow_diagram"] = "workflow_diagram"
    title_or_caption: str | None = None
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    other_visible_labels: list[str] = Field(
        default_factory=list,
        description="Legible text not assigned to a node or edge",
    )
    notes: str | None = None

    @field_validator("other_visible_labels", mode="before")
    @classmethod
    def _stringify_workflow_labels(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return ["" if x is None else str(x) for x in v]


class ExperimentalWorkflowExtraction(BaseModel):
    """Lab / experimental protocol shown as a flow or pipeline figure (same JSON shape as workflow_diagram)."""

    plot_type: Literal["experimental_workflow"] = "experimental_workflow"
    title_or_caption: str | None = None
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    other_visible_labels: list[str] = Field(
        default_factory=list,
        description="Legible text not assigned to a node or edge",
    )
    notes: str | None = None

    @field_validator("other_visible_labels", mode="before")
    @classmethod
    def _stringify_exp_workflow_labels(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return ["" if x is None else str(x) for x in v]


class UnknownPlotExtraction(BaseModel):
    """Fallback when plot_type is missing or not registered; preserves full JSON for later first-class support."""

    plot_type: Literal["unknown"] = "unknown"
    declared_plot_type: str | None = Field(
        default=None,
        description="Original plot_type string from the model or file, if any",
    )
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete parsed object before coercion",
    )


class TableImageExtraction(BaseModel):
    """Structured table read from a figure panel that is a table-as-image (not native PDF text)."""

    plot_type: Literal["table_image"] = "table_image"
    title_or_caption: str | None = None
    column_headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Merged cells, cut-off text, or other caveats",
    )

    @field_validator("column_headers", mode="before")
    @classmethod
    def _stringify_headers(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return ["" if x is None else str(x) for x in v]

    @field_validator("rows", mode="before")
    @classmethod
    def _stringify_rows(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        out: list[list[str]] = []
        for row in v:
            if not isinstance(row, list):
                continue
            out.append(["" if c is None else str(c) for c in row])
        return out
