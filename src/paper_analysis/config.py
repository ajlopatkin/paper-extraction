from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PdfConfig(BaseModel):
    path: Path = Path("data/paper.pdf")


class ArtifactsConfig(BaseModel):
    figures_dir: Path = Path("artifacts/figures")
    extractions_dir: Path = Path("artifacts/extractions")


class FigureTarget(BaseModel):
    id: str
    page: int = Field(ge=1, description="1-based page index")
    plot_type: Literal[
        "box_plot",
        "line_chart",
        "line_plot",
        "table_image",
        "plasmid_map",
        "workflow_diagram",
        "experimental_workflow",
    ]
    bbox_pdf: list[float] | None = Field(
        default=None,
        description="x0,y0,x1,y1 in PDF points (page coordinates)",
    )
    bbox_norm: list[float] | None = Field(
        default=None,
        description="x0,y0,x1,y1 as fractions of page width/height (0-1)",
    )
    render_dpi: int = Field(default=200, ge=72, le=600)
    skip: bool = Field(
        default=False,
        description="If true, omit from extract-figures and run-vision (decorative / non-data crops).",
    )


class ComparisonRule(BaseModel):
    figure_id: str
    source_contains: str
    label_column: str | None = Field(
        default=None,
        description="Column to match box-plot group labels (e.g. Field_name)",
    )
    value_column: str
    strip_tilde: bool = True


class EvaluationConfig(BaseModel):
    spreadsheet_path: Path = Path("data/ground_truth.xlsx")
    sheet: str = "Measurements"
    source_column: str = "Source_in_paper"
    comparisons: list[ComparisonRule] = Field(default_factory=list)
    relative_tolerance: float = Field(default=0.10, ge=0.0)
    absolute_tolerance: float = Field(default=0.01, ge=0.0)


class VisionConfig(BaseModel):
    provider: Literal["anthropic", "openai"] | None = None
    model: str | None = None


class ExportConfig(BaseModel):
    """Write a Measurements-style spreadsheet from vision JSON, text candidates, and optional native tables."""

    output_path: Path = Path("artifacts/export/extractions.xlsx")
    include_native_pdf_tables: bool = Field(
        default=True,
        description="Add one row per non-empty cell from artifacts/text/tables.json (find_tables).",
    )


class TextPipelineConfig(BaseModel):
    """Text + native-table extraction; optional LLM pass for measurement candidates."""

    output_dir: Path = Path("artifacts/text")
    pages: list[int] | None = Field(
        default=None,
        description="1-based page numbers to include; null = entire document",
    )
    max_context_chars: int = Field(
        default=120000,
        ge=8000,
        le=500000,
        description="Max characters of body text sent to the LLM (head+tail if exceeded)",
    )
    llm_model: str | None = Field(
        default=None,
        description="Override text LLM model; defaults to vision.model or env ANTHROPIC_MODEL / OPENAI_MODEL",
    )


class AutoDiscoverConfig(BaseModel):
    """Derive figure crops from embedded rasters, vector blocks, and clustered path geometry."""

    enabled: bool = False
    strategy: Literal["only_auto", "append_to_manual"] = Field(
        default="only_auto",
        description="only_auto: ignore manual figures list; append_to_manual: manual first, then auto",
    )
    default_plot_type: Literal[
        "box_plot",
        "line_chart",
        "line_plot",
        "table_image",
        "plasmid_map",
        "workflow_diagram",
        "experimental_workflow",
    ] = Field(
        default="box_plot",
        description="Vision prompt type for every discovered region (override per-id for tables, plasmid maps, etc.)",
    )
    pages: list[int] | None = Field(
        default=None,
        description="1-based page numbers to scan; null = all pages",
    )
    min_area_frac: float = Field(
        default=0.008,
        ge=0.0,
        le=1.0,
        description="Min bbox area as a fraction of full page area",
    )
    min_width_frac: float = Field(default=0.03, ge=0.0, le=1.0)
    min_height_frac: float = Field(default=0.03, ge=0.0, le=1.0)
    duplicate_merge_iou: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="First pass: merge only near-duplicate bboxes (dict vs xref vs blocks)",
    )
    merge_fragments: bool = Field(
        default=False,
        description=(
            "Second pass: merge rects with merge_iou_threshold (set True for multi-tile raster figures; "
            "False keeps separate panels that only touch or slightly overlap)"
        ),
    )
    merge_iou_threshold: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
        description="Second pass IoU threshold; only used when merge_fragments is true",
    )
    padding_frac: float = Field(
        default=0.01,
        ge=0.0,
        le=0.2,
        description="Expand each bbox by this fraction of min(page width, height)",
    )
    max_figures_per_page: int | None = Field(
        default=None,
        ge=1,
        description="Keep only the N largest regions per page after filtering",
    )
    include_vector_graphics: bool = Field(
        default=True,
        description=(
            "Include vector plots (box/line charts drawn as PDF paths). "
            "Raster-only discovery misses these; journal figures are often vector."
        ),
    )
    drawing_cluster_gap_frac: float = Field(
        default=0.012,
        ge=0.0,
        le=0.15,
        description="Merge drawing/path bboxes within this gap (fraction of min(page width,height))",
    )
    min_drawing_primitive_area_frac: float = Field(
        default=0.000002,
        ge=0.0,
        le=1.0,
        description="Drop tiny path bboxes before clustering (noise / hairlines)",
    )
    max_drawing_primitives: int = Field(
        default=20000,
        ge=100,
        description="Cap path rectangles per page (safety on pathological PDFs)",
    )
    id_template: str = Field(
        default="p{page:03d}_fig{index:02d}",
        description="Python format with {page} (1-based) and {index} (1-based on that page)",
    )
    render_dpi: int = Field(default=200, ge=72, le=600)


class PocConfig(BaseModel):
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    figures: list[FigureTarget] = Field(default_factory=list)
    exclude_figure_ids: list[str] = Field(
        default_factory=list,
        description="Figure ids to omit from extract-figures / run-vision (applies with poc figures or --figures-config).",
    )
    auto_discover: AutoDiscoverConfig | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    text: TextPipelineConfig | None = None
    export: ExportConfig | None = None

    @classmethod
    def load(cls, path: Path) -> PocConfig:
        path = path.resolve()
        project_root = path.parent.parent if path.parent.name == "config" else path.parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(_resolve_paths(raw, project_root))


_PATH_KEYS = frozenset(
    {"path", "spreadsheet_path", "figures_dir", "extractions_dir", "output_dir", "output_path"}
)


def _resolve_paths(obj: Any, project_root: Path) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _PATH_KEYS and isinstance(v, str):
                out[k] = _resolve_path_str(v, project_root)
            else:
                out[k] = _resolve_paths(v, project_root)
        return out
    if isinstance(obj, list):
        return [_resolve_paths(v, project_root) for v in obj]
    return obj


def _resolve_path_str(s: str, project_root: Path) -> str:
    p = Path(s)
    if p.is_absolute():
        return str(p.resolve())
    return str((project_root / p).resolve())


class FiguresListRoot(BaseModel):
    """Standalone YAML: `figures:` list (e.g. discover-bboxes default file) plus optional exclusions."""

    figures: list[FigureTarget]
    exclude_figure_ids: list[str] = Field(
        default_factory=list,
        description="Ids to drop without editing each figure entry (merged with poc exclude_figure_ids).",
    )


def load_figures_yaml_root(path: Path) -> FiguresListRoot:
    """Load a standalone YAML with top-level `figures:` (and optional `exclude_figure_ids:`)."""
    path = path.resolve()
    project_root = path.parent.parent if path.parent.name == "config" else path.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "figures" not in raw:
        raise ValueError(
            f"Expected YAML with top-level 'figures:' list in {path} "
            "(e.g. output of: paper-analysis discover-bboxes ...)."
        )
    resolved = _resolve_paths(raw, project_root)
    return FiguresListRoot.model_validate(resolved)


def load_figure_targets_yaml(path: Path) -> list[FigureTarget]:
    """Load figures from a standalone YAML (no exclusion filtering; prefer resolve_figure_targets)."""
    return load_figures_yaml_root(path).figures


def apply_figure_exclusions(
    targets: list[FigureTarget],
    cfg: PocConfig,
    *,
    extra_exclude_ids: Collection[str] | None = None,
) -> list[FigureTarget]:
    """Drop skipped figures and ids in poc.exclude_figure_ids plus optional file-level excludes."""
    ban = set(cfg.exclude_figure_ids)
    if extra_exclude_ids:
        ban.update(extra_exclude_ids)
    out: list[FigureTarget] = []
    for t in targets:
        if t.skip:
            continue
        if t.id in ban:
            continue
        out.append(t)
    return out


def resolve_figure_targets(
    cfg: PocConfig,
    *,
    figures_config: Path | None = None,
) -> list[FigureTarget]:
    """Effective targets: optional figures file wins over poc `figures` / auto_discover; exclusions applied."""
    if figures_config is not None:
        root = load_figures_yaml_root(figures_config)
        targets = root.figures
        extra = root.exclude_figure_ids
    else:
        targets = _effective_figure_targets_from_poc(cfg)
        extra = []
    return apply_figure_exclusions(targets, cfg, extra_exclude_ids=extra)


def _effective_figure_targets_from_poc(cfg: PocConfig) -> list[FigureTarget]:
    from paper_analysis.pdf_discovery import get_effective_figure_targets

    return get_effective_figure_targets(cfg)


def effective_text_config(cfg: PocConfig) -> TextPipelineConfig:
    """Text pipeline settings; use defaults when `text:` is absent in YAML."""
    return cfg.text if cfg.text is not None else TextPipelineConfig()


def effective_export_config(cfg: PocConfig) -> ExportConfig:
    """Export settings; use defaults when `export:` is absent in YAML."""
    return cfg.export if cfg.export is not None else ExportConfig()
