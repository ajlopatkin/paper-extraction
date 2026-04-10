from __future__ import annotations

import json
import os
import re
from typing import Literal, Protocol

from paper_analysis.prompts import (
    BOX_PLOT_SYSTEM,
    BOX_PLOT_USER,
    LINE_CHART_SYSTEM,
    LINE_CHART_USER,
    PLASMID_MAP_SYSTEM,
    PLASMID_MAP_USER,
    TABLE_IMAGE_SYSTEM,
    TABLE_IMAGE_USER,
    EXPERIMENTAL_WORKFLOW_SYSTEM,
    EXPERIMENTAL_WORKFLOW_USER,
    WORKFLOW_DIAGRAM_SYSTEM,
    WORKFLOW_DIAGRAM_USER,
)
from paper_analysis.plot_type_dispatch import ExtractionModel, parse_extraction_dict

PlotType = Literal[
    "box_plot",
    "line_chart",
    "line_plot",
    "table_image",
    "plasmid_map",
    "workflow_diagram",
    "experimental_workflow",
]


def strip_markdown_json_fence(text: str) -> str:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?", s, re.IGNORECASE)
    if m:
        s = s[m.end() :]
    if s.endswith("```"):
        s = s[: -3].strip()
    return s.strip()


def load_json_object_from_model_text(text: str) -> dict:
    """Parse the first JSON object from model output (handles preamble, fences, multi-block joins)."""
    s = strip_markdown_json_fence(text)
    if not s:
        raise ValueError("Model returned empty text; no JSON to parse.")
    dec = json.JSONDecoder()
    s2 = s.strip()
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for i, c in enumerate(s2):
        if c != "{":
            continue
        try:
            obj, _end = dec.raw_decode(s2, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    preview = s2[:500].replace("\n", "\\n")
    raise ValueError(f"Could not parse a JSON object from model response. Preview: {preview!r}")


def parse_extraction_text(text: str) -> ExtractionModel:
    raw = load_json_object_from_model_text(text)
    return parse_extraction_dict(raw, context="vision model response")


class VisionClient(Protocol):
    def extract_figure(
        self,
        image_png: bytes,
        plot_type: PlotType,
        *,
        max_retries: int = 1,
    ) -> ExtractionModel: ...


def _prompts_for(plot_type: PlotType) -> tuple[str, str]:
    if plot_type == "box_plot":
        return BOX_PLOT_SYSTEM, BOX_PLOT_USER
    if plot_type in ("line_chart", "line_plot"):
        return LINE_CHART_SYSTEM, LINE_CHART_USER
    if plot_type == "plasmid_map":
        return PLASMID_MAP_SYSTEM, PLASMID_MAP_USER
    if plot_type == "workflow_diagram":
        return WORKFLOW_DIAGRAM_SYSTEM, WORKFLOW_DIAGRAM_USER
    if plot_type == "experimental_workflow":
        return EXPERIMENTAL_WORKFLOW_SYSTEM, EXPERIMENTAL_WORKFLOW_USER
    return TABLE_IMAGE_SYSTEM, TABLE_IMAGE_USER


def build_vision_client(
    provider: Literal["anthropic", "openai"] | str | None = None,
    model: str | None = None,
):
    prov = (provider or os.environ.get("PAPER_ANALYSIS_VISION_PROVIDER") or "anthropic").strip().lower()
    if prov == "anthropic":
        from paper_analysis.vision.anthropic_client import AnthropicVisionClient

        return AnthropicVisionClient(model=model)
    if prov == "openai":
        from paper_analysis.vision.openai_client import OpenAIVisionClient

        return OpenAIVisionClient(model=model)
    raise ValueError(f"Unknown vision provider: {prov!r}")
