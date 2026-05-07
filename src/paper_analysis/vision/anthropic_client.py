from __future__ import annotations

import base64
import json
import os
from typing import Any

import anthropic
from pydantic import ValidationError

from paper_analysis.prompts import JSON_FIX_SUFFIX
from paper_analysis.vision.base import PlotType, _prompts_for, classify_prompts, parse_extraction_text

_VALID_PLOT_TYPES = frozenset({
    "box_plot", "line_chart", "line_plot", "heatmap", "table_image",
    "plasmid_map", "workflow_diagram", "experimental_workflow",
})


class AnthropicVisionClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self._client = anthropic.Anthropic()

    def _classify_figure(self, image_block: dict) -> str:
        """Ask the model to classify the figure type before extraction."""
        system, user_text = classify_prompts()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=64,
            system=system,
            messages=[{
                "role": "user",
                "content": [image_block, {"type": "text", "text": user_text}],
            }],
        )
        raw = "".join(
            (getattr(b, "text", None) or "").strip()
            for b in resp.content if b.type == "text"
        ).strip().lower().replace(" ", "_")
        if raw in _VALID_PLOT_TYPES:
            return raw
        for valid in _VALID_PLOT_TYPES:
            if valid in raw:
                return valid
        return "table_image"

    def extract_figure(
        self,
        image_png: bytes,
        plot_type: PlotType,
        *,
        max_retries: int = 1,
    ):
        b64 = base64.standard_b64encode(image_png).decode("ascii")
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        }
        if plot_type == "auto":
            plot_type = self._classify_figure(image_block)

        system, user_text = _prompts_for(plot_type)
        attempts = max_retries + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            text = user_text if attempt == 0 else user_text + JSON_FIX_SUFFIX
            messages = [
                {
                    "role": "user",
                    "content": [
                        image_block,
                        {"type": "text", "text": text},
                    ],
                }
            ]
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=16384,
                system=system,
                messages=messages,
            )
            blocks = [b for b in resp.content if b.type == "text"]
            last_text = "\n".join(
                (getattr(b, "text", None) or "").strip() for b in blocks
            ).strip()
            try:
                return parse_extraction_text(last_text)
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                last_err = e
                if attempt >= attempts - 1:
                    raise
        raise RuntimeError(f"extraction failed: {last_err}")


# ---------------------------------------------------------------------------
# Batch API request builders
# ---------------------------------------------------------------------------

def _make_image_block(image_b64: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
    }


def build_classify_batch_request(
    custom_id: str,
    image_b64: str,
    model: str,
) -> dict[str, Any]:
    """Build a batch request dict for figure classification."""
    system, user_text = classify_prompts()
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 64,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _make_image_block(image_b64),
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        },
    }


def build_extract_batch_request(
    custom_id: str,
    image_b64: str,
    plot_type: str,
    model: str,
) -> dict[str, Any]:
    """Build a batch request dict for figure data extraction."""
    system, user_text = _prompts_for(plot_type)
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 16384,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _make_image_block(image_b64),
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        },
    }


def parse_classify_result(text: str) -> str:
    """Parse a classification batch result into a plot_type string."""
    raw = text.strip().lower().replace(" ", "_")
    if raw in _VALID_PLOT_TYPES:
        return raw
    for valid in _VALID_PLOT_TYPES:
        if valid in raw:
            return valid
    return "table_image"
