from __future__ import annotations

import base64
import json
import os
import anthropic
from pydantic import ValidationError

from paper_analysis.prompts import JSON_FIX_SUFFIX
from paper_analysis.vision.base import PlotType, _prompts_for, parse_extraction_text


class AnthropicVisionClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self._client = anthropic.Anthropic()

    def extract_figure(
        self,
        image_png: bytes,
        plot_type: PlotType,
        *,
        max_retries: int = 1,
    ):
        system, user_text = _prompts_for(plot_type)
        b64 = base64.standard_b64encode(image_png).decode("ascii")
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        }
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
                max_tokens=8192,
                system=system,
                messages=messages,
            )
            blocks = [b for b in resp.content if b.type == "text"]
            # Join all text blocks (some models/API versions split prose + JSON, or omit first block text).
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
