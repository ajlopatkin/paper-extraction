from __future__ import annotations

import json
import os
import re

import anthropic
from openai import OpenAI
from pydantic import ValidationError

from paper_analysis.text_prompts import TEXT_JSON_FIX, TEXT_MEASUREMENT_SYSTEM, TEXT_MEASUREMENT_USER
from paper_analysis.text_schemas import TextExtractionBatch


def strip_markdown_json_fence(text: str) -> str:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?", s, re.IGNORECASE)
    if m:
        s = s[m.end() :]
    if s.endswith("```"):
        s = s[: -3].strip()
    return s.strip()


def parse_text_extraction_batch(text: str) -> TextExtractionBatch:
    raw = json.loads(strip_markdown_json_fence(text))
    return TextExtractionBatch.model_validate(raw)


def run_text_analysis_anthropic(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> TextExtractionBatch:
    m = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic()
    attempts = max_retries + 1
    user_msg = TEXT_MEASUREMENT_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + TEXT_JSON_FIX
        resp = client.messages.create(
            model=m,
            max_tokens=16384,
            system=TEXT_MEASUREMENT_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        blocks = [b for b in resp.content if b.type == "text"]
        body = blocks[0].text if blocks else ""
        try:
            return parse_text_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"text analysis failed: {last_err}")


def run_text_analysis_openai(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> TextExtractionBatch:
    m = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI()
    attempts = max_retries + 1
    user_msg = TEXT_MEASUREMENT_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + TEXT_JSON_FIX
        resp = client.chat.completions.create(
            model=m,
            max_tokens=16384,
            messages=[
                {"role": "system", "content": TEXT_MEASUREMENT_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        body = resp.choices[0].message.content or ""
        try:
            return parse_text_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"text analysis failed: {last_err}")


def run_text_analysis(
    user_content: str,
    *,
    provider: str,
    model: str | None = None,
    max_retries: int = 1,
) -> TextExtractionBatch:
    prov = provider.strip().lower()
    if prov == "anthropic":
        return run_text_analysis_anthropic(user_content, model=model, max_retries=max_retries)
    if prov == "openai":
        return run_text_analysis_openai(user_content, model=model, max_retries=max_retries)
    raise ValueError(f"Unknown text LLM provider: {provider!r}")
