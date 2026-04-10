from __future__ import annotations

import json
import os

import anthropic
from openai import OpenAI
from pydantic import ValidationError

from paper_analysis.runs_prompts import RUNS_JSON_FIX, RUNS_METADATA_SYSTEM, RUNS_METADATA_USER
from paper_analysis.runs_schemas import RunExtractionBatch
from paper_analysis.text_analyze_llm import strip_markdown_json_fence


def parse_run_extraction_batch(text: str) -> RunExtractionBatch:
    raw = json.loads(strip_markdown_json_fence(text))
    return RunExtractionBatch.model_validate(raw)


def run_runs_analysis_anthropic(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> RunExtractionBatch:
    m = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic()
    attempts = max_retries + 1
    user_msg = RUNS_METADATA_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + RUNS_JSON_FIX
        resp = client.messages.create(
            model=m,
            max_tokens=16384,
            system=RUNS_METADATA_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        blocks = [b for b in resp.content if b.type == "text"]
        body = blocks[0].text if blocks else ""
        try:
            return parse_run_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"runs analysis failed: {last_err}")


def run_runs_analysis_openai(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> RunExtractionBatch:
    m = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI()
    attempts = max_retries + 1
    user_msg = RUNS_METADATA_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + RUNS_JSON_FIX
        resp = client.chat.completions.create(
            model=m,
            max_tokens=16384,
            messages=[
                {"role": "system", "content": RUNS_METADATA_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        body = resp.choices[0].message.content or ""
        try:
            return parse_run_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"runs analysis failed: {last_err}")


def run_runs_analysis(
    user_content: str,
    *,
    provider: str,
    model: str | None = None,
    max_retries: int = 1,
) -> RunExtractionBatch:
    prov = provider.strip().lower()
    if prov == "anthropic":
        return run_runs_analysis_anthropic(user_content, model=model, max_retries=max_retries)
    if prov == "openai":
        return run_runs_analysis_openai(user_content, model=model, max_retries=max_retries)
    raise ValueError(f"Unknown runs LLM provider: {provider!r}")
