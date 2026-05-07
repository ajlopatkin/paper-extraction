from __future__ import annotations

import json
import os
import re
import warnings

import anthropic
from openai import OpenAI
from pydantic import ValidationError

from paper_analysis.combined_prompts import COMBINED_JSON_FIX, COMBINED_SYSTEM, COMBINED_USER
from paper_analysis.combined_schemas import CombinedExtractionBatch, CombinedMeasurementCandidate
from paper_analysis.vision.base import load_json_object_from_model_text

_MAX_TOKENS = 64000
_MAX_CONTINUATION_PASSES = 3

_CONTINUATION_PROMPT = """\
Your previous extraction was truncated because the output was too long.
You already extracted the following measurement IDs:
{extracted_ids}

Your last measurement was {last_id}. CONTINUE extracting from where you left off.
- Start numbering from {next_id}
- Do NOT repeat any measurement above
- Output the SAME JSON format: {{"result_type":"combined_extraction","candidates":[...]}}
- Include all remaining measurements you haven't yet output

--- INPUT ---

{content}"""


def _salvage_truncated_json(text: str) -> CombinedExtractionBatch | None:
    """Try to recover candidates from truncated JSON output.

    When the model hits the token limit, the JSON is cut mid-stream.
    We try to close open arrays/objects and parse whatever candidates
    were completed before truncation.
    """
    json_match = re.search(r"\{", text)
    if not json_match:
        return None
    fragment = text[json_match.start():]

    candidates_match = re.search(r'"candidates"\s*:\s*\[', fragment)
    if not candidates_match:
        return None

    arr_start = candidates_match.end()
    depth = 1
    i = arr_start
    last_complete = arr_start
    while i < len(fragment) and depth > 0:
        ch = fragment[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1:
                last_complete = i + 1
        elif ch == "]" and depth == 1:
            depth = 0
            last_complete = i + 1
        i += 1

    truncated_arr = fragment[arr_start:last_complete]
    if truncated_arr.rstrip().endswith(","):
        truncated_arr = truncated_arr.rstrip().rstrip(",")

    repaired = '{"result_type":"combined_extraction","candidates":[' + truncated_arr + '],"notes":"output was truncated"}'
    try:
        raw = json.loads(repaired)
        batch = CombinedExtractionBatch.model_validate(raw)
        if batch.candidates:
            warnings.warn(
                f"Salvaged {len(batch.candidates)} candidates from truncated output"
            )
            return batch
    except (json.JSONDecodeError, ValidationError):
        pass
    return None


def parse_combined_extraction_batch(text: str) -> CombinedExtractionBatch:
    raw = load_json_object_from_model_text(text)
    return CombinedExtractionBatch.model_validate(raw)


def _collect_anthropic_stream(stream: anthropic.Stream) -> str:
    """Consume a streaming response and return the concatenated text."""
    parts: list[str] = []
    for event in stream:
        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
            parts.append(event.delta.text)
    return "".join(parts)


def _anthropic_call(client: anthropic.Anthropic, model: str, user_msg: str) -> str:
    """Single Anthropic streaming call, returns raw text."""
    with client.messages.stream(
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=0.1,
        system=COMBINED_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        return _collect_anthropic_stream(stream)


def _merge_candidates(
    all_candidates: list[CombinedMeasurementCandidate],
) -> CombinedExtractionBatch:
    """Deduplicate and merge candidates from multiple passes."""
    seen_ids: set[str] = set()
    merged: list[CombinedMeasurementCandidate] = []
    for c in all_candidates:
        if c.measurement_id not in seen_ids:
            seen_ids.add(c.measurement_id)
            merged.append(c)
    return CombinedExtractionBatch(
        result_type="combined_extraction",
        candidates=merged,
        notes=f"Merged from continuation passes ({len(merged)} total rows)",
    )


def _try_continuation_anthropic(
    client: anthropic.Anthropic,
    model: str,
    user_content: str,
    initial_batch: CombinedExtractionBatch,
) -> CombinedExtractionBatch:
    """Make continuation calls until we get a non-truncated response or hit the limit."""
    all_candidates = list(initial_batch.candidates)

    for pass_num in range(_MAX_CONTINUATION_PASSES):
        extracted_ids = [c.measurement_id for c in all_candidates]
        last_id = extracted_ids[-1] if extracted_ids else "MEAS_000"
        last_num = 0
        for eid in extracted_ids:
            try:
                last_num = max(last_num, int(eid.replace("MEAS_", "")))
            except ValueError:
                pass
        next_id = f"MEAS_{last_num + 1:03d}"

        cont_msg = _CONTINUATION_PROMPT.format(
            extracted_ids=", ".join(extracted_ids),
            last_id=last_id,
            next_id=next_id,
            content=user_content,
        )
        warnings.warn(
            f"Continuation pass {pass_num + 1}: {len(all_candidates)} rows so far, "
            f"requesting from {next_id}"
        )
        body = _anthropic_call(client, model, cont_msg)
        try:
            batch = parse_combined_extraction_batch(body)
            all_candidates.extend(batch.candidates)
            return _merge_candidates(all_candidates)
        except (json.JSONDecodeError, ValueError, ValidationError):
            salvaged = _salvage_truncated_json(body)
            if salvaged is not None and salvaged.candidates:
                all_candidates.extend(salvaged.candidates)
            else:
                break

    return _merge_candidates(all_candidates)


def run_combined_analysis_anthropic(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> CombinedExtractionBatch:
    m = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic(timeout=600.0)
    attempts = max_retries + 1
    user_msg = COMBINED_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + COMBINED_JSON_FIX
        body = _anthropic_call(client, m, text)
        try:
            return parse_combined_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            salvaged = _salvage_truncated_json(body)
            if salvaged is not None:
                return _try_continuation_anthropic(client, m, user_content, salvaged)
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"combined analysis failed: {last_err}")


def _openai_call(client: OpenAI, model: str, user_msg: str) -> str:
    """Single OpenAI call, returns raw text."""
    resp = client.chat.completions.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=0.1,
        messages=[
            {"role": "system", "content": COMBINED_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content or ""


def _try_continuation_openai(
    client: OpenAI,
    model: str,
    user_content: str,
    initial_batch: CombinedExtractionBatch,
) -> CombinedExtractionBatch:
    """Make continuation calls for OpenAI until complete or limit reached."""
    all_candidates = list(initial_batch.candidates)

    for pass_num in range(_MAX_CONTINUATION_PASSES):
        extracted_ids = [c.measurement_id for c in all_candidates]
        last_id = extracted_ids[-1] if extracted_ids else "MEAS_000"
        last_num = 0
        for eid in extracted_ids:
            try:
                last_num = max(last_num, int(eid.replace("MEAS_", "")))
            except ValueError:
                pass
        next_id = f"MEAS_{last_num + 1:03d}"

        cont_msg = _CONTINUATION_PROMPT.format(
            extracted_ids=", ".join(extracted_ids),
            last_id=last_id,
            next_id=next_id,
            content=user_content,
        )
        warnings.warn(
            f"Continuation pass {pass_num + 1}: {len(all_candidates)} rows so far, "
            f"requesting from {next_id}"
        )
        body = _openai_call(client, model, cont_msg)
        try:
            batch = parse_combined_extraction_batch(body)
            all_candidates.extend(batch.candidates)
            return _merge_candidates(all_candidates)
        except (json.JSONDecodeError, ValueError, ValidationError):
            salvaged = _salvage_truncated_json(body)
            if salvaged is not None and salvaged.candidates:
                all_candidates.extend(salvaged.candidates)
            else:
                break

    return _merge_candidates(all_candidates)


def run_combined_analysis_openai(
    user_content: str,
    *,
    model: str | None = None,
    max_retries: int = 1,
) -> CombinedExtractionBatch:
    m = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI()
    attempts = max_retries + 1
    user_msg = COMBINED_USER + "\n\n--- INPUT ---\n\n" + user_content
    last_err: Exception | None = None
    for attempt in range(attempts):
        text = user_msg if attempt == 0 else user_msg + COMBINED_JSON_FIX
        body = _openai_call(client, m, text)
        try:
            return parse_combined_extraction_batch(body)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            salvaged = _salvage_truncated_json(body)
            if salvaged is not None:
                return _try_continuation_openai(client, m, user_content, salvaged)
            last_err = e
            if attempt >= attempts - 1:
                raise
    raise RuntimeError(f"combined analysis failed: {last_err}")


def run_combined_analysis(
    user_content: str,
    *,
    provider: str,
    model: str | None = None,
    max_retries: int = 1,
) -> CombinedExtractionBatch:
    prov = provider.strip().lower()
    if prov == "anthropic":
        return run_combined_analysis_anthropic(user_content, model=model, max_retries=max_retries)
    if prov == "openai":
        return run_combined_analysis_openai(user_content, model=model, max_retries=max_retries)
    raise ValueError(f"Unknown combined LLM provider: {provider!r}")


# ---------------------------------------------------------------------------
# Batch API request builders (Anthropic Message Batches with prompt caching)
# ---------------------------------------------------------------------------

_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}


def build_combined_batch_request(
    paper_id: str,
    user_content: str,
    model: str,
) -> dict:
    """Build a single Anthropic batch request dict for combined extraction.

    Uses structured content blocks so that the static system + instruction
    prefix can be cached across all papers in the batch (1h TTL).
    """
    return {
        "custom_id": f"{paper_id}__combined",
        "params": {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0.1,
            "system": [
                {
                    "type": "text",
                    "text": COMBINED_SYSTEM,
                    "cache_control": _CACHE_CONTROL_1H,
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": COMBINED_USER,
                            "cache_control": _CACHE_CONTROL_1H,
                        },
                        {
                            "type": "text",
                            "text": "\n\n--- INPUT ---\n\n" + user_content,
                        },
                    ],
                }
            ],
        },
    }


def build_continuation_batch_request(
    paper_id: str,
    user_content: str,
    model: str,
    extracted_ids: list[str],
    pass_num: int,
) -> dict | None:
    """Build a continuation batch request for a paper whose output was truncated."""
    if not extracted_ids:
        return None
    last_id = extracted_ids[-1]
    last_num = 0
    for eid in extracted_ids:
        try:
            last_num = max(last_num, int(eid.replace("MEAS_", "")))
        except ValueError:
            pass
    next_id = f"MEAS_{last_num + 1:03d}"

    cont_msg = _CONTINUATION_PROMPT.format(
        extracted_ids=", ".join(extracted_ids),
        last_id=last_id,
        next_id=next_id,
        content=user_content,
    )

    return {
        "custom_id": f"{paper_id}__continuation_{pass_num}",
        "params": {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0.1,
            "system": [
                {
                    "type": "text",
                    "text": COMBINED_SYSTEM,
                    "cache_control": _CACHE_CONTROL_1H,
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": cont_msg,
                        },
                    ],
                }
            ],
        },
    }


def parse_batch_result_text(
    text: str,
    stop_reason: str | None = None,
) -> tuple[CombinedExtractionBatch | None, bool]:
    """Parse a batch result into a CombinedExtractionBatch.

    Returns (batch_or_none, needs_continuation).
    needs_continuation is True when the output appears truncated.
    """
    try:
        batch = parse_combined_extraction_batch(text)
        return batch, False
    except (json.JSONDecodeError, ValueError, ValidationError):
        salvaged = _salvage_truncated_json(text)
        if salvaged is not None and salvaged.candidates:
            needs_cont = stop_reason == "max_tokens"
            return salvaged, needs_cont
        return None, False
