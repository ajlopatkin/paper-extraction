from __future__ import annotations

TEXT_MEASUREMENT_SYSTEM = """You extract experimental measurements and numeric findings from scientific paper text and from tables represented as markdown or TSV. Reply with a single JSON object only — no markdown fences, no commentary. Prefer precision over recall: only include values that are clearly reported as data (not citations, not figure-only references unless the number is written in text)."""

TEXT_MEASUREMENT_USER = """You are given (1) native PDF tables serialized as markdown, then (2) optional body text from the same paper.

Task: list **candidate measurements** that could populate a spreadsheet with columns like Field_name, Raw_value, Raw_units, Source_in_paper, Notes.

Rules:
- For each candidate, set source_type to "table" if it comes from the table section below; otherwise "body_text".
- source_in_paper should name the table ("Table 2") or approximate location ("page 5", "Results").
- supporting_quote must be a short verbatim fragment.
- Use raw_value as a string preserving symbols like ~ or inequalities if present.
- Do not invent values. Skip values that appear only in figures unless explicitly repeated in text/tables here.

Return exactly this JSON shape:
{{
  "result_type": "text_measurement_candidates",
  "candidates": [
    {{
      "field_name": string,
      "raw_value": string,
      "raw_units": string or null,
      "source_in_paper": string,
      "source_type": "table" or "body_text",
      "supporting_quote": string,
      "confidence": string or null
    }}
  ],
  "notes": string or null
}}"""

TEXT_JSON_FIX = "\n\nYour previous reply was not valid JSON for the schema. Return only one corrected JSON object."
