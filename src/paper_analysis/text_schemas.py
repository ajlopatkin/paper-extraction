from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TextMeasurementCandidate(BaseModel):
    """One numeric or structured finding from paper text / native tables (for ground-truth alignment)."""

    field_name: str = Field(..., description="Measurement or quantity name (spreadsheet Field_name analog)")
    raw_value: str = Field(..., description="Value as written, e.g. 130, ~4.05, 1.2e6")
    raw_units: str | None = None
    source_in_paper: str = Field(
        ...,
        description='Where it appears, e.g. "Table 2", "page 5", "Results paragraph"',
    )
    source_type: Literal["table", "body_text"] = Field(
        ...,
        description="Whether taken from a detected text table vs prose",
    )
    supporting_quote: str = Field(
        ...,
        description="Short verbatim snippet (cell row or sentence) evidencing the extraction",
    )
    confidence: str | None = Field(
        default=None,
        description='Optional: "high" / "medium" / "low" or brief caveat',
    )


class TextExtractionBatch(BaseModel):
    """LLM output wrapper for validation and versioning."""

    result_type: Literal["text_measurement_candidates"] = "text_measurement_candidates"
    candidates: list[TextMeasurementCandidate] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Overall caveats (truncation, ambiguous tables, etc.)",
    )
