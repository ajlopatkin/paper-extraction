from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunMetadataCandidate(BaseModel):
    """One experimental run / condition extracted from the paper text."""

    run_id: str = Field(
        ...,
        description="Sequential identifier, e.g. RUN_001. Assign in order of appearance.",
    )
    run_description: str = Field(
        ...,
        description=(
            "Short human-readable summary of the condition, "
            "e.g. 'Initial fitness cost: MW2 + pMW2 (control)'"
        ),
    )
    paper_id: str = Field(
        ...,
        description=(
            "Identifier derived from first author surname and year, "
            "e.g. 'Smith_2023'. Must be identical for every run."
        ),
    )
    experiment_type: str = Field(
        ...,
        description=(
            "Category: growth_curve_assay, evolution_experiment, "
            "plasmid_curing_assay, conjugation_assay, MIC_assay, etc."
        ),
    )
    temperature: Any = Field(
        default=None,
        description="Incubation temperature in °C (number or null)",
    )
    media: str | None = Field(
        default=None,
        description="Growth medium, e.g. TSB, LB, MHB. Include supplements if stated.",
    )
    culture_format: str | None = Field(
        default=None,
        description=(
            "Vessel / format, e.g. '96-well microplate', "
            "'liquid culture tube', 'agar plate'"
        ),
    )
    shaking_speed_rpm: Any = Field(
        default=None,
        description="Shaking speed in rpm, or null / '—' if static or not stated",
    )
    duration_h: Any = Field(
        default=None,
        description="Total duration in hours (number, string like '24 + 336', or null)",
    )
    replicates_biological: int | None = Field(
        default=None,
        description="Number of biological replicates",
    )
    selection_antibiotic: str | None = Field(
        default=None,
        description="Antibiotic used for selection, or '—' if none",
    )
    selection_concentration: str | None = Field(
        default=None,
        description="Concentration of selection antibiotic, e.g. '5 µg/mL', or '—'",
    )
    initial_dilution: str | None = Field(
        default=None,
        description="Starting dilution factor, or '—' if not applicable",
    )
    species: str | None = Field(
        default=None,
        description="Organism species, e.g. 'Staphylococcus aureus', 'Escherichia coli'",
    )
    strain_id: str | None = Field(
        default=None,
        description="Strain name / designation, e.g. 'MW2 PFt0', 'RN4220 (lab strain)'",
    )
    sequence_type: str | None = Field(
        default=None,
        description="MLST or clonal complex, e.g. 'CC1/USA400', 'ST398'",
    )
    isolation_source: str | None = Field(
        default=None,
        description="Origin category, e.g. 'clinical_CA-MRSA', 'livestock_clinical', 'lab_strain'",
    )
    plasmid_name: str | None = Field(
        default=None,
        description="Plasmid or construct name, e.g. 'pMW2', 'pUR2940'",
    )
    plasmid_family: str | None = Field(
        default=None,
        description="Plasmid family / replicon type, e.g. 'pSK-family', 'ColE1'",
    )
    plasmid_size_kb: Any = Field(
        default=None,
        description="Plasmid size in kilobases (number, '~2.0', or null)",
    )
    conjugative: str | None = Field(
        default=None,
        description="Transfer capability: 'conjugative', 'mobilizable', 'non-mobilizable', etc.",
    )
    resistance_genes: str | None = Field(
        default=None,
        description="Semicolon-separated resistance / marker genes, e.g. 'cadDX; blaIRZ; arsBC'",
    )
    plasmid_accession: str | None = Field(
        default=None,
        description="GenBank / NCBI accession, e.g. 'NC_005011.1'",
    )
    measured_outcomes: str | None = Field(
        default=None,
        description=(
            "Semicolon-separated outcomes measured for this run, "
            "e.g. 'AUC_OD595; lag_phase_duration'"
        ),
    )
    supporting_evidence: str | None = Field(
        default=None,
        description=(
            "Brief note on where in the paper this run is described "
            "(section, table, figure reference)"
        ),
    )
    confidence: str | None = Field(
        default=None,
        description="'high' / 'medium' / 'low' or brief caveat",
    )


class RunExtractionBatch(BaseModel):
    """LLM output wrapper for run metadata extraction."""

    result_type: Literal["run_metadata_candidates"] = "run_metadata_candidates"
    candidates: list[RunMetadataCandidate] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Overall caveats (ambiguous conditions, missing details, etc.)",
    )
