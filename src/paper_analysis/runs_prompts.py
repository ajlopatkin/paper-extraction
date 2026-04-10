from __future__ import annotations

RUNS_METADATA_SYSTEM = """You extract structured experimental metadata from scientific papers. Each distinct experimental condition, treatment arm, or assay setup should become one "run". Reply with a single JSON object only — no markdown fences, no commentary. Use null for unknown values and "—" for explicitly not-applicable values."""

RUNS_METADATA_USER = """You are given text and tables from a scientific paper. Your task is to identify every distinct experimental run or condition and extract structured metadata for each.

A "run" is a unique combination of experiment type + strain/organism + treatment/plasmid/condition. For example:
- Each strain × plasmid combination tested in a fitness assay is a separate run.
- Each arm of an evolution experiment is a separate run.
- Each target tested in a curing/knockout assay is a separate run.
- Control conditions are runs too (mark them clearly in run_description).

Rules:
1. Assign sequential run_id values: RUN_001, RUN_002, etc., in order of appearance in the paper.
2. For paper_id, derive from the first/corresponding author surname and publication year (e.g. "Dorado-Morales_2021"). Use the SAME paper_id for every run.
3. Fill in as many fields as the paper supports. Use null for genuinely unknown values; use "—" for values that are explicitly not applicable (e.g. no antibiotic selection).
4. For resistance_genes and measured_outcomes, use semicolon-separated lists.
5. For plasmid_size_kb, use a number when known, a string like "~2.0" when approximate, or null.
6. For shaking_speed_rpm, duration_h, and temperature, prefer numbers when possible.
7. Be thorough: include controls, negative controls, and evolved/mutant variants as separate runs.
8. Look for experimental details in Methods, Results, figure legends, table footnotes, and supplementary references.

Return exactly this JSON shape:
{{
  "result_type": "run_metadata_candidates",
  "candidates": [
    {{
      "run_id": "RUN_001",
      "run_description": "short human-readable summary",
      "paper_id": "Author_Year",
      "experiment_type": "growth_curve_assay | evolution_experiment | plasmid_curing_assay | conjugation_assay | MIC_assay | other",
      "temperature": number or null,
      "media": "string or null",
      "culture_format": "string or null",
      "shaking_speed_rpm": number or "—" or null,
      "duration_h": number or string or null,
      "replicates_biological": integer or null,
      "selection_antibiotic": "string or —",
      "selection_concentration": "string or —",
      "initial_dilution": "string or —",
      "species": "string or null",
      "strain_id": "string or null",
      "sequence_type": "string or null",
      "isolation_source": "string or null",
      "plasmid_name": "string or null",
      "plasmid_family": "string or null",
      "plasmid_size_kb": number or string or null,
      "conjugative": "conjugative | mobilizable | non-mobilizable | string or null",
      "resistance_genes": "semicolon-separated or null",
      "plasmid_accession": "string or null",
      "measured_outcomes": "semicolon-separated or null",
      "supporting_evidence": "section/figure/table reference",
      "confidence": "high | medium | low or null"
    }}
  ],
  "notes": "string or null"
}}"""

RUNS_JSON_FIX = "\n\nYour previous reply was not valid JSON for the schema. Return only one corrected JSON object."
