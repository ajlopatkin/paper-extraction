# paper-analysis — figure + text extraction POC

Proof-of-concept: (1) crop figures, run a vision LLM for plots / table-as-image, compare to a ground-truth spreadsheet where configured; (2) extract native PDF text and text tables, then an optional text LLM pass to surface measurement-style candidates for the same spreadsheet world (Runs / Measurements).

## Setup

```bash
cd paper-analysis
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

## Data layout

Place inputs locally (not committed):

- `data/paper.pdf` — source paper
- `data/ground_truth.xlsx` — spreadsheet with Runs / Measurements sheets (optional- for model evaluation)
- `artifacts/text/` — created by `extract-text` / `analyze-text` (under `.gitignore` with other `artifacts/`)

## Configuration

Edit `config/poc.yaml`:

- `pdf.path` — path to the PDF (default `data/paper.pdf`)
- `figures` — each target: `id`, `page` (1-based), `plot_type` (`box_plot` | `line_chart` | `line_plot` | `table_image` | `plasmid_map` | `workflow_diagram` | `experimental_workflow`), and either `bbox_pdf` (x0,y0,x1,y1 in PDF points) or `bbox_norm` (0–1 fractions of page width/height). `line_plot` is an alias of `line_chart` (same JSON schema and prompts). Use `plasmid_map` for vector maps, `workflow_diagram` for generic flowcharts, and `experimental_workflow` for lab / study-protocol pipeline figures (not numeric charts). Optional `skip: true` on an entry skips both cropping and vision for that id (decorative bitmaps, logos, non-data panels).
- `exclude_figure_ids` (optional) — list of `id` strings to omit from `extract-figures` and `run-vision` without removing them from a saved discover file. Applies together with `skip` and with `exclude_figure_ids` in a `--figures-config` YAML (see below).
- `auto_discover` (optional) — derive `bbox_pdf` from embedded images on each page (merge overlaps, drop tiny regions). Use for batch runs instead of hand-drawing every box.
- `evaluation` — sheet name, column names, filters, and numeric tolerance
- `text` (optional) — output directory (`output_dir`), `pages` filter, `max_context_chars` for the LLM, optional `llm_model` override (defaults follow `vision`)
- `export` (optional) — `output_path` for the Excel workbook from `export-results`, and `include_native_pdf_tables` (whether to flatten `artifacts/text/tables.json` into rows)

## Image Analysis Pipeline

### Automatic bounding boxes for PDF images

Discovery collects rasters from `get_image_rects`, `get_text("dict"/"rawdict")` image blocks (`TEXT_PRESERVE_IMAGES`), and `get_text("blocks")` image rows. It also collects vector charts (default `include_vector_graphics: true`): type-3 textpage blocks (`TEXT_COLLECT_VECTORS`) plus `page.get_drawings()` path bboxes, then clusters nearby primitives using `drawing_cluster_gap_frac` so one box/line plot becomes one region. Journal PDFs often store data plots as vectors and decorative art as embedded PNGs—raster-only discovery explains “illustrations yes, plots no.”

Merging is two-phase: (1) `duplicate_merge_iou` (~0.82) collapses the same region from multiple sources; (2) optional `merge_fragments: true` + `merge_iou_threshold` merges multi-tile rasters. With `merge_fragments: false` (default), separate raster panels stay separate. Regions smaller than `min_area_frac` / `min_width_frac` / `min_height_frac` are dropped after merging.

- `auto_discover.enabled: true` with `strategy: only_auto` — `extract-figures` / `run-vision` use only discovered targets (the manual `figures:` list is ignored).
- `strategy: append_to_manual` — run manual `figures` first, then append discovered regions (useful for a few hand-picked panels plus auto coverage).

Tune thresholds per venue if needed. Limits: vector-only figures (pure PDF drawing with no embedded bitmap) may not appear as images; those papers still need another strategy (e.g. layout model or full-page vision) in a later iteration.

Emit a YAML block to review or paste into config:

```bash
# Writes config/discovered_figures.yaml by default (same directory as --config)
paper-analysis discover-bboxes --config config/poc.yaml
# Override path, or print to stdout for piping:
paper-analysis discover-bboxes --config config/poc.yaml -o path/to/figures.yaml
paper-analysis discover-bboxes --config config/poc.yaml --stdout
```

Important: `poc.yaml` does not automatically read `discovered_figures.yaml`. With `auto_discover.enabled: true` in `poc.yaml`, `extract-figures` / `run-vision` re-run discovery in memory (results can differ from an earlier `discover-bboxes` run). To use a saved discover file as the source of truth, pass `--figures-config` to both commands (see CLI below). Alternatively, copy the `figures:` block into `poc.yaml` and set `auto_discover.enabled: false`.

Skipping junk / non-data crops: Discovery often picks decorative images as well as plots. To avoid wasting API calls, use any of: (1) `skip: true` on a figure entry in YAML; (2) top-level `exclude_figure_ids:` in `poc.yaml` (applies even when using `--figures-config`); (3) `exclude_figure_ids:` next to `figures:` in the discover YAML so you can re-run `discover-bboxes` and refresh the file without deleting figure rows. Excluded ids are not cropped and not sent to the vision model; remove stale `artifacts/figures/<id>.png` and `artifacts/extractions/<id>.json` yourself if they were created earlier.

If the file only contains `figures:` and no list items, run with `--verbose` to see per-page counts (`xref_rects`, `image_blocks`, `after_merge`, `after_size_filter`). Earlier versions skipped image blocks because PyMuPDF requires the `TEXT_PRESERVE_IMAGES` flag on `get_text("dict")`; that is now set internally. If `xref_rects` and `image_blocks` are both 0, the page likely has vector-only artwork (no bitmaps in the text layer). If counts are high until `after_size_filter`, try `--relaxed` or lower `min_*_frac` in YAML. If `raw` is 1 on a page that visually has many panels, the publisher likely stored one composite image for the whole figure—auto-crop cannot split panels without a separate panel-detection step. If `raw` is large but `after_dedupe` is 1, try `merge_fragments: false` (default now); if tiles of one figure stay split, set `merge_fragments: true`.

### Finding bounding boxes manually

Render a page to PNG and pick coordinates:

```bash
python tools/preview_page.py data/paper.pdf --page 5 --out artifacts/page_preview.png
```

Open the image in any viewer; note pixel coords. PDF user space matches PyMuPDF pixmap coords when using the same zoom.

### Table as image (`table_image`)

Some papers render a table as a graphic (screenshot-style or flattened artwork). Native PDF table extraction will not see cell text; use the same crop → vision flow as figures, with `plot_type: table_image`. Discovery does not auto-detect “this region is a table”; after `discover-bboxes`, edit the YAML entry (or add a manual `figures` row) to set `plot_type: table_image` for that crop. `run-vision` then writes JSON with `column_headers`, `rows`, optional `title_or_caption` and `notes` (see `[schemas.py](src/paper_analysis/schemas.py)` `TableImageExtraction`). `evaluate` does not compare `table_image` to numeric spreadsheet rows in this POC—use the JSON or a later mapper. True text tables (selectable text in the PDF) still belong in a text/table pipeline (e.g. `find_tables`, pdfplumber), not this vision path.

## Text Analysis Pipeline

Targets experimental findings expressed as selectable PDF text and native text tables (PyMuPDF `find_tables()`), which often carry numbers that also appear in a Measurements sheet (e.g. “Table 2”, inline results). This is separate from figure vision (plots, `table_image`).


| Step       | Command                                                 | Output                                                                                                                                                                                                                                                             |
| ---------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Dump    | `paper-analysis extract-text --config config/poc.yaml`  | `artifacts/text/pages.json` (per-page plain text), `artifacts/text/tables.json` (detected tables with rows). No API key.                                                                                                                                           |
| 2. Analyze | `paper-analysis analyze-text --config config/poc.yaml`  | `artifacts/text/candidate_measurements.json` — LLM-structured list (`TextMeasurementCandidate`: `field_name`, `raw_value`, `source_in_paper`, `source_type` = `table` | `body_text`, etc.). Run before `export-results` if you want those rows in the Excel sheet. |
| 3. Runs    | `paper-analysis extract-runs --config config/poc.yaml`  | `artifacts/text/candidate_runs.json` — LLM-structured list of experimental run metadata (`RunMetadataCandidate`: protocol details, strain/plasmid info, measured outcomes). Run before `export-results` if you want the Extracted_Runs sheet in the Excel workbook. |


**Run metadata extraction** (step 3) identifies each distinct experimental condition in the paper — every unique combination of experiment type, strain, plasmid, and treatment — and extracts structured metadata: protocol parameters (temperature, media, duration, replicates), biological details (species, strain, sequence type, plasmid characteristics, resistance genes), and what was measured. The LLM assigns sequential `Run_ID` values (`RUN_001`, `RUN_002`, …) and derives a `Paper_ID` from the first author and year. This mirrors the Runs sheet in a typical ground-truth spreadsheet. Requires `extract-text` first (same `pages.json` + `tables.json` input as `analyze-text`).

Limits: `find_tables` misses some publisher layouts; complex tables may need pdfplumber/Camelot later. Body text beyond `text.max_context_chars` is head+tail truncated for the LLM. `evaluate` does not yet score text candidates or run metadata against the spreadsheet—merge or validate in a follow-up step.

## CLI

`extract-figures` and `run-vision` — two ways to supply figure targets


| Command pattern                              | Where the figure list comes from                                                                                                                                                    |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config config/poc.yaml` only              | `poc.yaml` only: the `figures:` block and/or `auto_discover` (discovery runs again in memory when `auto_discover.enabled` is true). `discovered_figures.yaml` is never read.        |
| `--config …` and `--figures-config` / `-f` … | The standalone YAML’s top-level `figures:` (e.g. `config/discovered_figures.yaml` from `discover-bboxes`). `poc.yaml` still supplies PDF path, artifacts dirs, and vision settings. |


Use `--config` alone when everything you need is in `poc.yaml` (manual crops, or `auto_discover` defined there). Use `--figures-config` when your authoritative crops live in a separate file—typical after `discover-bboxes` (writes `config/discovered_figures.yaml` by default)—so extraction matches that saved list instead of re-deriving it from `poc.yaml`.

```bash
# List embedded images on a page (optional helper)
paper-analysis list-images data/paper.pdf --page 5

# Suggested figures: YAML from embedded image bboxes (no API)
# Either keep discovered bboxes in discovered_figures.yaml, or move them to poc.yaml
paper-analysis discover-bboxes --config config/poc.yaml   # writes to config/discovered_figures.yaml
paper-analysis discover-bboxes --config config/poc.yaml --out /path/to/output # customer output loc
paper-analysis discover-bboxes --config config/poc.yaml -v --relaxed  # debug empty output
paper-analysis discover-bboxes --config config/poc.yaml --stdout      # print only, no file

# Targets from poc.yaml only (manual figures and/or auto_discover migrated to poc.yaml)
paper-analysis extract-figures --config config/poc.yaml
paper-analysis run-vision --config config/poc.yaml

# Targets from a saved discover file (must match extract + vision)
paper-analysis extract-figures --config config/poc.yaml --figures-config config/discovered_figures.yaml
paper-analysis run-vision --config config/poc.yaml --figures-config config/discovered_figures.yaml

# Inspect spreadsheet columns / rows matching a source filter (no API calls)
paper-analysis inspect-sheet --config config/poc.yaml

# Compare extractions to ground truth
paper-analysis evaluate --config config/poc.yaml

# Text + native tables (no vision images). Run before export-results if you want text rows in the workbook.
paper-analysis extract-text --config config/poc.yaml
paper-analysis analyze-text --config config/poc.yaml

# Run metadata: extract structured experiment conditions (requires extract-text first).
paper-analysis extract-runs --config config/poc.yaml

# Export Measurements + Runs sheets: merges vision JSON + text candidates (if present) + native table cells (if configured) + run metadata (if present)
paper-analysis export-results --config config/poc.yaml
# Same as extract-figures / run-vision: limit vision rows to a saved `figures:` file
paper-analysis export-results --config config/poc.yaml --figures-config config/discovered_figures.yaml
```

Environment:

- `PAPER_ANALYSIS_VISION_PROVIDER` — `anthropic` (default) or `openai`
- Model overrides: `ANTHROPIC_MODEL`, `OPENAI_MODEL`, or `vision.model` in YAML

## Outputs

- `artifacts/figures/` — cropped PNGs
- `artifacts/extractions/` — one JSON file per figure id (`plot_type`: `box_plot`, `line_chart`, `line_plot`, `table_image`, `plasmid_map`, `workflow_diagram`, `experimental_workflow`, or `unknown`). If the model (or an old file) uses an unregistered `plot_type`, it is coerced to `unknown` with the original string in `declared_plot_type` and the full object in `raw`; Python emits a `UserWarning` so you can add a first-class schema in `schemas.py`, prompts in `prompts.py`, and a branch in `plot_type_dispatch.py`.
- `artifacts/text/pages.json` — raw page text from `extract-text`
- `artifacts/text/tables.json` — native tables from `find_tables`
- `artifacts/text/candidate_measurements.json` — text LLM candidates from `analyze-text` (`[text_schemas.py](src/paper_analysis/text_schemas.py)`)
- `artifacts/text/candidate_runs.json` — run metadata from `extract-runs` (`[runs_schemas.py](src/paper_analysis/runs_schemas.py)`)

`export-results` writes `export.output_path` (default `artifacts/export/extractions.xlsx`) with up to three sheets:

- Extracted_Measurements — columns aligned with a typical ground-truth Measurements sheet (`Run_ID`, `Field_name`, `Raw_value`, `Raw_units`, `Source_in_paper`, `Notes`) plus provenance: `Extraction_pipeline` (`figure_vision` | `text_llm` | `native_pdf_table`), `Extraction_method` (e.g. `box_plot_median`, `line_chart_point`, `line_plot_point`, `table_image_cell`, `plasmid_map_feature`, `workflow_diagram_node`, `experimental_workflow_node`, `text_llm_candidate`, `native_table_cell`), `Source_artifact` (path to the JSON), `Source_id`, `Plot_type`, `Page` (from `pNNN`_-style figure ids when present), `Axis_or_context`, `Confidence`, `Supporting_evidence`.
- Extracted_Runs (when `candidate_runs.json` exists) — columns aligned with a typical ground-truth Runs sheet: `Run_ID`, `Run_description`, `Paper_ID`, `experiment_type`, `temperature`, `media`, `culture_format`, `shaking_speed_rpm`, `duration_h`, `replicates_biological`, `selection_antibiotic`, `selection_concentration`, `initial_dilution`, `species`, `strain_id`, `sequence_type`, `isolation_source`, `plasmid_name`, `plasmid_family`, `plasmid_size_kb`, `conjugative`, `resistance_genes`, `plasmid_accession`, `measured_outcomes`, `supporting_evidence`, `confidence`.
- Export_Meta — export timestamp, PDF path, row counts, flags.

Vision rows come from every `artifacts/extractions/*.json` (or only ids in `--figures-config` when passed). Text LLM rows require `analyze-text` to have run first so `artifacts/text/candidate_measurements.json` exists; otherwise export still succeeds with only vision (and native-table) rows. Native table rows need `extract-text` (for `tables.json`) when `export.include_native_pdf_tables` is true. Run metadata rows require `extract-runs` to have run first so `artifacts/text/candidate_runs.json` exists; otherwise export proceeds without the Extracted_Runs sheet.

A later mapper can still join these rows into canonical Runs / Measurements with human-assigned `Run_ID` and tighter `Source_in_paper` phrasing to match a specific spreadsheet.
