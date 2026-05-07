# paper-analysis

Automated extraction of structured experimental data from scientific papers about plasmid biology, horizontal gene transfer, and bacterial genetics. Processes PDFs end-to-end: text extraction, figure discovery, vision-model analysis, LLM-based measurement extraction, and post-processing — producing denormalized tables of experimental measurements with full run metadata.

## Quick Start

### Processing New Papers (No Ground Truth)

The primary use case: extract structured data from papers you haven't manually annotated.

```bash
# 1. Setup
cd paper-analysis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# 2. Drop PDFs into data/papers/

# 3. Create a minimal batch config (or add to existing batch.yaml)
#    Papers not listed in paper_map auto-derive an ID from the filename.

# 4. Run the full pipeline (sequential mode — good for small batches)
paper-analysis batch-run -c config/batch.yaml

# 5. Post-process (no API calls — expands vision data, filters false positives)
paper-analysis batch-postprocess -c config/batch.yaml

# 6. Collect results
#    Each paper's output is at:
#      artifacts/<Paper_ID>/text/candidate_combined.json
```

That's it. Each paper gets its own artifact directory under `artifacts/<Paper_ID>/` with text, figures, vision extractions, and the final `candidate_combined.json` — a flat table of measurement rows with full experimental metadata.

**No ground truth is needed.** The `batch-evaluate` step is entirely optional and only relevant when you have a manually curated spreadsheet to compare against.

You can also process a single new paper without touching the batch config:

```bash
# Process one PDF at a time via --paper (uses filename stem as Paper_ID)
paper-analysis batch-run -c config/batch.yaml -p NewPaper_2025
paper-analysis batch-postprocess -c config/batch.yaml -p NewPaper_2025
```

### Large-Scale Processing (Batch API Mode)

For processing hundreds to hundreds of thousands of papers, use `--batch-api` to leverage the Anthropic Message Batches API. This provides **50% cost reduction**, **prompt caching** (90% savings on the static instruction prefix), and **two-tier model routing** (cheap model for simple papers, premium for complex ones).

```bash
# Run with batch API — all API calls submitted asynchronously
paper-analysis batch-run -c config/batch.yaml --batch-api

# Resume an interrupted batch run
paper-analysis batch-run -c config/batch.yaml --batch-api --resume

# Check status of active batches
paper-analysis batch-status -c config/batch.yaml
```

At scale, batch API mode reduces cost from ~$0.90/paper to ~$0.06/paper (a ~15x reduction), enabling 100K-paper runs for ~$6K instead of ~$90K. See [Batch API Mode](#batch-api-mode) for details.

### Adding New Papers to an Existing Batch

To add papers to a batch that already has a `paper_map`:

1. Drop the PDF into `data/papers/`.
2. Optionally add a `paper_map` entry in `batch.yaml` to assign a custom `Paper_ID`. If omitted, the filename stem is used (e.g., `Smith_2024.pdf` becomes `Smith_2024`).
3. Run `batch-run` with `-p` to process just the new paper without re-running everything.

### Evaluating Against Ground Truth (Optional)

If you have a ground truth spreadsheet for benchmarking:

```bash
# Configure ground_truth_path and ground_truth_sheet in batch.yaml, then:
paper-analysis batch-evaluate -c config/batch.yaml
paper-analysis batch-evaluate -c config/batch.yaml --no-gaps  # skip gap analysis
```

## What the Pipeline Does

For each paper, `batch-run` executes five stages:

| Stage | What happens | API calls? |
|-------|-------------|------------|
| 1. Extract text | Dump PDF text + native tables via PyMuPDF | No |
| 2. Discover figures | Find figure bounding boxes in the PDF | No |
| 3. Vision extraction | Classify each figure (box plot, line chart, heatmap, etc.) and extract structured data | Yes (vision LLM) |
| 4. Combined extraction | LLM reads text + tables + figure data, produces denormalized measurement rows | Yes (text LLM) |
| 5. Post-processing | Expand vision data points into individual rows, derive qualitative fields, filter by master field list | No |

The final output is `artifacts/<Paper_ID>/text/candidate_combined.json` — a flat table where every measurement row carries full experimental context (strain, plasmid, conditions, etc.).

In **sequential mode** (default), each paper processes one at a time with synchronous API calls. In **batch API mode** (`--batch-api`), local prep runs for all papers first, then all API calls are submitted as Anthropic Message Batches for asynchronous processing at 50% reduced cost.

### Post-Processing Details

`batch-postprocess` runs three zero-API-cost steps on existing artifacts:

1. **Vision data expansion**: Each line-chart point, box-plot median, heatmap cell, and table-image value becomes its own measurement row (the LLM often summarizes figures rather than emitting individual data points).

2. **Qualitative field derivation**: For papers with pairwise conjugation matrices (e.g., Gama 2017), derives `conjugation_detection_status` and `conjugation_interaction_direction` from extracted `conjugation_rate` values.

3. **Master Field List filter**: Uses `MASTER_FIELD_LIST_Combined_15papers.xlsx` to remove rows with field names not recognized by either the curated field list or the synonym mapping system. Eliminates false positives from experimental-setup parameters the LLM over-extracts.

## Data Layout

```
paper-analysis/
├── config/
│   └── batch.yaml              # Batch processing config
├── data/
│   ├── papers/                  # Source PDFs (not committed)
│   ├── RUNS_MEASUREMENTS_15files.xlsx      # Ground truth
│   └── MASTER_FIELD_LIST_Combined_15papers.xlsx  # Field definitions
├── artifacts/                   # All outputs (gitignored)
│   └── <Paper_ID>/
│       ├── text/
│       │   ├── pages.json                  # Raw page text
│       │   ├── tables.json                 # Native tables
│       │   └── candidate_combined.json     # Final extraction output
│       ├── figures/                        # Cropped figure PNGs
│       └── extractions/                    # Per-figure vision JSONs
├── src/paper_analysis/
│   ├── batch.py                 # Batch orchestration (sequential + batch API)
│   ├── batch_api.py             # Anthropic Message Batches API wrapper
│   ├── batch_evaluate.py        # Evaluation + synonym mapping
│   ├── postprocess.py           # Post-processing + MFL filter
│   ├── combined_prompts.py      # LLM extraction prompts
│   ├── combined_analyze_llm.py  # LLM API calls + batch request builders
│   ├── cli.py                   # All CLI commands
│   └── vision/                  # Vision model clients
└── PROMPT_ITERATION_LOG.md      # Iteration history + metrics
```

## Configuration

### `config/batch.yaml`

```yaml
papers_dir: data/papers              # required: directory containing PDFs
artifacts_dir: artifacts             # required: where outputs go

# Optional — only needed for batch-evaluate:
ground_truth_path: data/RUNS_MEASUREMENTS_15files.xlsx
ground_truth_sheet: RUNS_MEASUREMENTS_Master

# Optional — improves false-positive filtering in post-processing:
field_list_path: data/MASTER_FIELD_LIST_Combined_15papers.xlsx

# Optional — map filenames to custom Paper_IDs. Unmapped PDFs use filename stem.
paper_map:
  - filename: Chen_2017.pdf
    paper_id: "006_Chen_2017"
  # ... one entry per paper

vision:
  provider: anthropic      # or openai
  model: null              # uses provider default

auto_discover:
  enabled: true
  strategy: only_auto
  default_plot_type: auto  # auto-classify each figure
```

**Minimal config for new papers** (no ground truth, no MFL):

```yaml
papers_dir: data/papers
artifacts_dir: artifacts
auto_discover:
  enabled: true
  strategy: only_auto
  default_plot_type: auto
```

### Two-Tier Model Config (for `--batch-api` mode)

The `tier` section controls model routing when using `--batch-api`. Papers are assigned to a cheap or premium model based on heuristics and explicit overrides:

```yaml
tier:
  default_tier: cheap                       # "cheap" or "premium"
  cheap_model: claude-haiku-4-5             # fast, low-cost model
  premium_model: claude-sonnet-4-20250514   # higher accuracy for complex papers

  # Explicit per-paper overrides (take priority over auto rules)
  premium_papers: []          # paper IDs forced to premium
  cheap_papers: []            # paper IDs forced to cheap (even if auto rules say premium)

  # Automatic routing heuristics
  auto_tier_rules:
    enabled: true
    page_threshold: 30        # papers with >30 pages → premium
    figure_threshold: 20      # papers with >20 figures → premium
    scanned_text_ratio: 0.3   # >30% of pages lack native tables → premium
```

The routing priority is: explicit `cheap_papers` > explicit `premium_papers` > auto heuristics > `default_tier`.

### Environment Variables

Set in `.env`:

- `ANTHROPIC_API_KEY` — required for Anthropic provider
- `OPENAI_API_KEY` — required for OpenAI provider
- `PAPER_ANALYSIS_VISION_PROVIDER` — `anthropic` (default) or `openai`
- `ANTHROPIC_MODEL` / `OPENAI_MODEL` — optional model overrides

## CLI Reference

### Batch Commands (recommended)

```bash
# Full pipeline: text + figures + vision + LLM + postprocess for all papers
paper-analysis batch-run -c config/batch.yaml

# Process specific papers only
paper-analysis batch-run -c config/batch.yaml -p Smith_2024 -p Jones_2023

# Skip vision (text-only extraction, no figure processing)
paper-analysis batch-run -c config/batch.yaml --skip-vision

# Batch API mode (50% cost savings, async, prompt caching, tiered models)
paper-analysis batch-run -c config/batch.yaml --batch-api

# Resume an interrupted batch API run
paper-analysis batch-run -c config/batch.yaml --batch-api --resume

# Custom poll interval (default 30s)
paper-analysis batch-run -c config/batch.yaml --batch-api --poll-interval 60

# Check status of active batch API runs
paper-analysis batch-status -c config/batch.yaml

# Post-process only (no API calls, runs on existing artifacts)
paper-analysis batch-postprocess -c config/batch.yaml

# Post-process specific papers
paper-analysis batch-postprocess -c config/batch.yaml -p 003_Gama_2017

# Evaluate against ground truth (optional, requires ground_truth_path in config)
paper-analysis batch-evaluate -c config/batch.yaml
paper-analysis batch-evaluate -c config/batch.yaml --no-gaps  # skip gap analysis
```

### Single-Paper Commands

For working with individual papers via `config/poc.yaml`:

```bash
paper-analysis extract-text -c config/poc.yaml       # dump text + tables
paper-analysis extract-figures -c config/poc.yaml     # crop figure panels
paper-analysis run-vision -c config/poc.yaml          # vision LLM on crops
paper-analysis extract-combined -c config/poc.yaml    # combined LLM extraction
paper-analysis export-results -c config/poc.yaml      # export to Excel
```

### Discovery and Debugging

```bash
# List embedded images on a page
paper-analysis list-images data/paper.pdf --page 5

# Discover figure bounding boxes
paper-analysis discover-bboxes -c config/poc.yaml
paper-analysis discover-bboxes -c config/poc.yaml -v --relaxed  # debug empty output

# Inspect ground truth spreadsheet
paper-analysis inspect-sheet -c config/poc.yaml

# Evaluate single paper against ground truth
paper-analysis evaluate -c config/poc.yaml
```

## How Evaluation Works (Optional)

`batch-evaluate` is only needed when benchmarking against a ground truth spreadsheet. It compares extracted rows against ground truth using:

- **Field name matching**: A synonym system (`_SYNONYM_GROUPS` in `batch_evaluate.py`) canonicalizes 330+ field name variants into groups. For example, `conjugation_rate`, `conjugation_rate_estimate`, `conjugation_transfer_frequency`, and `transfer_frequency` all map to the same canonical form.
- **Key construction**: Rows are matched on `(paper_id, field_name_canonical, raw_value_rounded)` with a loose fallback on `(paper_id, field_name_canonical)`.
- **Metrics**: Precision, Recall, F1-score, and MAE (for numeric values).
- **Gap analysis**: Reports missing field types per paper to guide synonym expansion.

## Master Field List (Optional)

`MASTER_FIELD_LIST_Combined_15papers.xlsx` is an optional curated vocabulary that improves precision by filtering false positives during post-processing. Without it, the pipeline still works — you just get more extracted rows (some of which may be experimental setup parameters rather than true measurements). It provides:

| Column | Use |
|--------|-----|
| `Field_Name` | Primary normalised field name |
| `Synonyms` | Semicolon-separated aliases |
| `Field_Category` | Classification (e.g., `conjugation_measurement`, `fitness_measurement`, `paper_metadata`) |
| `Canonical_Field` | `Yes` for 35 highest-priority fields |
| `Papers` | Which papers use this field (e.g., "Gama 2017; Igler 2021") |
| `Data_Type` | Expected type (`float`, `integer`, `string`, `categorical`) |
| `Definition` | Human-readable description |
| `Normalization_Logic` | How to convert raw values to normalised form |

The post-processing step uses this to filter out false positives: extracted rows whose field names don't appear in the MFL (or the synonym system) are removed.

## Performance

Current metrics across 15 papers (Iteration 7):

| Metric | Value |
|--------|-------|
| Ground truth rows | 2,586 |
| Extracted rows | 1,442 |
| Matched rows | 915 |
| Precision | 0.635 |
| Recall | 0.354 |
| F1 | 0.454 |
| False positive rate | 36.5% |

See `PROMPT_ITERATION_LOG.md` for full iteration history (Iterations 0-7, F1 improved from 0.113 to 0.454 — a 302% gain).

## Architecture

### Vision Pipeline

Figures are auto-classified into plot types before extraction:

- **box_plot**: Medians, quartiles, whiskers per group
- **line_chart**: Per-series, per-timepoint (x, y) values
- **heatmap**: Row/column/value/annotation per cell (for interaction matrices)
- **table_image**: Column headers + data rows (for scanned tables)

Classification aliases handle model variability (e.g., `scatter_plot` → `line_chart`, `stacked_bar` → `box_plot`, `interaction_matrix` → `heatmap`).

### Combined LLM Extraction

The combined extraction prompt guides the LLM to produce a denormalized table with:

- Sequential `Run_ID` and `Measurement_ID` assignments
- Full run metadata on every row (species, strain, plasmid, conditions)
- Canonical `field_name` vocabulary (130+ names across 12 categories)
- Continuation-based chunking for papers exceeding the 64K output token limit

### Post-Processing Pipeline

Zero-API-cost steps that run on existing artifacts:

1. Vision data expansion (line chart points → individual rows)
2. Qualitative field derivation (conjugation rates → detection status)
3. Master Field List whitelist filter (removes false positives)

Each step is idempotent. Original LLM output is preserved in `.bak` files.

## Batch API Mode

The `--batch-api` flag switches from sequential (one paper at a time) to asynchronous batch processing using the [Anthropic Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/message-batches). This combines three cost optimizations:

### How It Works

The pipeline runs in four phases:

1. **Local prep** — Extract text, discover figures, and crop PNGs for all papers. No API calls.
2. **Vision batch** — Submit all figure classification requests as a batch, wait for results, then submit all extraction requests as a second batch.
3. **Combined extraction batch** — Submit all combined extraction requests as a single batch with prompt caching and tier-based model selection. Automatically handles truncated outputs with continuation batches.
4. **Post-processing** — Expand vision data and apply MFL filtering. No API calls.

### Cost Savings

| Optimization | Savings | How |
|-------------|---------|-----|
| Batch API | 50% | Anthropic charges half price for async batch processing |
| Prompt caching | ~90% on static prefix | The ~4K-token instruction prompt is identical across all papers; with 1-hour TTL cache, subsequent reads cost 0.1x base price |
| Tier routing | ~3-5x on simple papers | Haiku 4.5 is ~3x cheaper than Sonnet 4; most papers don't need the premium model |

**Estimated cost at scale:**

| Scale | Sequential (Sonnet 4) | Batch API (tiered) |
|-------|----------------------|-------------------|
| 20 papers | ~$18 | ~$1.20 |
| 1,000 papers | ~$900 | ~$60 |
| 100,000 papers | ~$90,000 | ~$6,000 |

### Resume Support

Batch state is persisted to `artifacts/_batch_state/` so interrupted runs can be resumed:

```bash
# If a run is interrupted (Ctrl-C, network error, etc.)
paper-analysis batch-run -c config/batch.yaml --batch-api --resume
```

The `batch-status` command shows the state of all active batches without modifying anything:

```bash
paper-analysis batch-status -c config/batch.yaml
```

## Outputs

### Per-Paper Artifacts

- `artifacts/<Paper_ID>/text/candidate_combined.json` — **primary output**: denormalized measurement rows with run metadata
- `artifacts/<Paper_ID>/text/candidate_combined.json.bak` — pre-post-processing backup of LLM output
- `artifacts/<Paper_ID>/text/pages.json` — raw page text
- `artifacts/<Paper_ID>/text/tables.json` — native PDF tables
- `artifacts/<Paper_ID>/figures/*.png` — cropped figure images
- `artifacts/<Paper_ID>/extractions/*.json` — per-figure vision extraction results

### Understanding `candidate_combined.json`

Each row in the `candidates` array is a single measurement with full context:

```json
{
  "measurement_id": "M_001",
  "run_id": "RUN_001",
  "paper_id": "Smith_2024",
  "experiment_type": "conjugation_assay",
  "species": "Escherichia coli",
  "strain_id": "K-12 MG1655",
  "plasmid_name": "R388",
  "medium": "LB broth",
  "temperature": "37",
  "field_name": "conjugation_rate",
  "measurement_time_h": "4",
  "raw_value": "1.2e-3",
  "raw_units": "transconjugants/donor",
  "source_in_paper": "Table 2",
  "confidence": "high"
}
```

Key fields: `field_name` identifies the measurement type, `raw_value` + `raw_units` carry the data, and `run_id` groups measurements from the same experimental condition. The run metadata columns (`species`, `strain_id`, `plasmid_name`, `medium`, etc.) are denormalized onto every row for easy filtering.

### Export (Single-Paper Mode)

`export-results` writes an Excel workbook with sheets:

- **Combined_Runs_Measurements** — denormalized rows (recommended)
- **Extracted_Measurements** — vision + text LLM candidates
- **Extracted_Runs** — structured run metadata
- **Export_Meta** — provenance and row counts
