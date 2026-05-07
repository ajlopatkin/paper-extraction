"""Batch processing: iterate over a directory of papers and run the full pipeline on each.

Supports two execution modes:
- Sequential (default): processes papers one at a time with synchronous API calls.
- Batch API (--batch-api): uses the Anthropic Message Batches API for 50% cost
  reduction, prompt caching, and two-tier model routing.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from paper_analysis.batch_api import (
    BatchRequestItem,
    BatchState,
    submit_poll_collect,
    resume_collect,
)
from paper_analysis.combined_analyze_llm import (
    build_combined_batch_request,
    build_continuation_batch_request,
    parse_batch_result_text,
    run_combined_analysis,
)
from paper_analysis.combined_schemas import CombinedExtractionBatch
from paper_analysis.config import (
    BatchConfig,
    FigureTarget,
    PocConfig,
    effective_text_config,
    resolve_figure_targets,
)
from paper_analysis.pdf_discovery import discover_figure_targets
from paper_analysis.pdf_figures import extract_all_figures
from paper_analysis.pdf_text_tables import (
    build_figure_summary,
    build_llm_context,
    load_text_artifacts,
    run_text_dump,
)
from paper_analysis.postprocess import run_postprocess_for_paper
from paper_analysis.vision.anthropic_client import (
    build_classify_batch_request,
    build_extract_batch_request,
    parse_classify_result,
)
from paper_analysis.vision.base import build_vision_client, parse_extraction_text


@dataclass
class PaperResult:
    paper_id: str
    pdf_path: Path
    n_figures: int = 0
    n_extractions: int = 0
    n_combined_rows: int = 0
    error: str | None = None


@dataclass
class BatchResult:
    papers: list[PaperResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[PaperResult]:
        return [p for p in self.papers if p.error is None]

    @property
    def failed(self) -> list[PaperResult]:
        return [p for p in self.papers if p.error is not None]

    def summary(self) -> str:
        lines = [f"Batch complete: {len(self.succeeded)}/{len(self.papers)} papers succeeded"]
        total_rows = sum(p.n_combined_rows for p in self.succeeded)
        lines.append(f"Total combined rows: {total_rows}")
        if self.failed:
            lines.append("Failed papers:")
            for p in self.failed:
                lines.append(f"  {p.paper_id}: {p.error}")
        return "\n".join(lines)


def list_pdfs(batch_cfg: BatchConfig) -> list[tuple[Path, str]]:
    """Return sorted list of (pdf_path, paper_id) for all PDFs in papers_dir."""
    papers_dir = batch_cfg.papers_dir
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}")
    return [(p, batch_cfg.paper_id_for(p.name)) for p in pdfs]


def run_pipeline_for_paper(
    batch_cfg: BatchConfig,
    pdf_path: Path,
    paper_id: str,
    *,
    skip_vision: bool = False,
    echo: callable = print,
) -> PaperResult:
    """Run the full extract-text -> figures -> vision -> combined pipeline for one paper."""
    result = PaperResult(paper_id=paper_id, pdf_path=pdf_path)
    cfg = batch_cfg.poc_config_for_paper(pdf_path, paper_id)

    try:
        # 1. Extract text + native tables
        tc = effective_text_config(cfg)
        echo(f"  [{paper_id}] extract-text ...")
        run_text_dump(cfg.pdf, tc)

        # 2. Discover + extract figures (increase recursion limit for complex PDFs)
        targets = []
        if cfg.auto_discover and cfg.auto_discover.enabled:
            echo(f"  [{paper_id}] discover-bboxes ...")
            old_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(max(old_limit, 10000))
                targets = resolve_figure_targets(cfg)
            except RecursionError:
                echo(f"  [{paper_id}] discover-bboxes hit recursion limit, skipping figures")
                targets = []
            finally:
                sys.setrecursionlimit(old_limit)
            result.n_figures = len(targets)
            if targets:
                echo(f"  [{paper_id}] extract-figures ({len(targets)} targets) ...")
                extract_all_figures(cfg.pdf, targets, cfg.artifacts)

        # 3. Run vision on each crop
        if targets and not skip_vision:
            provider = cfg.vision.provider or os.environ.get(
                "PAPER_ANALYSIS_VISION_PROVIDER", "anthropic"
            )
            client = build_vision_client(provider=provider, model=cfg.vision.model)
            ext_dir = cfg.artifacts.extractions_dir
            ext_dir.mkdir(parents=True, exist_ok=True)
            fig_dir = cfg.artifacts.figures_dir
            n_ext = 0
            for fig in targets:
                png_path = fig_dir / f"{fig.id}.png"
                if not png_path.is_file():
                    continue
                data = png_path.read_bytes()
                try:
                    extraction = client.extract_figure(data, fig.plot_type, max_retries=1)
                    dest = ext_dir / f"{fig.id}.json"
                    dest.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
                    n_ext += 1
                except Exception:
                    echo(f"  [{paper_id}] vision failed for {fig.id}: {traceback.format_exc()}")
            result.n_extractions = n_ext
            echo(f"  [{paper_id}] run-vision ({n_ext} extractions)")

        # 4. Combined extraction
        echo(f"  [{paper_id}] extract-combined ...")
        pages, tables = load_text_artifacts(tc)
        content = build_llm_context(pages, tables, tc.max_context_chars)
        fig_summary = build_figure_summary(cfg.artifacts.extractions_dir)
        if fig_summary:
            content = content + "\n\n" + fig_summary

        provider = cfg.vision.provider or os.environ.get(
            "PAPER_ANALYSIS_VISION_PROVIDER", "anthropic"
        )
        model = tc.llm_model or cfg.vision.model
        batch_out = run_combined_analysis(content, provider=provider, model=model, max_retries=2)
        if not batch_out.candidates:
            echo(f"  [{paper_id}] got 0 candidates, retrying...")
            batch_out = run_combined_analysis(content, provider=provider, model=model, max_retries=2)

        out_path = tc.output_dir / "candidate_combined.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(batch_out.model_dump_json(indent=2), encoding="utf-8")

        # 5. Post-process: expand vision data into individual measurement rows
        ext_dir = cfg.artifacts.extractions_dir
        if ext_dir.is_dir() and any(ext_dir.glob("*.json")):
            run_postprocess_for_paper(
                paper_id, ext_dir, out_path,
                mfl_path=batch_cfg.field_list_path,
                echo=echo,
            )
            merged = json.loads(out_path.read_text(encoding="utf-8"))
            result.n_combined_rows = len(merged.get("candidates", []))
        else:
            result.n_combined_rows = len(batch_out.candidates)

        n_runs = len({c.get("run_id", "") for c in json.loads(out_path.read_text(encoding="utf-8")).get("candidates", [])})
        echo(
            f"  [{paper_id}] done: {result.n_combined_rows} measurements, {n_runs} runs"
        )

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        echo(f"  [{paper_id}] FAILED: {result.error}")

    return result


def run_batch(
    batch_cfg: BatchConfig,
    *,
    papers: list[str] | None = None,
    skip_vision: bool = False,
    echo: callable = print,
) -> BatchResult:
    """Run the pipeline on all (or selected) papers."""
    all_pdfs = list_pdfs(batch_cfg)
    if papers:
        filter_set = set(papers)
        all_pdfs = [(p, pid) for p, pid in all_pdfs if pid in filter_set or p.stem in filter_set]
        if not all_pdfs:
            raise ValueError(f"No PDFs matched filter: {papers}")

    echo(f"Processing {len(all_pdfs)} papers...")
    batch_result = BatchResult()
    for pdf_path, paper_id in all_pdfs:
        echo(f"\n--- {paper_id} ({pdf_path.name}) ---")
        pr = run_pipeline_for_paper(
            batch_cfg, pdf_path, paper_id, skip_vision=skip_vision, echo=echo
        )
        batch_result.papers.append(pr)

    echo(f"\n{batch_result.summary()}")
    return batch_result


def merge_combined_outputs(batch_cfg: BatchConfig) -> list[dict]:
    """Load all per-paper candidate_combined.json and merge into one list."""
    all_rows: list[dict] = []
    papers_dir = batch_cfg.papers_dir
    if not papers_dir.is_dir():
        return all_rows
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        paper_id = batch_cfg.paper_id_for(pdf_path.name)
        combined_path = batch_cfg.artifacts_dir / paper_id / "text" / "candidate_combined.json"
        if not combined_path.is_file():
            continue
        try:
            data = json.loads(combined_path.read_text(encoding="utf-8"))
            for c in data.get("candidates", []):
                c["_paper_id_mapped"] = paper_id
                all_rows.append(c)
        except (json.JSONDecodeError, OSError):
            continue
    return all_rows


# ---------------------------------------------------------------------------
# Batch API mode: Anthropic Message Batches with prompt caching + tiered models
# ---------------------------------------------------------------------------

@dataclass
class _PaperPrep:
    """Intermediate state for a paper after local preprocessing."""

    paper_id: str
    pdf_path: Path
    cfg: PocConfig
    targets: list[FigureTarget]
    n_pages: int = 0
    n_pages_with_tables: int = 0


def _run_local_prep(
    batch_cfg: BatchConfig,
    pdf_path: Path,
    paper_id: str,
    echo: callable = print,
) -> _PaperPrep:
    """Phase 1: extract text, discover figures, crop PNGs. No API calls."""
    cfg = batch_cfg.poc_config_for_paper(pdf_path, paper_id)
    tc = effective_text_config(cfg)

    echo(f"  [{paper_id}] extract-text ...")
    run_text_dump(cfg.pdf, tc)

    pages, tables = load_text_artifacts(tc)
    n_pages = len(pages)
    n_pages_with_tables = sum(1 for t in tables if t) if tables else 0

    targets: list[FigureTarget] = []
    if cfg.auto_discover and cfg.auto_discover.enabled:
        echo(f"  [{paper_id}] discover-bboxes ...")
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(max(old_limit, 10000))
            targets = resolve_figure_targets(cfg)
        except RecursionError:
            echo(f"  [{paper_id}] discover-bboxes hit recursion limit, skipping figures")
            targets = []
        finally:
            sys.setrecursionlimit(old_limit)

        if targets:
            echo(f"  [{paper_id}] extract-figures ({len(targets)} targets) ...")
            extract_all_figures(cfg.pdf, targets, cfg.artifacts)

    return _PaperPrep(
        paper_id=paper_id,
        pdf_path=pdf_path,
        cfg=cfg,
        targets=targets,
        n_pages=n_pages,
        n_pages_with_tables=n_pages_with_tables,
    )


def _collect_figure_images(
    prep: _PaperPrep,
) -> dict[str, tuple[str, str]]:
    """Load cropped PNGs as base64, keyed by figure_id. Returns {fig_id: (b64, plot_type)}."""
    fig_dir = prep.cfg.artifacts.figures_dir
    images: dict[str, tuple[str, str]] = {}
    for fig in prep.targets:
        png_path = fig_dir / f"{fig.id}.png"
        if png_path.is_file():
            b64 = base64.standard_b64encode(png_path.read_bytes()).decode("ascii")
            images[fig.id] = (b64, fig.plot_type)
    return images


def run_batch_api(
    batch_cfg: BatchConfig,
    *,
    papers: list[str] | None = None,
    skip_vision: bool = False,
    resume: bool = False,
    poll_interval: float = 30.0,
    echo: callable = print,
) -> BatchResult:
    """Run the pipeline using Anthropic Message Batches API for all API calls.

    Phases:
      1. Local prep (text extraction, figure discovery, figure cropping)
      2. Vision batch (classify → extract) via Anthropic Batches API
      3. Combined extraction batch with prompt caching and tiered models
      4. Post-processing (local)
    """
    all_pdfs = list_pdfs(batch_cfg)
    if papers:
        filter_set = set(papers)
        all_pdfs = [
            (p, pid) for p, pid in all_pdfs
            if pid in filter_set or p.stem in filter_set
        ]
        if not all_pdfs:
            raise ValueError(f"No PDFs matched filter: {papers}")

    state_dir = batch_cfg.artifacts_dir / "_batch_state"
    client = anthropic.Anthropic(timeout=600.0)

    # ── Phase 1: Local prep ──────────────────────────────────────────────
    echo(f"\n{'='*60}")
    echo(f"Phase 1: Local preprocessing for {len(all_pdfs)} papers")
    echo(f"{'='*60}")

    preps: list[_PaperPrep] = []
    for pdf_path, paper_id in all_pdfs:
        echo(f"\n--- {paper_id} ({pdf_path.name}) ---")
        try:
            prep = _run_local_prep(batch_cfg, pdf_path, paper_id, echo=echo)
            preps.append(prep)
        except Exception as exc:
            echo(f"  [{paper_id}] local prep FAILED: {exc}")

    echo(f"\nPhase 1 complete: {len(preps)}/{len(all_pdfs)} papers prepared")

    # ── Phase 2: Vision batch ────────────────────────────────────────────
    if not skip_vision:
        _run_vision_batch_phase(batch_cfg, preps, client, state_dir, resume, poll_interval, echo)
    else:
        echo("\nPhase 2: Vision SKIPPED (--skip-vision)")

    # ── Phase 3: Combined extraction batch ───────────────────────────────
    _run_combined_batch_phase(batch_cfg, preps, client, state_dir, resume, poll_interval, echo)

    # ── Phase 4: Post-processing ─────────────────────────────────────────
    echo(f"\n{'='*60}")
    echo("Phase 4: Post-processing")
    echo(f"{'='*60}")

    batch_result = BatchResult()
    for prep in preps:
        result = PaperResult(
            paper_id=prep.paper_id,
            pdf_path=prep.pdf_path,
            n_figures=len(prep.targets),
        )
        try:
            tc = effective_text_config(prep.cfg)
            out_path = tc.output_dir / "candidate_combined.json"
            ext_dir = prep.cfg.artifacts.extractions_dir

            if not out_path.is_file():
                result.error = "No combined output produced"
                batch_result.papers.append(result)
                continue

            if ext_dir.is_dir() and any(ext_dir.glob("*.json")):
                run_postprocess_for_paper(
                    prep.paper_id, ext_dir, out_path,
                    mfl_path=batch_cfg.field_list_path,
                    echo=echo,
                )

            merged = json.loads(out_path.read_text(encoding="utf-8"))
            result.n_combined_rows = len(merged.get("candidates", []))
            n_runs = len({c.get("run_id", "") for c in merged.get("candidates", [])})
            echo(f"  [{prep.paper_id}] done: {result.n_combined_rows} measurements, {n_runs} runs")
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            echo(f"  [{prep.paper_id}] post-process FAILED: {result.error}")

        batch_result.papers.append(result)

    echo(f"\n{batch_result.summary()}")
    return batch_result


def _run_vision_batch_phase(
    batch_cfg: BatchConfig,
    preps: list[_PaperPrep],
    client: anthropic.Anthropic,
    state_dir: Path,
    resume: bool,
    poll_interval: float,
    echo: callable,
) -> None:
    """Phase 2: classify figures, then extract data — both via batch API."""
    echo(f"\n{'='*60}")
    echo("Phase 2: Vision batch (classify → extract)")
    echo(f"{'='*60}")

    vision_model = (
        batch_cfg.vision.model
        or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    )

    # Collect all figure images across all papers
    # paper_figures: {paper_id: {fig_id: (b64, plot_type)}}
    paper_figures: dict[str, dict[str, tuple[str, str]]] = {}
    for prep in preps:
        images = _collect_figure_images(prep)
        if images:
            paper_figures[prep.paper_id] = images

    if not paper_figures:
        echo("  No figures to process.")
        return

    total_figs = sum(len(figs) for figs in paper_figures.values())
    echo(f"  {total_figs} figures across {len(paper_figures)} papers")

    # 2a. Classification batch — only for figures with plot_type == "auto"
    classify_requests: list[BatchRequestItem] = []
    for paper_id, figs in paper_figures.items():
        for fig_id, (b64, plot_type) in figs.items():
            if plot_type == "auto":
                cid = f"{paper_id}__{fig_id}__classify"
                req = build_classify_batch_request(cid, b64, vision_model)
                classify_requests.append(BatchRequestItem(custom_id=req["custom_id"], params=req["params"]))

    # Map from paper_id__fig_id to resolved plot_type
    resolved_types: dict[str, str] = {}

    if classify_requests:
        echo(f"\n  Submitting {len(classify_requests)} classification requests...")
        classify_state_path = state_dir / "vision_classify_state.json"

        if resume:
            saved = BatchState.load(classify_state_path)
            if saved.batch_ids and not saved.completed:
                echo("  Resuming classification batch...")
                classify_results = resume_collect(
                    classify_state_path, client=client,
                    poll_interval=poll_interval, echo=echo,
                )
            elif saved.completed:
                echo("  Classification already completed, re-collecting...")
                from paper_analysis.batch_api import collect_results
                classify_results = collect_results(saved.batch_ids, client=client, echo=echo)
            else:
                classify_results = submit_poll_collect(
                    classify_requests, client=client,
                    state_path=classify_state_path, phase_name="Vision Classify",
                    poll_interval=poll_interval, echo=echo,
                )
        else:
            classify_results = submit_poll_collect(
                classify_requests, client=client,
                state_path=classify_state_path, phase_name="Vision Classify",
                poll_interval=poll_interval, echo=echo,
            )

        for r in classify_results:
            if r.result_type == "succeeded":
                resolved_types[r.custom_id.replace("__classify", "")] = parse_classify_result(r.text)
    else:
        echo("  No figures need classification (all have explicit plot types).")

    # 2b. Extraction batch
    extract_requests: list[BatchRequestItem] = []
    for paper_id, figs in paper_figures.items():
        for fig_id, (b64, plot_type) in figs.items():
            key = f"{paper_id}__{fig_id}"
            if plot_type == "auto":
                pt = resolved_types.get(key, "table_image")
            else:
                pt = plot_type
            cid = f"{paper_id}__{fig_id}__extract"
            req = build_extract_batch_request(cid, b64, pt, vision_model)
            extract_requests.append(BatchRequestItem(custom_id=req["custom_id"], params=req["params"]))

    if extract_requests:
        echo(f"\n  Submitting {len(extract_requests)} extraction requests...")
        extract_state_path = state_dir / "vision_extract_state.json"

        if resume:
            saved = BatchState.load(extract_state_path)
            if saved.batch_ids and not saved.completed:
                extract_results = resume_collect(
                    extract_state_path, client=client,
                    poll_interval=poll_interval, echo=echo,
                )
            elif saved.completed:
                from paper_analysis.batch_api import collect_results
                extract_results = collect_results(saved.batch_ids, client=client, echo=echo)
            else:
                extract_results = submit_poll_collect(
                    extract_requests, client=client,
                    state_path=extract_state_path, phase_name="Vision Extract",
                    poll_interval=poll_interval, echo=echo,
                )
        else:
            extract_results = submit_poll_collect(
                extract_requests, client=client,
                state_path=extract_state_path, phase_name="Vision Extract",
                poll_interval=poll_interval, echo=echo,
            )

        # Write extraction results to per-paper JSON files
        for r in extract_results:
            if r.result_type != "succeeded":
                continue
            parts = r.custom_id.replace("__extract", "").split("__", 1)
            if len(parts) != 2:
                continue
            paper_id, fig_id = parts
            prep = next((p for p in preps if p.paper_id == paper_id), None)
            if prep is None:
                continue
            ext_dir = prep.cfg.artifacts.extractions_dir
            ext_dir.mkdir(parents=True, exist_ok=True)
            try:
                extraction = parse_extraction_text(r.text)
                dest = ext_dir / f"{fig_id}.json"
                dest.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                echo(f"  [{paper_id}] vision parse failed for {fig_id}")

    echo("\n  Vision batch phase complete.")


def _run_combined_batch_phase(
    batch_cfg: BatchConfig,
    preps: list[_PaperPrep],
    client: anthropic.Anthropic,
    state_dir: Path,
    resume: bool,
    poll_interval: float,
    echo: callable,
) -> None:
    """Phase 3: combined extraction via batch API with prompt caching and tiered models."""
    echo(f"\n{'='*60}")
    echo("Phase 3: Combined extraction batch")
    echo(f"{'='*60}")

    # Build per-paper content and batch requests
    # Track content for continuation passes
    paper_content: dict[str, str] = {}
    paper_models: dict[str, str] = {}
    requests: list[BatchRequestItem] = []

    tier_counts = {"cheap": 0, "premium": 0}

    for prep in preps:
        tc = effective_text_config(prep.cfg)
        try:
            pages, tables = load_text_artifacts(tc)
        except Exception as exc:
            echo(f"  [{prep.paper_id}] failed to load text artifacts: {exc}")
            continue

        content = build_llm_context(pages, tables, tc.max_context_chars)
        fig_summary = build_figure_summary(prep.cfg.artifacts.extractions_dir)
        if fig_summary:
            content = content + "\n\n" + fig_summary

        model = batch_cfg.model_for_paper(
            prep.paper_id,
            n_pages=prep.n_pages,
            n_figures=len(prep.targets),
            n_pages_with_tables=prep.n_pages_with_tables,
        )
        paper_content[prep.paper_id] = content
        paper_models[prep.paper_id] = model

        if model == batch_cfg.tier.cheap_model:
            tier_counts["cheap"] += 1
        else:
            tier_counts["premium"] += 1

        req = build_combined_batch_request(prep.paper_id, content, model)
        requests.append(BatchRequestItem(custom_id=req["custom_id"], params=req["params"]))

    echo(f"  {len(requests)} papers: {tier_counts['cheap']} cheap, {tier_counts['premium']} premium")

    if not requests:
        echo("  No combined extraction requests to submit.")
        return

    combined_state_path = state_dir / "combined_state.json"

    if resume:
        saved = BatchState.load(combined_state_path)
        if saved.batch_ids and not saved.completed:
            results = resume_collect(
                combined_state_path, client=client,
                poll_interval=poll_interval, echo=echo,
            )
        elif saved.completed:
            from paper_analysis.batch_api import collect_results as _collect
            results = _collect(saved.batch_ids, client=client, echo=echo)
        else:
            results = submit_poll_collect(
                requests, client=client,
                state_path=combined_state_path, phase_name="Combined Extraction",
                poll_interval=poll_interval, echo=echo,
            )
    else:
        results = submit_poll_collect(
            requests, client=client,
            state_path=combined_state_path, phase_name="Combined Extraction",
            poll_interval=poll_interval, echo=echo,
        )

    # Parse results and identify papers needing continuation
    paper_batches: dict[str, CombinedExtractionBatch] = {}
    needs_continuation: dict[str, list[str]] = {}  # paper_id -> extracted IDs so far

    for r in results:
        if r.result_type != "succeeded":
            continue
        paper_id = r.custom_id.replace("__combined", "").split("__continuation_")[0]
        batch_out, needs_cont = parse_batch_result_text(r.text, r.stop_reason)
        if batch_out is None:
            echo(f"  [{paper_id}] combined extraction returned unparseable output")
            continue

        if paper_id in paper_batches:
            existing = paper_batches[paper_id]
            seen = {c.measurement_id for c in existing.candidates}
            for c in batch_out.candidates:
                if c.measurement_id not in seen:
                    existing.candidates.append(c)
                    seen.add(c.measurement_id)
        else:
            paper_batches[paper_id] = batch_out

        if needs_cont:
            ids = [c.measurement_id for c in paper_batches[paper_id].candidates]
            needs_continuation[paper_id] = ids

    # Continuation passes for truncated outputs
    for pass_num in range(1, 4):
        if not needs_continuation:
            break
        echo(f"\n  Continuation pass {pass_num}: {len(needs_continuation)} papers need more data")

        cont_requests: list[BatchRequestItem] = []
        for paper_id, extracted_ids in needs_continuation.items():
            model = paper_models.get(paper_id, batch_cfg.tier.cheap_model)
            content = paper_content.get(paper_id, "")
            req = build_continuation_batch_request(
                paper_id, content, model, extracted_ids, pass_num,
            )
            if req:
                cont_requests.append(BatchRequestItem(custom_id=req["custom_id"], params=req["params"]))

        if not cont_requests:
            break

        cont_state_path = state_dir / f"combined_continuation_{pass_num}_state.json"
        cont_results = submit_poll_collect(
            cont_requests, client=client,
            state_path=cont_state_path, phase_name=f"Continuation Pass {pass_num}",
            poll_interval=poll_interval, echo=echo,
        )

        needs_continuation = {}
        for r in cont_results:
            if r.result_type != "succeeded":
                continue
            paper_id = r.custom_id.split("__continuation_")[0]
            batch_out, needs_cont = parse_batch_result_text(r.text, r.stop_reason)
            if batch_out is None:
                continue

            if paper_id in paper_batches:
                existing = paper_batches[paper_id]
                seen = {c.measurement_id for c in existing.candidates}
                for c in batch_out.candidates:
                    if c.measurement_id not in seen:
                        existing.candidates.append(c)
                        seen.add(c.measurement_id)
            else:
                paper_batches[paper_id] = batch_out

            if needs_cont:
                ids = [c.measurement_id for c in paper_batches[paper_id].candidates]
                needs_continuation[paper_id] = ids

    # Write results to per-paper candidate_combined.json
    for prep in preps:
        batch_out = paper_batches.get(prep.paper_id)
        if batch_out is None:
            continue
        tc = effective_text_config(prep.cfg)
        out_path = tc.output_dir / "candidate_combined.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(batch_out.model_dump_json(indent=2), encoding="utf-8")

    echo(f"\n  Combined extraction complete: {len(paper_batches)} papers produced output")
