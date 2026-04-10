from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from paper_analysis.config import (
    AutoDiscoverConfig,
    PocConfig,
    effective_export_config,
    effective_text_config,
    resolve_figure_targets,
)
from paper_analysis.evaluate import format_report, inspect_measurements, run_evaluation
from paper_analysis.export import export_workbook
from paper_analysis.pdf_discovery import (
    apply_relaxed_thresholds,
    discover_figure_targets,
    format_figures_yaml,
    run_discovery_diagnostics,
)
from paper_analysis.pdf_figures import extract_all_figures, list_page_images
from paper_analysis.pdf_text_tables import build_llm_context, load_text_artifacts, run_text_dump
from paper_analysis.runs_analyze_llm import run_runs_analysis
from paper_analysis.text_analyze_llm import run_text_analysis
from paper_analysis.vision.base import build_vision_client

load_dotenv()

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load_cfg(config: Path) -> PocConfig:
    return PocConfig.load(config.resolve())


@app.command("extract-figures")
def extract_figures_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
    figures_config: Path | None = typer.Option(
        None,
        "--figures-config",
        "-f",
        exists=True,
        readable=True,
        help="YAML with top-level `figures:` only (e.g. config/discovered_figures.yaml from discover-bboxes). Overrides poc figures/auto_discover.",
    ),
) -> None:
    """Crop figure regions from the PDF into artifacts/figures."""
    cfg = _load_cfg(config)
    targets = resolve_figure_targets(cfg, figures_config=figures_config)
    if figures_config is not None:
        typer.echo(f"Using figures from {figures_config.resolve()} (poc `figures` / auto_discover ignored).")
    if not targets:
        typer.echo(
            "No figure targets: use --figures-config path/to/discovered_figures.yaml, "
            "or add `figures:` to poc.yaml, or set `auto_discover.enabled: true`."
        )
        raise typer.Exit(1)
    written = extract_all_figures(cfg.pdf, targets, cfg.artifacts)
    for fid, p in written.items():
        typer.echo(f"wrote {p}")


@app.command("run-vision")
def run_vision_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
    figures_config: Path | None = typer.Option(
        None,
        "--figures-config",
        "-f",
        exists=True,
        readable=True,
        help="Same as extract-figures: use this `figures:` list for ids and plot_type.",
    ),
) -> None:
    """Call the vision API for each configured figure and write JSON extractions."""
    cfg = _load_cfg(config)
    provider = cfg.vision.provider or os.environ.get("PAPER_ANALYSIS_VISION_PROVIDER", "anthropic")
    client = build_vision_client(provider=provider, model=cfg.vision.model)
    out_dir = cfg.artifacts.extractions_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = cfg.artifacts.figures_dir
    targets = resolve_figure_targets(cfg, figures_config=figures_config)
    if figures_config is not None:
        typer.echo(f"Using figures from {figures_config.resolve()} (poc `figures` / auto_discover ignored).")
    if not targets:
        typer.echo(
            "No figure targets; use --figures-config, or configure poc.yaml `figures` / auto_discover."
        )
        raise typer.Exit(1)
    for fig in targets:
        png_path = fig_dir / f"{fig.id}.png"
        if not png_path.is_file():
            raise typer.BadParameter(f"Missing cropped image {png_path}; run extract-figures first")
        data = png_path.read_bytes()
        result = client.extract_figure(data, fig.plot_type, max_retries=1)
        dest = out_dir / f"{fig.id}.json"
        dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"wrote {dest}")


@app.command("extract-text")
def extract_text_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
) -> None:
    """Extract selectable page text and native PDF tables to artifacts/text (no LLM)."""
    cfg = _load_cfg(config)
    tc = effective_text_config(cfg)
    pages_path, tables_path = run_text_dump(cfg.pdf, tc)
    typer.echo(f"wrote {pages_path}")
    typer.echo(f"wrote {tables_path}")


@app.command("analyze-text")
def analyze_text_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
) -> None:
    """LLM pass: read pages.json + tables.json and write candidate_measurements.json (uses API key)."""
    cfg = _load_cfg(config)
    tc = effective_text_config(cfg)
    try:
        pages, tables = load_text_artifacts(tc)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    content = build_llm_context(pages, tables, tc.max_context_chars)
    provider = cfg.vision.provider or os.environ.get("PAPER_ANALYSIS_VISION_PROVIDER", "anthropic")
    model = tc.llm_model or cfg.vision.model
    batch = run_text_analysis(content, provider=provider, model=model, max_retries=1)
    out = tc.output_dir / "candidate_measurements.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"wrote {out} ({len(batch.candidates)} candidates)")


@app.command("extract-runs")
def extract_runs_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
) -> None:
    """LLM pass: read pages.json + tables.json and write candidate_runs.json (uses API key)."""
    cfg = _load_cfg(config)
    tc = effective_text_config(cfg)
    try:
        pages, tables = load_text_artifacts(tc)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    content = build_llm_context(pages, tables, tc.max_context_chars)
    provider = cfg.vision.provider or os.environ.get("PAPER_ANALYSIS_VISION_PROVIDER", "anthropic")
    model = tc.llm_model or cfg.vision.model
    batch = run_runs_analysis(content, provider=provider, model=model, max_retries=1)
    out = tc.output_dir / "candidate_runs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"wrote {out} ({len(batch.candidates)} runs)")


@app.command("inspect-sheet")
def inspect_sheet_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
) -> None:
    """Print column names and rows matching evaluation filters (no API calls)."""
    cfg = _load_cfg(config)
    inspect_measurements(cfg)


@app.command("evaluate")
def evaluate_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
) -> None:
    """Compare extraction JSON to the ground-truth spreadsheet."""
    cfg = _load_cfg(config)
    results = run_evaluation(cfg)
    if not results:
        typer.echo("No evaluation.comparisons configured; nothing to compare.")
        raise typer.Exit(0)
    typer.echo(format_report(results))


@app.command("export-results")
def export_results_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
    figures_config: Path | None = typer.Option(
        None,
        "--figures-config",
        "-f",
        exists=True,
        readable=True,
        help="If set, only include vision JSON whose stem is in this `figures:` list.",
    ),
) -> None:
    """Write Extracted_Measurements.xlsx: ground-truth-style rows + extraction metadata (no API calls)."""
    cfg = _load_cfg(config)
    allowed: set[str] | None = None
    if figures_config is not None:
        targets = resolve_figure_targets(cfg, figures_config=figures_config)
        allowed = {t.id for t in targets}
        typer.echo(f"Filtering vision extractions to {len(allowed)} ids from {figures_config.resolve()}.")
    ec = effective_export_config(cfg)
    out = export_workbook(cfg, allowed_figure_ids=allowed, export_cfg=ec)
    typer.echo(f"wrote {out}")


@app.command("discover-bboxes")
def discover_bboxes_cmd(
    config: Path = typer.Option(Path("config/poc.yaml"), "--config", "-c", exists=True),
    pdf: Path | None = typer.Option(
        None,
        "--pdf",
        help="Override PDF path from config (still uses discovery settings from YAML when present)",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Write YAML here (default: discovered_figures.yaml next to --config, e.g. config/discovered_figures.yaml).",
    ),
    stdout_only: bool = typer.Option(
        False,
        "--stdout",
        help="Print YAML to stdout only (no file write; overrides default --out path).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Per-page xref vs dict counts and filter stages"),
    relaxed: bool = typer.Option(
        False,
        "--relaxed",
        help="Looser min-size filters (use if everything is still filtered out)",
    ),
) -> None:
    """Write suggested figure entries to discovered_figures.yaml (next to --config) unless --stdout."""
    cfg = _load_cfg(config)
    disc = cfg.auto_discover if cfg.auto_discover is not None else AutoDiscoverConfig()
    if relaxed:
        disc = apply_relaxed_thresholds(disc)
    path = pdf if pdf is not None else cfg.pdf.path
    if verbose:
        typer.echo(
            f"Discovery on {path} (pages={disc.pages or 'all'}, "
            f"merge_fragments={disc.merge_fragments}, "
            f"include_vector_graphics={disc.include_vector_graphics})",
            err=True,
        )
        for st in run_discovery_diagnostics(path, disc):
            typer.echo(
                f"  page {st.page_1based}: raw={st.n_raw_total} "
                f"(xref={st.n_xref_rects} dict={st.n_image_blocks} blocks={st.n_blocks_rows} "
                f"vec_blocks={st.n_vector_blocks} draw_prim={st.n_drawing_primitives} "
                f"vec_clust={st.n_vector_clusters}) "
                f"after_dedupe={st.n_after_duplicate_merge} after_merge={st.n_after_merge} "
                f"kept={st.n_after_filter}",
                err=True,
            )
    targets = discover_figure_targets(path, disc)
    snippet = format_figures_yaml(targets)
    if not targets:
        typer.echo(
            "No figure regions found. Try: paper-analysis discover-bboxes ... --verbose\n"
            "If xref_rects and image_blocks are both 0, the PDF may use vector-only figures "
            "(no embedded bitmaps in the text layer). If counts drop only after "
            "after_size_filter, try --relaxed or lower min_*_frac in auto_discover.",
            err=True,
        )
    if stdout_only:
        typer.echo(snippet.rstrip("\n"))
    else:
        dest = out if out is not None else (config.resolve().parent / "discovered_figures.yaml")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(snippet, encoding="utf-8")
        typer.echo(f"wrote {dest}")


@app.command("list-images")
def list_images_cmd(
    pdf: Path = typer.Argument(..., exists=True, readable=True),
    page: int = typer.Option(..., "--page", "-p", help="1-based page number"),
) -> None:
    """List embedded images on a PDF page (xref and bbox in PDF coordinates)."""
    infos = list_page_images(pdf, page)
    if not infos:
        typer.echo("no embedded images on this page")
        return
    for item in infos:
        typer.echo(f"xref={item['xref']} bbox={item['bbox']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
