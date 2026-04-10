from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from paper_analysis.config import PdfConfig, TextPipelineConfig


def _page_indices(doc: fitz.Document, pages_1based: list[int] | None) -> list[int]:
    n = doc.page_count
    if pages_1based is None:
        return list(range(n))
    out: list[int] = []
    for p in pages_1based:
        i = p - 1
        if 0 <= i < n:
            out.append(i)
    return sorted(set(out))


def extract_page_texts(pdf_path: Path, text_cfg: TextPipelineConfig) -> list[dict[str, Any]]:
    """Plain text per page (selectable PDF content)."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        pages_out: list[dict[str, Any]] = []
        for idx in _page_indices(doc, text_cfg.pages):
            page = doc.load_page(idx)
            t = page.get_text("text")
            pages_out.append(
                {
                    "page": idx + 1,
                    "char_count": len(t),
                    "text": t,
                }
            )
        return pages_out
    finally:
        doc.close()


def _table_to_rows(tab: Any) -> list[list[str | None]]:
    try:
        raw = tab.extract()
    except (AttributeError, TypeError, ValueError):
        return []
    if not raw:
        return []
    rows: list[list[str | None]] = []
    for row in raw:
        if row is None:
            continue
        cells: list[str | None] = []
        for c in row:
            if c is None:
                cells.append(None)
            else:
                s = str(c).strip()
                cells.append(s if s else None)
        rows.append(cells)
    return rows


def _bbox_to_list(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    try:
        r = fitz.Rect(bbox)
        return [r.x0, r.y0, r.x1, r.y1]
    except (TypeError, ValueError):
        return None


def extract_native_tables(pdf_path: Path, text_cfg: TextPipelineConfig) -> list[dict[str, Any]]:
    """Detect vector/text tables via PyMuPDF find_tables (empty list if unsupported or none)."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    tables_out: list[dict[str, Any]] = []
    global_idx = 0
    try:
        for idx in _page_indices(doc, text_cfg.pages):
            page = doc.load_page(idx)
            try:
                finder = page.find_tables()
            except AttributeError:
                continue
            table_list = getattr(finder, "tables", None)
            if table_list is None:
                try:
                    table_list = list(finder)
                except TypeError:
                    table_list = []
            if not table_list:
                continue
            for local_i, tab in enumerate(table_list):
                rows = _table_to_rows(tab)
                bbox = _bbox_to_list(getattr(tab, "bbox", None))
                tables_out.append(
                    {
                        "id": f"t{global_idx:03d}",
                        "page": idx + 1,
                        "index_on_page": local_i,
                        "bbox_pdf": bbox,
                        "n_rows": len(rows),
                        "n_cols": max((len(r) for r in rows), default=0),
                        "rows": rows,
                    }
                )
                global_idx += 1
        return tables_out
    finally:
        doc.close()


def tables_to_markdown(tables: list[dict[str, Any]]) -> str:
    """Render detected tables as markdown for LLM context."""
    parts: list[str] = []
    for t in tables:
        rows = t.get("rows") or []
        if not rows:
            continue
        header = f"### Table id={t['id']} page={t['page']} (native PDF table)\n\n"
        lines: list[str] = []
        for row in rows:
            cells = [(c if c is not None else "") for c in row]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        parts.append(header + "\n".join(lines))
    return "\n\n".join(parts) if parts else "(No native tables detected by find_tables.)"


def build_llm_context(
    pages: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    max_context_chars: int,
) -> str:
    """Assemble markdown tables + body text for the measurement-extraction LLM."""
    md = tables_to_markdown(tables)
    body, truncated = concatenate_page_texts(pages, max_context_chars)
    trunc_note = "\n\n(NOTE: body text was truncated for token limits.)\n" if truncated else ""
    return f"{md}\n\n--- BODY TEXT ---{trunc_note}\n\n{body}"


def concatenate_page_texts(pages: list[dict[str, Any]], max_chars: int) -> tuple[str, bool]:
    """Join page texts; if over max_chars, keep start + end with a truncation marker."""
    chunks = [f"--- Page {p['page']} ---\n{p['text']}" for p in pages]
    full = "\n\n".join(chunks)
    if len(full) <= max_chars:
        return full, False
    half = max_chars // 2 - 200
    head = full[:half]
    tail = full[-half:]
    return head + "\n\n...[CONTEXT TRUNCATED]...\n\n" + tail, True


def run_text_dump(pdf_cfg: PdfConfig, text_cfg: TextPipelineConfig) -> tuple[Path, Path]:
    """Write pages.json and tables.json under text_cfg.output_dir."""
    out_dir = text_cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_cfg.path
    pages = extract_page_texts(pdf_path, text_cfg)
    tables = extract_native_tables(pdf_path, text_cfg)
    pages_path = out_dir / "pages.json"
    tables_path = out_dir / "tables.json"
    pages_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    tables_path.write_text(
        json.dumps(tables, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return pages_path, tables_path


def load_text_artifacts(text_cfg: TextPipelineConfig) -> tuple[list[dict], list[dict]]:
    out_dir = text_cfg.output_dir
    p = out_dir / "pages.json"
    t = out_dir / "tables.json"
    if not p.is_file() or not t.is_file():
        raise FileNotFoundError(
            f"Missing {p.name} or {t.name} under {out_dir}; run `paper-analysis extract-text` first."
        )
    pages = json.loads(p.read_text(encoding="utf-8"))
    tables = json.loads(t.read_text(encoding="utf-8"))
    return pages, tables
