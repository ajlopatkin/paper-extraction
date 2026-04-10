from __future__ import annotations

from pathlib import Path

import fitz

from paper_analysis.config import ArtifactsConfig, FigureTarget, PdfConfig


def figure_clip_rect(page: fitz.Page, target: FigureTarget) -> fitz.Rect:
    r = page.rect
    if target.bbox_pdf is not None:
        x0, y0, x1, y1 = target.bbox_pdf
        return fitz.Rect(x0, y0, x1, y1)
    if target.bbox_norm is not None:
        nx0, ny0, nx1, ny1 = target.bbox_norm
        return fitz.Rect(
            r.x0 + nx0 * r.width,
            r.y0 + ny0 * r.height,
            r.x0 + nx1 * r.width,
            r.y0 + ny1 * r.height,
        )
    raise ValueError(f"Figure {target.id!r}: set bbox_pdf or bbox_norm")


def crop_figure_to_png(page: fitz.Page, target: FigureTarget) -> bytes:
    clip = figure_clip_rect(page, target)
    clip.intersect(page.rect)
    if clip.is_empty:
        raise ValueError(f"Figure {target.id!r}: empty clip after intersecting page")
    zoom = target.render_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png")


def extract_all_figures(
    pdf_cfg: PdfConfig,
    targets: list[FigureTarget],
    artifacts: ArtifactsConfig,
) -> dict[str, Path]:
    pdf_path = pdf_cfg.path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = artifacts.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    doc = fitz.open(pdf_path)
    try:
        for t in targets:
            idx = t.page - 1
            if idx < 0 or idx >= doc.page_count:
                raise ValueError(f"Figure {t.id!r}: page {t.page} out of range (1-{doc.page_count})")
            page = doc.load_page(idx)
            png = crop_figure_to_png(page, t)
            dest = out_dir / f"{t.id}.png"
            dest.write_bytes(png)
            written[t.id] = dest
    finally:
        doc.close()

    return written


def list_page_images(pdf_path: Path, page_1based: int) -> list[dict]:
    """Return embedded image info on a page (xref, bbox in PDF coords)."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        idx = page_1based - 1
        if idx < 0 or idx >= doc.page_count:
            raise ValueError(f"page {page_1based} out of range (1-{doc.page_count})")
        page = doc.load_page(idx)
        infos: list[dict] = []
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                infos.append(
                    {
                        "xref": xref,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    }
                )
        return infos
    finally:
        doc.close()


def render_page_png(pdf_path: Path, page_1based: int, dpi: int = 150) -> bytes:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        idx = page_1based - 1
        if idx < 0 or idx >= doc.page_count:
            raise ValueError(f"page {page_1based} out of range (1-{doc.page_count})")
        page = doc.load_page(idx)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()
