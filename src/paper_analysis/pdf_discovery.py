from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from paper_analysis.config import AutoDiscoverConfig, FigureTarget, PocConfig

# Image blocks appear in get_text("dict") only when this flag is set (PyMuPDF docs).
_TEXT_FLAGS_IMAGES = int(getattr(fitz, "TEXT_PRESERVE_IMAGES", 4096))
# Vector / path blocks (type 3) require TEXT_COLLECT_VECTORS in addition for dict extraction.
_TEXT_FLAGS_IMG_AND_VEC = _TEXT_FLAGS_IMAGES | int(getattr(fitz, "TEXT_COLLECT_VECTORS", 2048))


def _rect_area(r: fitz.Rect) -> float:
    return max(0.0, r.width) * max(0.0, r.height)


def _intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _rect_area(a) + _rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def _merge_pass(rects: list[fitz.Rect], iou_threshold: float) -> list[fitz.Rect]:
    """Single pass: merge first rect with any others that overlap enough; repeat on remainder."""
    rects = [fitz.Rect(r) for r in rects]
    out: list[fitz.Rect] = []
    while rects:
        cur = rects.pop(0)
        i = 0
        while i < len(rects):
            other = rects[i]
            if _iou(cur, other) >= iou_threshold:
                cur |= other
                rects.pop(i)
            else:
                i += 1
        out.append(cur)
    return out


def merge_overlapping_rects(rects: list[fitz.Rect], iou_threshold: float) -> list[fitz.Rect]:
    """Merge rectangles until stable (handles transitive chains)."""
    if not rects:
        return []
    prev: list[fitz.Rect] | None = None
    cur = [fitz.Rect(r) for r in rects]
    while prev is None or len(cur) != len(prev):
        prev = cur
        cur = _merge_pass(cur, iou_threshold)
    return cur


def _collect_image_rects_xref(page: fitz.Page) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        xref = img[0]
        for r in page.get_image_rects(xref):
            rects.append(fitz.Rect(r))
    return rects


def _collect_image_rects_textpage(page: fitz.Page) -> list[fitz.Rect]:
    """Image block bboxes from the text layer (often populated when xref rects are empty)."""
    rects: list[fitz.Rect] = []
    for fmt in ("dict", "rawdict"):
        try:
            dl = page.get_text(fmt, flags=_TEXT_FLAGS_IMAGES)
        except TypeError:
            dl = page.get_text(fmt)
        if not isinstance(dl, dict):
            continue
        for block in dl.get("blocks", []):
            if block.get("type") != 1:
                continue
            bb = block.get("bbox")
            if not bb or len(bb) < 4:
                continue
            r = fitz.Rect(bb[0], bb[1], bb[2], bb[3])
            if not r.is_empty:
                rects.append(r)
    return rects


def _collect_image_rects_blocks(page: fitz.Page) -> list[fitz.Rect]:
    """Image rows from get_text('blocks') — sometimes lists placements dict/rawdict miss."""
    rects: list[fitz.Rect] = []
    try:
        blocks = page.get_text("blocks", flags=_TEXT_FLAGS_IMAGES)
    except TypeError:
        blocks = page.get_text("blocks")
    if not isinstance(blocks, list):
        return rects
    for b in blocks:
        if not isinstance(b, (list, tuple)) or len(b) < 7:
            continue
        x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        block_type = b[6]
        if block_type != 1:
            continue
        r = fitz.Rect(x0, y0, x1, y1)
        if not r.is_empty:
            rects.append(r)
    return rects


def _collect_image_rects(page: fitz.Page) -> tuple[list[fitz.Rect], int, int, int]:
    xref_rects = _collect_image_rects_xref(page)
    dict_rects = _collect_image_rects_textpage(page)
    blocks_rects = _collect_image_rects_blocks(page)
    combined = [fitz.Rect(r) for r in xref_rects + dict_rects + blocks_rects]
    return combined, len(xref_rects), len(dict_rects), len(blocks_rects)


def _collect_vector_block_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Type-3 vector blocks from textpage (grouped path geometry)."""
    rects: list[fitz.Rect] = []
    try:
        dl = page.get_text("dict", flags=_TEXT_FLAGS_IMG_AND_VEC)
    except (TypeError, ValueError):
        dl = page.get_text("dict")
    if not isinstance(dl, dict):
        return rects
    for block in dl.get("blocks", []):
        if block.get("type") != 3:
            continue
        bb = block.get("bbox")
        if not bb or len(bb) < 4:
            continue
        r = fitz.Rect(bb[0], bb[1], bb[2], bb[3])
        if not r.is_empty:
            rects.append(r)
    return rects


def _collect_filtered_drawing_rects(
    page: fitz.Page, page_area: float, disc: AutoDiscoverConfig
) -> list[fitz.Rect]:
    """Per-path bboxes from get_drawings(); filtered and capped for clustering."""
    try:
        drawings = page.get_drawings()
    except (RuntimeError, ValueError, TypeError):
        return []
    min_a = disc.min_drawing_primitive_area_frac * page_area
    raw: list[fitz.Rect] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        rect = fitz.Rect(r)
        if rect.is_empty or _rect_area(rect) < min_a:
            continue
        raw.append(rect)
    min_eff = min_a
    while len(raw) > disc.max_drawing_primitives and min_eff < page_area * 0.08:
        min_eff *= 1.35
        raw = [r for r in raw if _rect_area(r) >= min_eff]
    if len(raw) > disc.max_drawing_primitives:
        raw = raw[: disc.max_drawing_primitives]
    return raw


def cluster_rects_by_gap(rects: list[fitz.Rect], gap: float, page_rect: fitz.Rect) -> list[fitz.Rect]:
    """Union nearby rectangles (same chart made of many strokes)."""
    rects = [fitz.Rect(r) for r in rects if not r.is_empty]
    if not rects:
        return []
    n = len(rects)
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    def expanded(r: fitz.Rect) -> fitz.Rect:
        return fitz.Rect(r.x0 - gap, r.y0 - gap, r.x1 + gap, r.y1 + gap)

    for i in range(n):
        ei = expanded(rects[i])
        for j in range(i + 1, n):
            if (
                _intersection_area(rects[i], rects[j]) > 0
                or _intersection_area(ei, rects[j]) > 0
                or _intersection_area(expanded(rects[j]), rects[i]) > 0
            ):
                union(i, j)

    groups: dict[int, list[fitz.Rect]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(rects[i])

    out: list[fitz.Rect] = []
    for group in groups.values():
        u = fitz.Rect(group[0])
        for r in group[1:]:
            u |= r
        u.intersect(page_rect)
        if not u.is_empty:
            out.append(u)
    return out


def _vector_figure_clusters(
    page: fitz.Page, disc: AutoDiscoverConfig
) -> tuple[list[fitz.Rect], int, int, int]:
    """Clusters from vector blocks + drawing paths (journal-style plots)."""
    pr = page.rect
    page_area = _rect_area(pr)
    if page_area <= 0:
        return [], 0, 0, 0
    vb = _collect_vector_block_rects(page)
    dr = _collect_filtered_drawing_rects(page, page_area, disc)
    primitives = [fitz.Rect(r) for r in vb + dr]
    if not primitives:
        return [], len(vb), len(dr), 0
    gap = disc.drawing_cluster_gap_frac * min(pr.width, pr.height)
    clusters = cluster_rects_by_gap(primitives, gap, pr)
    return clusters, len(vb), len(dr), len(clusters)


def _expand_rect(rect: fitz.Rect, page: fitz.Page, padding_frac: float) -> fitz.Rect:
    pr = page.rect
    span = min(pr.width, pr.height)
    pad = padding_frac * span
    r = fitz.Rect(rect)
    r.x0 -= pad
    r.y0 -= pad
    r.x1 += pad
    r.y1 += pad
    r.intersect(pr)
    return r


def _pages_to_scan(doc: fitz.Document, pages: list[int] | None) -> list[int]:
    n = doc.page_count
    if pages is None:
        return list(range(n))
    out: list[int] = []
    for p in pages:
        idx = p - 1
        if 0 <= idx < n:
            out.append(idx)
    return sorted(set(out))


@dataclass
class PageDiscoveryStats:
    page_1based: int
    n_xref_rects: int
    n_image_blocks: int
    n_blocks_rows: int
    n_vector_blocks: int
    n_drawing_primitives: int
    n_vector_clusters: int
    n_raw_total: int
    n_after_duplicate_merge: int
    n_after_merge: int
    n_after_filter: int


def discover_raw_rects_on_page(
    page: fitz.Page, disc: AutoDiscoverConfig, *, stats: PageDiscoveryStats | None = None
) -> list[fitz.Rect]:
    pr = page.rect
    page_area = _rect_area(pr)
    if page_area <= 0:
        if stats:
            stats.n_xref_rects = 0
            stats.n_image_blocks = 0
            stats.n_blocks_rows = 0
            stats.n_vector_blocks = 0
            stats.n_drawing_primitives = 0
            stats.n_vector_clusters = 0
            stats.n_raw_total = 0
            stats.n_after_duplicate_merge = 0
            stats.n_after_merge = 0
            stats.n_after_filter = 0
        return []

    rects, nx, nd, nb = _collect_image_rects(page)
    if disc.include_vector_graphics:
        vec_clusters, nvb, ndraw, nvcl = _vector_figure_clusters(page, disc)
        rects = rects + vec_clusters
    else:
        nvb, ndraw, nvcl = 0, 0, 0
    if stats:
        stats.n_xref_rects = nx
        stats.n_image_blocks = nd
        stats.n_blocks_rows = nb
        stats.n_vector_blocks = nvb
        stats.n_drawing_primitives = ndraw
        stats.n_vector_clusters = nvcl
        stats.n_raw_total = len(rects)

    deduped = merge_overlapping_rects(rects, disc.duplicate_merge_iou)
    if disc.merge_fragments:
        merged = merge_overlapping_rects(deduped, disc.merge_iou_threshold)
    else:
        merged = deduped
    if stats:
        stats.n_after_duplicate_merge = len(deduped)
        stats.n_after_merge = len(merged)

    min_w = disc.min_width_frac * pr.width
    min_h = disc.min_height_frac * pr.height
    min_area = disc.min_area_frac * page_area

    kept: list[fitz.Rect] = []
    for r in merged:
        r = fitz.Rect(r)
        r.intersect(pr)
        if r.is_empty:
            continue
        if r.width < min_w or r.height < min_h:
            continue
        if _rect_area(r) < min_area:
            continue
        kept.append(_expand_rect(r, page, disc.padding_frac))

    kept.sort(key=lambda rr: (rr.y0, rr.x0))
    if disc.max_figures_per_page is not None and len(kept) > disc.max_figures_per_page:
        kept = sorted(kept, key=_rect_area, reverse=True)[: disc.max_figures_per_page]
        kept.sort(key=lambda rr: (rr.y0, rr.x0))

    if stats:
        stats.n_after_filter = len(kept)
    return kept


def _format_figure_id(template: str, page_1based: int, index_1based: int) -> str:
    return template.format(page=page_1based, index=index_1based)


def discover_figure_targets(pdf_path: Path, disc: AutoDiscoverConfig) -> list[FigureTarget]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    targets: list[FigureTarget] = []
    try:
        page_indices = _pages_to_scan(doc, disc.pages)
        for idx in page_indices:
            page = doc.load_page(idx)
            rects = discover_raw_rects_on_page(page, disc)
            page_1 = idx + 1
            for i, rect in enumerate(rects, start=1):
                fid = _format_figure_id(disc.id_template, page_1, i)
                targets.append(
                    FigureTarget(
                        id=fid,
                        page=page_1,
                        plot_type=disc.default_plot_type,
                        bbox_pdf=[rect.x0, rect.y0, rect.x1, rect.y1],
                        bbox_norm=None,
                        render_dpi=disc.render_dpi,
                    )
                )
    finally:
        doc.close()

    return targets


def run_discovery_diagnostics(pdf_path: Path, disc: AutoDiscoverConfig) -> list[PageDiscoveryStats]:
    """Per-page counts: xref rects, dict/image-block rects, after merge, after size filter."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out: list[PageDiscoveryStats] = []
    doc = fitz.open(pdf_path)
    try:
        for idx in _pages_to_scan(doc, disc.pages):
            page = doc.load_page(idx)
            st = PageDiscoveryStats(
                page_1based=idx + 1,
                n_xref_rects=0,
                n_image_blocks=0,
                n_blocks_rows=0,
                n_vector_blocks=0,
                n_drawing_primitives=0,
                n_vector_clusters=0,
                n_raw_total=0,
                n_after_duplicate_merge=0,
                n_after_merge=0,
                n_after_filter=0,
            )
            discover_raw_rects_on_page(page, disc, stats=st)
            out.append(st)
    finally:
        doc.close()
    return out


def format_figures_yaml(targets: list[FigureTarget]) -> str:
    """Serialize figure targets as a `figures:` YAML block."""
    lines = ["figures:"]
    for t in targets:
        lines.append(f"  - id: {t.id}")
        lines.append(f"    page: {t.page}")
        lines.append(f"    plot_type: {t.plot_type}")
        lines.append(
            f"    bbox_pdf: [{t.bbox_pdf[0]:.2f}, {t.bbox_pdf[1]:.2f}, {t.bbox_pdf[2]:.2f}, {t.bbox_pdf[3]:.2f}]"
        )
        lines.append(f"    render_dpi: {t.render_dpi}")
    return "\n".join(lines) + "\n"


def discover_bboxes_yaml_snippet(pdf_path: Path, disc: AutoDiscoverConfig) -> str:
    """Human-readable YAML for manual paste or review."""
    return format_figures_yaml(discover_figure_targets(pdf_path, disc))


def apply_relaxed_thresholds(disc: AutoDiscoverConfig) -> AutoDiscoverConfig:
    """Looser size filters for debugging PDFs with small or oddly placed images."""
    return disc.model_copy(
        update={
            "min_area_frac": min(disc.min_area_frac, 0.002),
            "min_width_frac": min(disc.min_width_frac, 0.015),
            "min_height_frac": min(disc.min_height_frac, 0.015),
        }
    )


def get_effective_figure_targets(cfg: PocConfig) -> list[FigureTarget]:
    """Manual `figures` or auto-discovered targets, depending on config."""
    ad = cfg.auto_discover
    if ad is not None and ad.enabled:
        auto = discover_figure_targets(cfg.pdf.path, ad)
        if ad.strategy == "append_to_manual":
            return list(cfg.figures) + auto
        return auto
    return list(cfg.figures)
