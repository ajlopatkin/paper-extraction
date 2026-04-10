#!/usr/bin/env python3
"""Render a PDF page to PNG for picking bbox coordinates (see README)."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    p = argparse.ArgumentParser(description="Render a PDF page to PNG")
    p.add_argument("pdf", type=Path, help="Path to PDF")
    p.add_argument("--page", "-p", type=int, required=True, help="1-based page number")
    p.add_argument("--out", "-o", type=Path, required=True, help="Output PNG path")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI (default 150)")
    args = p.parse_args()
    pdf = args.pdf
    if not pdf.is_file():
        raise SystemExit(f"not found: {pdf}")
    doc = fitz.open(pdf)
    try:
        idx = args.page - 1
        if idx < 0 or idx >= doc.page_count:
            raise SystemExit(f"page {args.page} out of range (1-{doc.page_count})")
        page = doc.load_page(idx)
        r = page.rect
        zoom = args.dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(pix.tobytes("png"))
        w_pt, h_pt = r.width, r.height
        print(f"Wrote {args.out} ({pix.width}x{pix.height} px at {args.dpi} dpi)")
        print(f"Page size (PDF points): width={w_pt:.1f} height={h_pt:.1f}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
