"""
figures.py -- Phase 2: turn figures into searchable text chunks.

A logical figure is often stored as MANY scattered image fragments (finding #5:
p602 = 28 pieces). So we cluster image fragments into regions, render each region,
bundle its nearby caption/heading as context, and have the VLM describe it. The
caption text becomes a `figure_caption` chunk in the same unified text-space as
prose and tables -- the rest of the pipeline never sees a pixel.

Scope: raster figures (charts, diagrams, embedded images, equation images).
Vector-drawn figures (flowcharts made of lines, e.g. p9) have no image blocks and
are NOT detected here -- a separate get_drawings()-based path, deferred.
"""

from ingest import Chunk, render_table_image
from vlm import transcribe_figure


def _expand(b, m):
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _merge_overlapping(regions):
    """Merge any regions whose bboxes overlap (fixes the p808 over-segmentation,
    where two fragment clusters' union boxes overlap but the fragments didn't)."""
    changed = True
    while changed:
        changed = False
        out = []
        for r in regions:
            for i, o in enumerate(out):
                if _overlap(r, o):
                    out[i] = _union([r, o])
                    changed = True
                    break
            else:
                out.append(r)
        regions = out
    return regions


def cluster_image_regions(page, gap=24, min_size=80):
    """Group a page's image fragments into logical figure regions.

    Two fragments join if their bboxes -- each expanded by `gap` points -- overlap,
    transitively (connected components). Each component's union bbox is a region;
    regions smaller than `min_size` in either dimension are dropped as inline glyphs
    / decorations (e.g. the math-glyph images on p540), not figures.
    """
    boxes = [b["bbox"] for b in page.get_text("dict")["blocks"] if b.get("type") == 1]
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(_expand(boxes[i], gap), _expand(boxes[j], gap)):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])

    regions = [_union(g) for g in groups.values()]
    regions = [r for r in regions
               if r[2] - r[0] >= min_size and r[3] - r[1] >= min_size]
    return _merge_overlapping(regions)


def figure_context(page, region, margin=60):
    """Nearby text to ground the caption: the text block(s) just BELOW the region
    (where "Figure N. ..." captions live) that horizontally overlap it. Returns a
    joined string (may be empty)."""
    x0, y0, x1, y1 = region
    found = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:                 # text blocks only
            continue
        bx0, by0, bx1, by1 = b["bbox"]
        if 0 <= by0 - y1 <= margin and bx0 < x1 and bx1 > x0:   # just below, overlapping
            txt = " ".join(s["text"] for ln in b["lines"] for s in ln["spans"]).strip()
            if txt:
                found.append((by0, txt))
    found.sort()
    return " ".join(t for _, t in found)


def extract_figure_chunks(page, page_number, source_file):
    """Cluster -> render -> caption -> chunk. One figure_caption Chunk per region."""
    chunks = []
    for fig_idx, region in enumerate(cluster_image_regions(page)):
        image_path = render_table_image(page, region, path="/tmp/_vlm_figure.png")
        context = figure_context(page, region)
        description = transcribe_figure(image_path, context=context)
        text = f"{context}\n{description}".strip() if context else description
        chunks.append(Chunk(
            text=text,
            source_file=source_file,
            page_number=page_number,
            source_type="figure_caption",
            block_idx=fig_idx,                 # figure index on the page
        ))
    return chunks


if __name__ == "__main__":
    import sys
    import fitz
    from config import PDF_PATH

    pno = int(sys.argv[1]) if len(sys.argv) > 1 else 602
    page = fitz.open(PDF_PATH)[pno - 1]
    print(f"PAGE {pno}: {len(cluster_image_regions(page))} figure region(s)\n")
    for ch in extract_figure_chunks(page, pno, "mil-std-810h.pdf"):
        print(f"[{ch.chunk_id}]")
        print(ch.text)
        print()
