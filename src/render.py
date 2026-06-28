"""
render.py -- cited-page rendering + passage highlighting, shared by the front-ends.

Given a cited (file, page) [+ an optional figure block or the snippet text], rasterise the page
to a PIL image with the cited content MARKED: a yellow highlight over the passage, or an amber
box around a figure region. Annotations live only in memory -- the PDF on disk is never touched.

Factored out of app.py so both the Gradio app AND the FastAPI backend (api.py) share ONE renderer
-- the highlight logic can never drift between the two front-ends.
"""
import re
from pathlib import Path

import fitz
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
PDFS_DIR = _ROOT / "data" / "pdfs"
RENDER_DPI = 150            # crisp enough to read a full page in the click-to-open viewer

# Rendered (highlighted) cited pages, cached in memory by (file, page, fig-block, snippet-hash).
_PAGE_CACHE = {}


def phrases(text, n_words=7, max_phrases=10):
    """Break a source chunk into short word-runs to look for on the page.

    We search for several short phrases (not the whole chunk, which wraps across lines and
    rarely matches verbatim) and spread them across the chunk so the highlight covers the
    whole cited passage, not just its first line.
    """
    words = re.findall(r"\S+", text or "")
    groups = [" ".join(words[i:i + n_words]) for i in range(0, len(words), n_words)]
    groups = [g for g in groups if len(g) >= 12]
    if len(groups) <= max_phrases:
        return groups
    step = len(groups) / max_phrases            # evenly sample across the chunk
    return [groups[int(i * step)] for i in range(max_phrases)]


def highlight_rects(page, text, cap=14):
    """Rectangles on `page` where the source text appears (capped, best-effort)."""
    rects = []
    for phrase in phrases(text):
        try:
            rects.extend(page.search_for(phrase))
        except Exception:
            pass
        if len(rects) >= cap:
            break
    return rects[:cap]


def box_figure(page, block_idx):
    """Draw a box around figure `block_idx` on the page, re-deriving the figure region with
    the SAME clustering used at ingest (figures.cluster_image_regions, indexed by block_idx).
    So a figure-sourced answer shows the actual diagram, marked. Returns True if located."""
    try:
        from figures import cluster_image_regions
        regions = cluster_image_regions(page)
        if not (0 <= block_idx < len(regions)):
            return False
        annot = page.add_rect_annot(fitz.Rect(regions[block_idx]))
        annot.set_colors(stroke=(0.91, 0.61, 0.16))   # amber box
        annot.set_border(width=2.5)
        annot.update()
        return True
    except Exception:
        return False


def render_cited_page(source_file, page, snippet_text=None, figure_block=None):
    """Render a cited page to a PIL image with the cited content MARKED.

    For a FIGURE source (`figure_block` set), draw a box around the figure region so the user
    sees the actual diagram; otherwise highlight the cited passage text. Returns (image,
    marked?) -- marked is False when the content could not be located (a plain page is shown).
    Annotations live only in memory; the PDF on disk is never modified.
    """
    if not source_file or not page:
        return None, False
    key = (source_file, page, figure_block, hash(snippet_text or ""))
    if key in _PAGE_CACHE:
        return _PAGE_CACHE[key]
    pdf = PDFS_DIR / source_file
    if not pdf.exists():
        return None, False
    try:
        with fitz.open(str(pdf)) as doc:
            pg = doc[page - 1]                  # page_number is 1-based
            if figure_block is not None:
                marked = box_figure(pg, figure_block)
            else:
                rects = highlight_rects(pg, snippet_text) if snippet_text else []
                for r in rects:
                    annot = pg.add_highlight_annot(r)
                    annot.update()
                marked = bool(rects)
            pix = pg.get_pixmap(dpi=RENDER_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception:
        return None, False
    out = (img, marked)
    _PAGE_CACHE[key] = out
    return out


def snippet(text, limit=300):
    """A compact, single-line-collapsed preview of a source chunk."""
    t = re.sub(r"\s+", " ", text or "").strip()
    return (t[:limit].rstrip() + "…") if len(t) > limit else t
