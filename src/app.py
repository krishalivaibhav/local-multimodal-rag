"""
app.py -- a local Gradio chat UI over the RAG engine.

A thin front-end: every question goes through answer_question() (retrieval ->
grounded prompt -> local LLM -> citation resolution). The reply shows the grounded
answer, plus a "Cited sources" panel that, for each source the answer actually used:
  - shows a readable TEXT SNIPPET of the source (verify without opening the PDF), and
  - renders the cited PDF page with the cited passage HIGHLIGHTED (so the eye lands on
    the exact spot), as a compact thumbnail that opens to a full-page reader on click.
Auditability is the product: the user can SEE the grounding, not take it on trust.
Each answer carries native 👍/👎 thumbs whose votes are logged locally
(data/feedback/feedback.jsonl) -- the on-box feedback loop for an air-gapped deploy.

Runs at FULL retrieval strength (hybrid + parent-tables). Fully local -- local
store + local Ollama + local PDFs, no public share link.

Run:  cd src && python app.py     # then open http://127.0.0.1:7860
"""
import os

# Full retrieval strength by default: dense+BM25 hybrid + parent-tables + cross-encoder rerank
# (the stack behind finding #40 -- MULTIDOC citation 0.55->0.82, HARD value 0.70->0.75). Must be
# set BEFORE answer/query/config import (config reads these at import time). setdefault so an
# explicit env var still wins. RAG_RERANK adds ~3 s/query on the M2 CPU (near-zero on GPU/MPS).
os.environ.setdefault("RAG_HYBRID", "1")
os.environ.setdefault("RAG_PARENT_TABLES", "1")
os.environ.setdefault("RAG_RERANK", "1")
os.environ.setdefault("RAG_DEDUP_REVISIONS", "1")   # cross-revision citation dedup (finding #44)

import json
import re
import time
from pathlib import Path

import fitz
import gradio as gr
from PIL import Image

from answer import answer_question, answer_stream

_ROOT = Path(__file__).resolve().parent.parent
_PDFS_DIR = _ROOT / "data" / "pdfs"

# Feedback log -- local, under data/ (gitignored like the store/PDFs).
_FEEDBACK_DIR = _ROOT / "data" / "feedback"
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_LOG = _FEEDBACK_DIR / "feedback.jsonl"

# Rendered (highlighted) cited pages, cached in memory by (file, page, snippet-hash).
_PAGE_CACHE = {}
_RENDER_DPI = 150            # crisp enough to read a full page in the click-to-open viewer

_SOURCES_EMPTY = "_Ask a question — the sources behind the answer will appear here, " \
                 "with the cited passage highlighted on the page._"

# Document scope (mode 2): constrain retrieval to ONE doc. "All documents" (default) searches the
# whole store; picking a doc scopes retrieval to it -- which fixes the near-duplicate wrong-doc
# citation leak (810G vs 810H) and prunes the search space (faster).
_ALL_DOCS = "All documents"
_DOC_CHOICES = [_ALL_DOCS] + sorted(p.name for p in _PDFS_DIR.glob("*.pdf"))

EXAMPLES = [
    "In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of a rain test?",
    "In NASA's project-type classification (Table 3.11-1), what kind of mission is Type A?",
    "What kind of test stands are listed among facilities to consider in Table 6.1-2?",
]

# Greeting / chit-chat: a grounded QA system, not a chatbot, so a bare "hello" abstains.
# Intercept SHORT greetings with a friendly intro instead of a bare refusal.
_GREETING = re.compile(r"^\s*(hi|hello|hey|yo|hiya|howdy|good (morning|afternoon|evening)|"
                       r"thanks?|thank you|bye|goodbye)\b", re.I)
_INTRO = (
    "👋 I answer questions **from the loaded technical documents** (MIL-STD-810G/H, "
    "NASA Systems Engineering Handbook) — with citations, and I'll say so when an answer "
    "isn't in them. I don't chat or use outside knowledge.\n\n"
    "Try: *\"In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of "
    "a rain test?\"*"
)


# --- cited-passage highlighting --------------------------------------------------------

def _phrases(text, n_words=7, max_phrases=10):
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


def _highlight_rects(page, text, cap=14):
    """Rectangles on `page` where the source text appears (capped, best-effort)."""
    rects = []
    for phrase in _phrases(text):
        try:
            rects.extend(page.search_for(phrase))
        except Exception:
            pass
        if len(rects) >= cap:
            break
    return rects[:cap]


def _box_figure(page, block_idx):
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


def _render_cited_page(source_file, page, snippet_text=None, figure_block=None):
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
    pdf = _PDFS_DIR / source_file
    if not pdf.exists():
        return None, False
    try:
        with fitz.open(str(pdf)) as doc:
            pg = doc[page - 1]                  # page_number is 1-based
            if figure_block is not None:
                marked = _box_figure(pg, figure_block)
            else:
                rects = _highlight_rects(pg, snippet_text) if snippet_text else []
                for r in rects:
                    annot = pg.add_highlight_annot(r)
                    annot.update()
                marked = bool(rects)
            pix = pg.get_pixmap(dpi=_RENDER_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception:
        return None, False
    out = (img, marked)
    _PAGE_CACHE[key] = out
    return out


def _snippet(text, limit=300):
    """A compact, single-line-collapsed preview of a source chunk."""
    t = re.sub(r"\s+", " ", text or "").strip()
    return (t[:limit].rstrip() + "…") if len(t) > limit else t


# --- chat turn -------------------------------------------------------------------------

def add_user(message, history):
    """Echo the user's message immediately and clear the box (snappy UX)."""
    message = (message or "").strip()
    if not message:
        return history, gr.update()
    return history + [{"role": "user", "content": message}], ""


def _msg_text(content):
    """Extract plain text from a chatbot message's content. Gradio 6 may hand back
    `content` as a list of parts (e.g. [{"type":"text","text":...}]) rather than a str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else (p.get("text") or p.get("content") or "")
                 for p in content if isinstance(p, (str, dict))]
        return " ".join(p for p in parts if p).strip()
    return str(content or "")


def bot(history, doc_scope=_ALL_DOCS):
    """Answer the latest user turn, STREAMING the reply. A generator yielding
    (history, sources_md, gallery): the answer appears token-by-token (perceived latency), then
    the cited-sources panel fills in once the answer completes and citations resolve.

    doc_scope: the dropdown selection -- "All documents" (whole store) or a single source_file
    to constrain retrieval to (mode 2)."""
    if not history or history[-1].get("role") != "user":
        yield history, gr.update(), gr.update()
        return
    message = _msg_text(history[-1].get("content"))

    if _GREETING.match(message) and len(message.split()) <= 4:
        yield history + [{"role": "assistant", "content": _INTRO}], _SOURCES_EMPTY, []
        return

    sf = None if (not doc_scope or doc_scope == _ALL_DOCS) else doc_scope   # mode 2: scope to one doc

    res = None
    for partial, result in answer_stream(message, source_file=sf, history=history[:-1]):
        if result is None:                                  # still streaming the answer text
            yield history + [{"role": "assistant", "content": partial}], gr.update(), gr.update()
        else:
            res = result                                    # final item carries the resolved result

    # answer complete -> build the cited-sources panel + cited-page gallery
    cards, gallery, seen = [], [], set()
    for s in res["sources"]:
        if not s["cited"]:
            continue
        snip = _snippet(s.get("text", ""))
        if snip:
            cards.append(f"**`[{s['tag']}]`** · {s['citation']}\n\n> {snip}")
        if s["page"] and s["source_file"]:
            is_fig = s.get("source_type") == "figure_caption"
            fig_block = s.get("block_idx") if is_fig else None
            key = (s["source_file"], s["page"], fig_block)   # per-figure on a shared page
            if key in seen:
                continue
            seen.add(key)
            img, hit = _render_cited_page(s["source_file"], s["page"], s.get("text"),
                                          figure_block=fig_block)
            if img is not None:
                if is_fig:
                    badge = " · 🖼 figure" if hit else " · 🖼 figure (page)"
                else:
                    badge = " · 🟡 highlighted" if hit else ""
                gallery.append((img, f"{s['source_file']} · p{s['page']}{badge}"))

    if res["hallucinated"]:
        cards.append("⚠️ Model cited tags we did not provide: "
                     + ", ".join(f"`[{t}]`" for t in res["hallucinated"]))

    sources_md = "\n\n---\n\n".join(cards) if cards else \
        "_The answer cited no sources — it likely abstained because the documents don't cover this._"

    yield history + [{"role": "assistant", "content": res["answer"]}], sources_md, gallery


def log_vote(data: gr.LikeData):
    """Log a 👍/👎 on an answer to a local JSONL (one clean row per click). A 👎 becomes
    a candidate to reproduce, diagnose, and promote into the eval set."""
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "liked": bool(data.liked),
        "answer": data.value if isinstance(data.value, str) else str(data.value),
    }
    with open(_FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    gr.Info("👍 Thanks — saved." if data.liked else "👎 Noted — logged for review.")


_CSS = """
footer {display: none !important}
#sources_panel {max-height: 78vh; overflow-y: auto; padding-right: 4px;}
#sources_md blockquote {border-left: 3px solid #2a9d8f; background: #f5f8f8;
    padding: 6px 12px; margin: 4px 0; border-radius: 4px; font-size: 0.9em; color: #2b3a42;}
#sources_md hr {margin: 12px 0; border: none; border-top: 1px solid #e6e9ec;}
"""

_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="teal", neutral_hue="slate")

with gr.Blocks(title="Local Multimodal RAG — technical-PDF Q&A") as demo:
    gr.Markdown("## 📚 Local Multimodal RAG — technical-PDF Q&A")
    gr.Markdown("<sub>Grounded answers cited to the exact page — the cited passage is "
                "highlighted, and the system abstains when the documents don't cover it.</sub>")

    with gr.Row():
        with gr.Column(scale=3):
            doc_scope = gr.Dropdown(
                _DOC_CHOICES, value=_ALL_DOCS, label="📂 Search scope", filterable=True,
                info="Limit to one document (more precise, faster), or search everything (default).")
            chatbot = gr.Chatbot(
                height=560, show_label=False,
                # Render $$...$$ as block math (KaTeX). Block delimiters ONLY -- not
                # single $ -- so a literal dollar amount in an answer ("$5 per unit")
                # isn't mis-rendered as math.
                latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
            )
            with gr.Row():
                msg = gr.Textbox(show_label=False, container=False, autofocus=True, scale=9,
                                 placeholder="Ask about MIL-STD-810G/H or the NASA SE Handbook…")
                send = gr.Button("Send", variant="primary", scale=1, min_width=90)
            gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one")
        with gr.Column(scale=2, elem_id="sources_panel"):
            gr.Markdown("#### 📄 Cited sources")
            # Click a thumbnail to enlarge it; the ⤢ Fullscreen control reads it full screen
            # (Esc returns to the preview), and the ✕ closes the preview back to the grid.
            gallery = gr.Gallery(show_label=False, columns=2, height=320, object_fit="contain")
            sources_md = gr.Markdown(_SOURCES_EMPTY, elem_id="sources_md")

    # snappy two-step: echo the user turn + clear the box, THEN compute the answer.
    # show_progress_on=[chatbot] keeps the "thinking" spinner on the chat, not the Sources panel.
    for trigger in (msg.submit, send.click):
        trigger(add_user, [msg, chatbot], [chatbot, msg], queue=False).then(
            bot, [chatbot, doc_scope], [chatbot, sources_md, gallery], show_progress_on=[chatbot])
    chatbot.like(log_vote)


if __name__ == "__main__":
    # local only -- no public share link; the confidential corpus stays on the machine.
    # css hides Gradio's footer and styles the sources panel for a clean, professional UI.
    demo.queue().launch(css=_CSS, theme=_THEME)   # queue() required for streaming generators
