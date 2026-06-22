"""
app.py -- a local Gradio chat UI over the RAG engine.

A thin front-end: every question goes through answer_question() (retrieval ->
grounded prompt -> local LLM -> citation resolution). The reply shows the grounded
answer + all retrieved sources (cited ones marked), AND renders the cited PDF
pages in a gallery so a user can SEE the grounding -- auditability is the product.
Each answer carries native 👍/👎 thumbs whose votes are logged locally
(data/feedback/feedback.jsonl) -- the on-box feedback loop for an air-gapped deploy.

Runs at FULL retrieval strength (hybrid + parent-tables). Fully local -- local
store + local Ollama + local PDFs, no public share link.

Run:  cd src && python app.py     # then open http://127.0.0.1:7860
"""
import os

# Full retrieval strength by default (the config behind the 0.73 result). Must be set
# BEFORE answer/query/config import (config reads these at import time). setdefault so an
# explicit env var still wins.
os.environ.setdefault("RAG_HYBRID", "1")
os.environ.setdefault("RAG_PARENT_TABLES", "1")

import json
import re
import time
from pathlib import Path

import fitz
import gradio as gr
from PIL import Image

from answer import answer_question

_ROOT = Path(__file__).resolve().parent.parent
_PDFS_DIR = _ROOT / "data" / "pdfs"

# Feedback log -- local, under data/ (gitignored like the store/PDFs).
_FEEDBACK_DIR = _ROOT / "data" / "feedback"
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_LOG = _FEEDBACK_DIR / "feedback.jsonl"

# Rendered cited pages, cached in memory by (source_file, page) -- cheap to keep a few.
_PAGE_CACHE = {}

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


def _render_cited_page(source_file, page):
    """Render a cited PDF page to a PIL image (cached). Returns the image, or None."""
    if not source_file or not page:
        return None
    key = (source_file, page)
    if key in _PAGE_CACHE:
        return _PAGE_CACHE[key]
    pdf = _PDFS_DIR / source_file
    if not pdf.exists():
        return None
    try:
        with fitz.open(str(pdf)) as doc:
            pix = doc[page - 1].get_pixmap(dpi=110)        # page_number is 1-based
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception:
        return None
    _PAGE_CACHE[key] = img
    return img


def respond(message, history):
    """One chat turn -> (answer markdown, gallery of cited source pages)."""
    if _GREETING.match(message) and len(message.split()) <= 4:
        return _INTRO, []

    res = answer_question(message)
    cited = set(res["cited"])
    lines = [res["answer"], "", "**Sources** (✅ = cited in the answer):"]
    for tag, loc in res["citation_map"].items():
        mark = "✅" if tag in cited else "▫️"
        lines.append(f"- {mark} `[{tag}]` — {loc}")
    if res["hallucinated"]:
        lines.append("")
        lines.append("⚠️ The model cited tags we did not provide: "
                     + ", ".join(f"[{t}]" for t in res["hallucinated"]))

    # Render the CITED pages (deduped) so the user can see exactly what grounded the answer.
    gallery, seen = [], set()
    for s in res["sources"]:
        if not s["cited"] or not s["page"]:
            continue
        key = (s["source_file"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        img = _render_cited_page(s["source_file"], s["page"])
        if img is not None:
            gallery.append((img, f"{s['source_file']} · page {s['page']}"))

    return "\n".join(lines), gallery


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


with gr.Blocks(title="Local Multimodal RAG — technical-PDF Q&A") as demo:
    gr.Markdown("## Local Multimodal RAG — technical-PDF Q&A")
    # defined render=False so it can be placed BELOW the chat (after ChatInterface renders)
    gallery = gr.Gallery(label="📄 Cited source pages", columns=2, height=460, render=False)
    chatbot = gr.Chatbot(height=460)
    gr.ChatInterface(
        respond,
        chatbot=chatbot,
        additional_outputs=[gallery],
        examples=[
            "In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of a rain test?",
            "In NASA's project-type classification (Table 3.11-1), what kind of mission is Type A?",
            "What kind of test stands are listed among facilities to consider in Table 6.1-2?",
        ],
    )
    gallery.render()
    chatbot.like(log_vote)


if __name__ == "__main__":
    # local only -- no public share link; the confidential corpus stays on the machine.
    # css hides Gradio's footer (Use via API / Built with Gradio / Settings) for a minimal UI.
    demo.launch(css="footer {display: none !important}")
