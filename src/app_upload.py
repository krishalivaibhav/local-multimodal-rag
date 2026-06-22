"""
app_upload.py -- "Upload & Ask": a local Gradio app where a user uploads a PDF,
it is ingested on the spot, and they ask questions answered FROM that PDF with
citations + cited-page images.

Reuses the whole engine (ingest -> retrieve -> answer). The only new parts are the
upload widget and the ingest trigger. Uploaded docs go into an ISOLATED store
(data/chroma_uploads) so they never mix with the curated demo corpus (data/chroma).

v1 scope (honest): text-only ingestion (the figure/VLM path is the slow one --
deferred); single-user/demo; a new upload adds to the store and queries scope to the
latest uploaded doc. Fully local -- no public share link.

Run:  cd src && python app_upload.py     # then open http://127.0.0.1:7860
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Isolated store for uploaded docs -- NEVER touches the curated demo store (data/chroma).
# Set before any engine import (config reads CHROMA_DIR at import time).
os.environ["RAG_CHROMA_DIR"] = str(_ROOT / "data" / "chroma_uploads")
os.environ.setdefault("RAG_HYBRID", "1")
os.environ.setdefault("RAG_PARENT_TABLES", "1")

import json
import shutil
import time

import fitz
import gradio as gr
from PIL import Image

import ingest
from answer import answer_question

_UPLOADS_DIR = _ROOT / "data" / "uploads"          # uploaded PDFs, for cited-page rendering
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_DIR = _ROOT / "data" / "feedback"
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_LOG = _FEEDBACK_DIR / "feedback.jsonl"

_PAGE_CACHE = {}


def ingest_pdf(file, progress=gr.Progress()):
    """Ingest an uploaded PDF (text-only) into the isolated store.
    Returns (status_markdown, source_file_name) -- the name is stored in session state
    so queries scope to this document and cited pages render from it."""
    if not file:
        return "Upload a PDF to begin.", None
    name = Path(file).name
    dest = _UPLOADS_DIR / name
    shutil.copyfile(file, dest)                    # keep the PDF for cited-page rendering

    progress(0.1, desc=f"Parsing {name} ...")
    chunks, _ = ingest.build_all_chunks(str(dest), name, use_vlm=False)
    progress(0.6, desc=f"Embedding {len(chunks)} chunks ...")
    ingest.embed_and_store(chunks)
    # the BM25 index is cached on first query; invalidate it so the new doc is included
    import hybrid
    hybrid._cache.clear()
    progress(1.0, desc="Done")
    return f"✅ Ready — ask about **{name}** ({len(chunks)} chunks indexed).", name


def _render_cited_page(source_file, page):
    """Render a cited page of an uploaded PDF to a PIL image (cached)."""
    if not source_file or not page:
        return None
    key = (source_file, page)
    if key in _PAGE_CACHE:
        return _PAGE_CACHE[key]
    pdf = _UPLOADS_DIR / source_file
    if not pdf.exists():
        return None
    try:
        with fitz.open(str(pdf)) as doc:
            pix = doc[page - 1].get_pixmap(dpi=110)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception:
        return None
    _PAGE_CACHE[key] = img
    return img


def respond(message, history, current_doc):
    """One chat turn, scoped to the uploaded document -> (answer markdown, cited pages)."""
    if not current_doc:
        return "📄 Upload a PDF first, then ask about it.", []

    res = answer_question(message, source_file=current_doc)
    cited = set(res["cited"])
    lines = [res["answer"], "", "**Sources** (✅ = cited in the answer):"]
    for tag, loc in res["citation_map"].items():
        mark = "✅" if tag in cited else "▫️"
        lines.append(f"- {mark} `[{tag}]` — {loc}")

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
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "liked": bool(data.liked),
           "answer": data.value if isinstance(data.value, str) else str(data.value)}
    with open(_FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    gr.Info("👍 Thanks — saved." if data.liked else "👎 Noted — logged for review.")


with gr.Blocks(title="Upload & Ask — local RAG") as demo:
    gr.Markdown("## Upload & Ask — your PDF, answered with citations")
    current_doc = gr.State(None)
    with gr.Row():
        upload = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
        status = gr.Markdown("Upload a PDF to begin.")
    gallery = gr.Gallery(label="📄 Cited source pages", columns=2, height=460, render=False)
    chatbot = gr.Chatbot(height=420)
    gr.ChatInterface(
        respond,
        chatbot=chatbot,
        additional_inputs=[current_doc],
        additional_outputs=[gallery],
    )
    gallery.render()
    upload.upload(ingest_pdf, inputs=upload, outputs=[status, current_doc])
    chatbot.like(log_vote)


if __name__ == "__main__":
    # local only; css hides Gradio's footer for a minimal UI.
    demo.launch(css="footer {display: none !important}")
