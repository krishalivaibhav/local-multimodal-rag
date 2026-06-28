"""
api.py -- a FastAPI backend that exposes the local RAG pipeline (src/) over HTTP, so a custom
front-end (the "Atlas" design) can drive it. A THIN transport layer: nothing in the pipeline
changes -- it wraps answer_stream(), the PDF list, and the cited-page renderer.

  GET  /api/docs    the indexed PDFs (for the document selector / Mode-2 scoping)
  POST /api/ask     answer a question, STREAMED token-by-token over SSE, then the citations
  GET  /api/page    the cited page rasterised with the passage highlighted (PNG)

Runs fully on-device (127.0.0.1); no data leaves the machine. Same GPU/feature defaults as app.py
(rerank + cross-revision dedup on; embedder/reranker on MPS when available, LLM on the GPU).
"""
import os

# Query the FULL multimodal 23-doc store (chroma_mm), not the old 3-doc data/chroma default.
# The document selector lists every PDF in data/pdfs, so the store it queries must hold them all --
# otherwise a scoped question (e.g. a PHAK chapter) retrieves nothing and the answer is ungrounded.
# setdefault so an explicit RAG_CHROMA_DIR in the shell still wins. MUST precede the config import.
os.environ.setdefault(
    "RAG_CHROMA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_mm"),
)
os.environ.setdefault("RAG_RERANK", "1")
os.environ.setdefault("RAG_DEDUP_REVISIONS", "1")
try:                                                    # embedder + reranker on the Apple GPU
    import torch
    if torch.backends.mps.is_available():
        os.environ.setdefault("RAG_DEVICE", "mps")
except Exception:
    pass

import io
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import fitz
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from answer import answer_stream
from compute import try_compute
from render import PDFS_DIR, render_cited_page, snippet as make_snippet

ALL_DOCS = "all"
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Atlas RAG API")
# CORS open for local dev (front-end may be served from a different port while iterating).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- document list (for the selector) --------------------------------------------------

def _doc_list():
    """The indexed PDFs with page counts -- computed once at startup for the document selector."""
    docs = [{"id": ALL_DOCS, "name": "All documents", "kind": "all"}]
    for p in sorted(PDFS_DIR.glob("*.pdf")):
        try:
            with fitz.open(str(p)) as d:
                pages = d.page_count
        except Exception:
            pages = None
        docs.append({"id": p.name, "name": p.name, "kind": "pdf",
                     "meta": f"{pages} pages" if pages else "PDF"})
    docs[0]["meta"] = f"{len(docs) - 1} sources"
    return docs


_DOCS = _doc_list()


@app.get("/api/docs")
def api_docs():
    return {"documents": _DOCS}


# --- ask (streaming) -------------------------------------------------------------------

def _sse(event, data):
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_done(result):
    """Turn answer_stream's final result dict into the front-end's citation payload: cited sources
    numbered [1..n] by first appearance in the answer, each with a /api/page image URL."""
    if not result:
        return {"answer": "", "citations": [], "tag_to_n": {}, "abstained": True}
    answer = result["answer"]
    cited = result["cited"]
    by_tag = {s["tag"]: s for s in result["sources"]}

    order = []                                   # number citations by first appearance in the text
    for m in re.finditer(r"\[(S\d+)\]", answer):
        t = m.group(1)
        if t in cited and t not in order:
            order.append(t)
    for t in cited:                              # any cited tag with no inline marker -> append
        if t not in order:
            order.append(t)

    citations, tag_to_n = [], {}
    for i, tag in enumerate(order, start=1):
        s = by_tag.get(tag, {})
        tag_to_n[tag] = i
        page, sf = s.get("page"), s.get("source_file")
        is_fig = s.get("source_type") == "figure_caption"
        block = s.get("block_idx") if is_fig else None
        snip = make_snippet(s.get("text", ""), limit=260)
        img = None
        if sf and page:
            qp = {"file": sf, "page": page}
            qp["block" if block is not None else "q"] = block if block is not None else snip
            img = "/api/page?" + urlencode(qp)
        kind = "figure" if is_fig else ("table" if str(s.get("source_type", "")).startswith("table") else "pdf")
        match = s.get("match")
        citations.append({"n": i, "tag": tag, "doc": sf, "page": page,
                          "loc": f"p. {page}" if page else "", "kind": kind,
                          "source_type": s.get("source_type"), "snippet": snip, "image_url": img,
                          "match": round(match, 4) if match is not None else None,
                          "score": round(s["score"], 4) if s.get("score") is not None else None})
    return {"answer": answer, "citations": citations, "tag_to_n": tag_to_n,
            "abstained": len(cited) == 0}


@app.post("/api/ask")
def api_ask(payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    doc = payload.get("doc") or ALL_DOCS
    source_file = None if doc == ALL_DOCS else doc

    def stream():
        if not question:
            yield _sse("done", {"answer": "", "citations": []})
            return
        # math-compute path: deterministic sympy, no retrieval (the "plug in values" feature)
        comp = try_compute(question)
        if comp:
            yield _sse("stage", {"stage": "computing"})
            yield _sse("token", {"text": comp})
            yield _sse("done", {"answer": comp, "citations": [], "tag_to_n": {}, "compute": True})
            return
        # RAG path: stream the answer token-by-token, then the resolved citations
        yield _sse("stage", {"stage": "retrieving"})
        final, prev, first = None, "", True
        for partial, result in answer_stream(question, source_file=source_file):
            if result is None:
                if first:
                    yield _sse("stage", {"stage": "generating"})
                    first = False
                delta, prev = partial[len(prev):], partial
                if delta:
                    yield _sse("token", {"text": delta})
            else:
                final = result
        yield _sse("done", _build_done(final))

    return StreamingResponse(stream(), media_type="text/event-stream")


# --- cited page image ------------------------------------------------------------------

@app.get("/api/page")
def api_page(file: str, page: int, block: Optional[int] = None, q: Optional[str] = None):
    img, marked = render_cited_page(file, page, snippet_text=q, figure_block=block)
    if img is None:
        return Response(status_code=404)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Marked": "1" if marked else "0", "Cache-Control": "max-age=3600"})


@app.get("/api/health")
def api_health():
    return {"ok": True, "documents": len(_DOCS) - 1}


# Serve the custom front-end (built in step 2) from the same origin when it exists. Mounted LAST
# so the /api/* routes above always take precedence.
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    try:                                            # warm models so the first request is fast
        print("Warming up models (embedder + reranker + LLM)...", flush=True)
        for _ in answer_stream("warmup"):
            pass
        print("Models warm.", flush=True)
    except Exception as e:
        print(f"(warmup skipped: {e})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
