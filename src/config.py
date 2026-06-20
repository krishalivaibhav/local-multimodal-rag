"""
config.py -- one place for everything that changes between machines.

Every setting reads an environment variable with a sensible default, so you can
clone the repo on a new box (Windows + CUDA, the lab GPU, this Mac) and just set a
few env vars -- no code edits. Defaults reproduce the dev-Mac setup.

Env vars:
  RAG_PDF_PATH     path to the source PDF        (default: data/pdfs/mil-std-810h.pdf)
  RAG_CHROMA_DIR   vector store directory        (default: data/chroma)
  RAG_COLLECTION   Chroma collection name        (default: mil_std_docs)
  RAG_DEVICE       embedding device: cpu|cuda|mps(default: cpu)  -- ONLY affects the
                   sentence-transformers embedder; Ollama (VLM/LLM) picks its own GPU.
  RAG_EMBED_MODEL  embedding model               (default: BAAI/bge-small-en-v1.5)
  RAG_VLM_MODEL    vision model (Ollama tag)     (default: gemma3:4b)
  RAG_LLM_MODEL    answer model (Ollama tag)     (default: qwen2.5:7b-instruct-q4_K_M)
  RAG_RENDER_DPI   table-image render DPI        (default: 200)  -- lower = fewer vision
                   tokens = faster / less VRAM, but blurrier (risk of misreads).

Per-machine examples:
  Windows + 6GB RTX 3060:  RAG_DEVICE=cuda RAG_VLM_MODEL=gemma3:4b RAG_RENDER_DPI=150
  Windows + 8GB RTX 5050:  RAG_DEVICE=cuda RAG_VLM_MODEL=qwen2.5vl:7b RAG_RENDER_DPI=150
"""

import os
from pathlib import Path

# repo root = parent of this src/ directory; default data paths hang off it, so the
# project is relocatable (no absolute paths baked in).
_ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.environ.get(name, default)


PDF_PATH = _env("RAG_PDF_PATH", str(_ROOT / "data" / "pdfs" / "mil-std-810h.pdf"))
PDFS_DIR = _env("RAG_PDFS_DIR", str(_ROOT / "data" / "pdfs"))   # ingest ALL *.pdf here
CHROMA_DIR = _env("RAG_CHROMA_DIR", str(_ROOT / "data" / "chroma"))
COLLECTION_NAME = _env("RAG_COLLECTION", "mil_std_docs")

DEVICE = _env("RAG_DEVICE", "cpu")             # embedder device only (see note above)
EMBED_MODEL = _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
VLM_MODEL = _env("RAG_VLM_MODEL", "gemma3:4b")
LLM_MODEL = _env("RAG_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")

RENDER_DPI = int(_env("RAG_RENDER_DPI", "200"))
