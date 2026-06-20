"""
vlm.py -- the vision-model boundary for table (and later figure) transcription.

Mirror of generate.py: one job, one swappable model constant. Images are turned
into TEXT here, so the rest of the pipeline (embedding, retrieval, answer LLM)
stays in the unified text-space and never sees a pixel. Any (image, prompt)->text
model is a drop-in; swapping is a one-line change to VLM_MODEL (e.g. on a hardware
upgrade, or to qwen2.5-vl). Same Ollama call shape as the answer LLM, so dev (Metal)
and the target A1000 (CUDA) run identical code.
"""

import ollama

from config import VLM_MODEL  # vision-model choice (RAG_VLM_MODEL); swap per machine

# Robustness: a runaway generation on one bad table image once froze the whole
# 217-table build. Two guards: cap the output length (a transcription is short),
# and a per-call HTTP timeout as a backstop. One bad table must not wedge ingest.
MAX_TOKENS = 1536
TIMEOUT_S = 90
_client = ollama.Client(timeout=TIMEOUT_S)

# Faithful transcription, not interpretation. The multi-column-header instruction
# is the whole point: it is exactly the structure that breaks the text pipeline
# (finding #20: p1075 Table XXIV's "Temperature -> C/F" header).
TABLE_PROMPT = (
    "Transcribe this table to GitHub-flavored markdown. Preserve every row, "
    "column and header exactly as shown, including units. If a header spans "
    "multiple columns, repeat it so each column has its own complete label. "
    "Do not summarise, infer, or add anything. Output only the markdown table."
)


# Figures want DESCRIPTION (for retrieval), not verbatim transcription. The
# surrounding caption/heading is appended as context so the model isn't guessing
# from pixels alone (README: context bundling).
FIGURE_PROMPT = (
    "Describe this figure for search and question-answering. State what kind of "
    "figure it is (chart, diagram, schematic, plot, equation, ...), its title or "
    "caption if visible, the axes / labels / units, and the key relationship or "
    "information it conveys. If it is an equation, transcribe it. Be concise and "
    "factual; do not speculate beyond what is shown."
)


def _ask(image_path, prompt, model, temperature):
    response = _client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        options={"temperature": temperature, "num_predict": MAX_TOKENS},
    )
    return response["message"]["content"]


def transcribe_table(image_path, model=VLM_MODEL, temperature=0.0):
    """Render -> text: transcribe a table image to markdown via the local VLM.

    temperature=0.0: transcription must be faithful and reproducible, not creative.
    """
    return _ask(image_path, TABLE_PROMPT, model, temperature)


# VLM-as-parser: detect AND read tables from a whole page, so we catch BORDERLESS
# tables that the geometric detector (pdfplumber) misses entirely (finding #34/NASA
# p48 fell to garbled prose). The VLM sees the table whether or not it has rule lines.
PAGE_TABLE_PROMPT = (
    "This is a page from a technical document. Transcribe ONLY the data table(s) on "
    "it to GitHub-flavored markdown -- one markdown table per table, separated by a "
    "blank line. Preserve every row, column and header exactly, including units; if a "
    "header is grouped, give each sub-column a complete label. Ignore body prose, "
    "headers/footers and figures. If the page contains no data table, output exactly: NONE"
)


def transcribe_page_tables(image_path, model=VLM_MODEL, temperature=0.0):
    """VLM-as-parser: extract any tables on a full page to markdown (or 'NONE')."""
    return _ask(image_path, PAGE_TABLE_PROMPT, model, temperature)


def transcribe_figure(image_path, context="", model=VLM_MODEL, temperature=0.0):
    """Render -> text: describe a figure region for retrieval. `context` (the nearby
    caption / section heading) is appended so the model is grounded, not guessing."""
    prompt = FIGURE_PROMPT
    if context.strip():
        prompt += f"\n\nSurrounding text (for context, may be incomplete): {context.strip()}"
    return _ask(image_path, prompt, model, temperature)


if __name__ == "__main__":
    import sys
    # Smoke test: transcribe a pre-rendered table image (e.g. /tmp/tbl_p1075_t0.png).
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tbl_p1075_t0.png"
    print(f"VLM_MODEL: {VLM_MODEL}\nimage: {path}\n")
    print(transcribe_table(path))
