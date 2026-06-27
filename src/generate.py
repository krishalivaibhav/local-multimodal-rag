"""
generate.py -- the LLM boundary for answer generation.

One job: turn a list of chat messages into the model's reply text. Everything
model-specific is hidden behind this function, so the rest of the pipeline
(retrieval, prompt building, citation mapping) stays model-agnostic.

Why this shape:
  - Prompts are built as role/content MESSAGES, never raw special tokens. Ollama
    applies the selected model's own chat template, so switching models is a
    one-line change to MODEL -- the correct ChatML / Llama / Mistral formatting
    comes from the model itself. (See deferred: model swap is cheap by design.)
  - Ollama runs the same code on the dev Mac (Metal) and the target A1000 (CUDA).
"""

import ollama

from config import LLM_MODEL

# The single place the model choice lives (config.LLM_MODEL / RAG_LLM_MODEL).
# Swap target by setting that env var, e.g. "llama3.1:8b-instruct-q4_K_M".
MODEL = LLM_MODEL


def generate(messages, model=MODEL, temperature=0.0):
    """Send chat messages to the local LLM and return the reply text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    temperature=0.0: answer generation must be FAITHFUL to the retrieved
    context, not creative. Greedy decoding gives the same grounded answer every
    run (reproducible eval) and curbs hallucination.
    """
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def generate_stream(messages, model=MODEL, temperature=0.0):
    """Same grounded decoding as generate(), but YIELD the reply chunk-by-chunk as the model
    produces it (ollama stream=True). The caller concatenates the pieces; lets a UI show the
    answer appearing live, which masks the M2-CPU latency without changing the final text."""
    for chunk in ollama.chat(model=model, messages=messages,
                             options={"temperature": temperature}, stream=True):
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


# HyDE (Hypothetical Document Embeddings). Under-specified queries that DESCRIBE a table
# ("what test stands are in Table 6.1-2?") match the table's CAPTION, not its cells. HyDE
# asks the LLM to write the answer it EXPECTS, then we retrieve on that text -- so the
# query vector lands near the answer content. Confident, no-hedge instruction so the
# passage reads like a document excerpt (a refusal/"I'm unsure" would be useless to embed).
HYDE_PROMPT = (
    "Write a short, factual passage (2-3 sentences) that could plausibly answer the "
    "question below, as if excerpted from a technical reference document. Write it as a "
    "confident statement of fact; do not hedge, do not say you are unsure, do not ask for "
    "clarification. Output only the passage.\n\nQuestion: "
)


def hyde_query(question, model=MODEL, temperature=0.0):
    """Return a short hypothetical answer passage for `question` (HyDE).

    The caller embeds this alongside the question so retrieval matches answer content,
    not just the question's wording. temperature=0.0 -> one deterministic passage, so the
    eval is reproducible. Returns "" on any failure, so the caller falls back to the raw
    question -- HyDE can degrade to baseline but never break retrieval.
    """
    try:
        msgs = [{"role": "user", "content": HYDE_PROMPT + question}]
        return generate(msgs, model=model, temperature=temperature).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    # Smoke test: can the boundary reach a local model and get text back?
    reply = generate(
        [{"role": "user", "content": "Reply with exactly: pipeline online."}],
    )
    print(f"MODEL : {MODEL}")
    print(f"reply : {reply!r}")
