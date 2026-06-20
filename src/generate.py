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


if __name__ == "__main__":
    # Smoke test: can the boundary reach a local model and get text back?
    reply = generate(
        [{"role": "user", "content": "Reply with exactly: pipeline online."}],
    )
    print(f"MODEL : {MODEL}")
    print(f"reply : {reply!r}")
