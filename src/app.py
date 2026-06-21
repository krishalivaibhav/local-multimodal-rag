"""
app.py -- a local Gradio chat UI over the RAG engine.

A thin front-end: every question goes through answer_question() (retrieval ->
grounded prompt -> local LLM -> citation resolution), and the reply shows the
grounded answer PLUS all retrieved sources with the cited ones marked (decision
#4: auditability is the product). Each answer carries native 👍/👎 thumbs whose
votes are logged locally (data/feedback/feedback.jsonl) -- the on-box feedback
loop, which in an air-gapped deployment is the only way to learn what's failing.

Runs at FULL retrieval strength (hybrid + parent-tables). Fully local -- local
store + local Ollama, no public share link.

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

import gradio as gr

from answer import answer_question

# Feedback log -- local, under data/ (gitignored like the store/PDFs).
_FEEDBACK_DIR = Path(__file__).resolve().parent.parent / "data" / "feedback"
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_LOG = _FEEDBACK_DIR / "feedback.jsonl"

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


def respond(message, history):
    """One chat turn: answer the question, render the answer + cited sources."""
    if _GREETING.match(message) and len(message.split()) <= 4:
        return _INTRO
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
    return "\n".join(lines)


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
    gr.Markdown(
        "## Local Multimodal RAG — technical-PDF Q&A\n"
        "Ask about the ingested documents. Answers are **grounded in the PDFs with "
        "citations**, and the system **abstains rather than guess**. Rate any answer with "
        "the 👍 / 👎 beneath it — feedback is logged locally for review. Fully local."
    )
    chatbot = gr.Chatbot(height=460)
    gr.ChatInterface(
        respond,
        chatbot=chatbot,
        examples=[
            "In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of a rain test?",
            "In NASA's project-type classification (Table 3.11-1), what kind of mission is Type A?",
            "What kind of test stands are listed among facilities to consider in Table 6.1-2?",
        ],
    )
    chatbot.like(log_vote)


if __name__ == "__main__":
    # local only -- no public share link; the confidential corpus stays on the machine.
    demo.launch()
