"""
app.py -- a local Gradio chat UI over the RAG engine.

A thin front-end, nothing more: every question goes through answer_question()
(retrieval -> grounded prompt -> local LLM -> citation resolution), and the reply
shows the grounded answer PLUS the resolved citations, with the sources the answer
actually cited marked. We show ALL retrieved sources (cited and not), because
auditability is the product (decision #4): a user should see both what retrieval
found and what the answer claimed to stand on.

Fully local -- local vector store + local Ollama; no data leaves the machine, and
no public share link. The engine is unchanged; this is pure presentation.

Run:  cd src && python app.py     # then open the printed http://127.0.0.1:7860
"""
import re

import gradio as gr

from answer import answer_question

# Greeting / chit-chat: this is a grounded document-QA system, not a general chatbot,
# so a bare "hello" correctly abstains. But a bare refusal reads as broken, so we
# intercept SHORT greetings and explain what the assistant does instead. The word-count
# guard avoids catching a real question that merely starts with "hi".
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
    """One chat turn: answer the question, render the answer + cited sources.

    Abstention flows through naturally -- the engine returns its fixed refusal
    sentence as the answer, and the retrieved sources still show (transparent:
    'here is what was found; the model declined to answer from it')."""
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


demo = gr.ChatInterface(
    respond,
    title="Local Multimodal RAG — technical-PDF Q&A",
    description=("Ask about the ingested documents. Answers are grounded in the PDFs with "
                 "citations, and the system abstains rather than guess. Fully local."),
    # Demo the system at its honest best -- prose + tables (its strengths), not the
    # equation-heavy ASD(Peak) definition (which surfaces the math-extraction limit).
    examples=[
        "In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of a rain test?",
        "In NASA's project-type classification (Table 3.11-1), what kind of mission is Type A?",
        "What kind of test stands are listed among facilities to consider in Table 6.1-2?",
    ],
)


if __name__ == "__main__":
    # local only -- no public share link, the confidential corpus stays on the machine.
    demo.launch()
