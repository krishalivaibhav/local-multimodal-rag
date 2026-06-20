# Results

Measured results for the local multimodal RAG pipeline. Every number here is from the
evaluation harness (`src/eval_retrieval.py`, `src/eval_answer.py`) on hand-verified test
sets. Fully local: no hosted APIs, open-weights only. The narrative behind each number is
in `findings.md` (35 findings); design rationale is in `defense_qa.md`.

## TL;DR

- **Single document (MIL-STD-810H), independent test set:** retrieval Hit@5 **1.00** /
  MRR 0.764; answer **value 0.80 / citation 0.65** (F1 **0.84 / 0.68**), with reliable
  abstention. This is the honest headline number.
- **Three-document corpus** (810H + 810G + NASA SE Handbook): cross-document routing fixed
  to **precision 1.00**; generalization to a new-domain doc holds (NASA ~5/6).
- **The honest weak spot is tables** — reading is solved (VLM), representation-for-QA is the
  open frontier (an unsolved problem in the field, not just here).

## How quality was measured (the methodology that makes the numbers trustworthy)

- **Independent test set.** The first test questions were drafted *from chunks in the store*,
  so every answer was retrievable by construction -- value looked like **0.92**. Re-authoring
  questions from the *document* (a "HARD" set, some user-authored blind) dropped it to the
  honest **0.80**. The 0.12 gap *was* selection bias. (findings #16/#18)
- **Value and citation scored separately.** Two cases are mirror images -- one wrong-value/
  right-citation, one right-value/wrong-citation -- proving a single metric would lie. Both
  are reported always. (findings #14/#15)
- **Precision / recall / F1**, not just accuracy: precision = correct among *answered*,
  recall = correct among *answerable*. High precision + lower recall = the system fails by
  *abstaining*, not fabricating -- the safe direction for confidential data.
- **Negative results are kept.** Several promising ideas were measured and *rejected*
  (below) -- the discipline, not just the wins.

## Answer quality (single document, MIL-STD-810H)

| Test set | value acc | value F1 | citation acc | citation F1 | abstention |
|---|---|---|---|---|---|
| Optimistic (drawn from chunks) | 0.92 | -- | 0.85 | -- | -- |
| **HARD (independent, n=20+3)** | **0.80** | **0.84** | **0.65** | **0.68** | 3/3 refused unanswerable, 1/20 over-abstain |

By content type (HARD set): **prose 13/13**, tables **3/7**. Prose Q&A is effectively solved;
tables are the weak spot.

## Multi-document corpus (810H + 810G + NASA SE Handbook)

First real corpus test: 3 born-digital PDFs, one store. 810G is a *near-duplicate* of 810H
(prior version); NASA is a different domain/style. This exercised the multi-doc path for the
first time and surfaced -- then fixed -- real bugs.

| Stage (MULTIDOC, n=11) | retrieval Hit@5 | answer value | note |
|---|---|---|---|
| Plain dense | 0.55 | 0.45 | near-duplicate 810H out-competes 810G; doc name in query ignored |
| + document routing | 0.64 | **0.55** | alias router -> scope to the named doc; citation precision -> 1.00 |
| + hybrid (dense+BM25) | 0.91\* | 0.45 | \*inflated by repeated page-header chunks |
| + header de-noise | **0.73** (honest) | **0.55** | dropped ~20k tiny header/footer chunks; hardest case (immersion) fixed |

- **Generalization holds:** NASA (genuinely different doc) answers ~5/6 with the *same*
  heuristics -- not overfit to MIL-STD.
- **Cross-document citations** now name the source document (the single-doc setup hid this
  ambiguity -- "page 501" is ambiguous across 3 docs). (finding #29)
- **Abstention held under multi-doc:** the system refused rather than answer from the wrong
  document; it never fabricated a confident wrong-doc answer.

## Tables -- the weak spot, decomposed honestly

| Sub-problem | Status |
|---|---|
| Reading (transcribe a table) | **Solved** -- local VLM (gemma3:4b/gemma4/qwen2.5-vl) reads ruled, unruled and complex tables |
| Detection (find borderless tables) | **Solved** -- VLM-as-parser catches tables pdfplumber misses |
| Representation-for-QA (chunk so the model finds the right cell) | **Open** -- per-row picks the wrong row for ambiguous queries; whole-table doesn't retrieve. Hard in the field too. |

Best measured: **per-row VLM chunks took tables 2/7 -> 3/7** (single doc). The bottleneck is
*representation*, not reading. (findings #21/#22/#24/#33/#35)

## Figures (Phase 2, validated)

The figures pipeline (`figures.py`) clusters fragmented image pieces into one region, bundles
the caption, and has the VLM describe it. Validated end-to-end on p602 (28 image fragments ->
1 correct region -> usable caption with the exact figure title). Built and proven; not yet
measured at corpus scale. (finding #28)

## Ideas that were measured and REJECTED (the discipline)

| Idea | Result | Why kept out |
|---|---|---|
| Suppress the prose copy of tables | value 0.80 -> 0.75 | dual streams are *complementary*, not redundant (finding #20) |
| Query-time table reranker | 0.80 -> 0.75 | boosting tables pulls wrong-page tables (finding #26) |
| Verbose-header VLM prompt | garbled transcription | small VLM degrades on complex instructions (finding #25) |
| "Drop TOC pages" | hypothesis disproved | the polluting pages weren't a TOC; real cause was embedder ranking (finding #32) |
| Stronger VLM (qwen) on the M2 | too slow / verbose | M2 hardware limit, not quality -- belongs on the GPU |

## Limitations (open, honestly tracked in `deferred.md`)

- Table **representation-for-QA** (the frontier, #26).
- **Math/equation cells** garble in text extraction (#3).
- **Scanned/OCR** path designed, not built (#6).
- A few test questions are **ambiguous** (e.g. NASA p48) -- to tighten (#26).
- VLM table pipeline measured on a strong reader; **full-corpus VLM+figures run pending the
  lab GPU** (qwen2.5-vl; gemma4 is M2-only on VRAM).

## Reproduce

```bash
cd src
python eval_retrieval.py hard          # retrieval: Hit@k / MRR
python eval_answer.py hard             # answer: value / citation / F1 / abstention
python eval_answer.py multidoc         # multi-doc set (add RAG_HYBRID=1 for hybrid search)
```

## Stack

Fully local: PyMuPDF + pdfplumber (parsing), `bge-small-en-v1.5` (embeddings), ChromaDB
(vector store), Ollama-served `gemma3:4b` (VLM) and `qwen2.5:7b-instruct` (answer LLM), with
dense+BM25 hybrid retrieval and a deterministic document router. Raw Python -- no LangChain
orchestration. Every model is swappable via one env var (`RAG_VLM_MODEL` / `RAG_LLM_MODEL`),
so the system scales up with better hardware without code changes.
