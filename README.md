# Local Multimodal RAG Pipeline for Technical PDFs

*A retrieval-augmented question-answering system for confidential technical documents — one that handles prose, tables, and figures as first-class citizens, runs entirely on-premise, and is designed to be debugged, evaluated, and extended one component at a time.*

---

## The problem

Technical documents — manuals, reference books, engineering specifications — are not just paragraphs of text. The information that matters most is often locked inside **tables** (numerical values, parameter references) and **diagrams** (schematics, charts, annotated figures). A modern engineer reading such a document spends as much time looking at tables and figures as at prose.

Existing tools handle this poorly:

- **Text-only RAG** silently strips tables into mangled rows and ignores figures completely, missing roughly half the information in a diagram- and table-heavy corpus.
- **Generic PDF-to-Markdown converters** flatten table structure and drop spatial metadata (page coordinates, bounding boxes), which is exactly the information needed for citations.
- **Vision-native retrieval approaches** (e.g. ColPali) embed entire page images using a vision-language model — strong on visual cues, but heavy on hardware and difficult to debug per component.
- **Hosted multimodal APIs** (GPT-4V, Claude, Gemini) work well out of the box but require sending documents to a third party — not viable for confidential data.

The result is a real gap: there's no clean, locally-deployable system that treats prose, tables, and figures with the structural care each deserves and gives a user grounded, citable answers to questions over them.

This project is my attempt to close that gap.

---

## What I'm building

A **fully local multimodal RAG (Retrieval-Augmented Generation) system** that takes a technical PDF, extracts each kind of content using a dedicated specialised extractor, and indexes everything into a unified text-space index. A user can then ask natural-language questions and get answers grounded in the actual document — with citations back to the specific page, table, or figure.

The whole pipeline runs on-premise. No document ever leaves the network. Every model used — embedding, vision-language, generation — is an open-weights model downloaded and run locally.

A user-facing summary:

> *Drop a PDF in. Ask a question in plain English. Get an answer grounded in the document, with citations pointing to the exact source (page, table, figure). Works on confidential data because nothing leaves your machine.*

---

## How it works

The system has two phases — an **offline ingestion phase** that runs once per document, and an **online query phase** that runs every time a user asks something.

### Ingestion phase

For each page of a PDF:

1. **Detect document type.** Born-digital PDFs (text is directly extractable) take the fast path. Scanned pages — where the entire page is essentially an image — take a slower path through layout detection + OCR. (This second path is phase-2 work; the first build targets born-digital documents.)

2. **Route each page element to a specialised handler:**
   - **Prose** → standard text extraction with PyMuPDF, chunked using recursive character splitting.
   - **Tables** → structure-preserving extraction with pdfplumber. The tool detects ruled grid lines and outputs structured rows.
   - **Figures** → cropped and passed to a vision-language model (VLM) for captioning.

3. **Chunk each stream appropriately:**
   - **Prose chunks** are typical text passages.
   - **Tables** are chunked **per row** — each row becomes a self-contained chunk, prefixed with the table's column headers and a brief description of what the table covers. This is the central design decision for table handling and is explained below.
   - **Figures** become text chunks consisting of the VLM-generated caption.

4. **Carry metadata through everything.** Each chunk — regardless of source — is stored with its provenance: file path, page number, source type (prose / table / figure), and location on the page. Citations depend entirely on this metadata being captured at ingestion and threaded through the entire pipeline.

5. **Embed and store.** Each chunk is embedded using a small open-weights embedding model (`BAAI/bge-small-en-v1.5`) and stored in a vector database (ChromaDB) alongside its metadata.

### Query phase

When a user asks a question:

1. The question is embedded using the same embedding model used at ingestion.
2. The vector database returns the top-*k* chunks most semantically similar to the question.
3. The retrieved chunks (prose, table rows, figure captions — all mixed, all already text) are formatted into a prompt, each labelled with its source metadata.
4. A local LLM reads the prompt and generates an answer that cites which sources it used.
5. Source labels in the answer are mapped back to real file/page/figure references for display.

---

## The key design decisions

A few choices that define this project's character, with the reasoning behind each:

### Hybrid per-element routing, not end-to-end conversion

The alternative would be a single tool (MarkItDown, Docling) that turns the whole PDF into one block of Markdown or HTML. Hybrid routing preserves **spatial metadata** (which is required for citations and keeps vision-native retrieval as a future option) and gives **per-stream control** — different chunking strategies for different content types — which a black-box converter does not.

### Per-row chunking for tables, not whole-table chunks

A whole table embedded as a single vector produces an "averaged" representation — it matches every query weakly and no query strongly. By making each row a self-contained chunk (with column headers and a brief table description prepended), every row becomes individually queryable. The cost is that *cross-row* and *cross-table* reasoning is weaker — but those queries are rarer in practice than specific-value lookups, which is the common case for technical documents.

### Context bundling for figures

A figure on its own is ambiguous — the same line chart could be "turn radius vs. speed" or "fuel consumption vs. speed" depending on labels. To produce useful captions, the VLM receives not just the cropped image but also its surrounding **caption line**, the **section header** it sits under, and any **in-text reference** to the figure from the body text. This grounds the model in the actual context of the figure rather than asking it to guess from pixels alone.

### Unified text-space embedding

All three content types — prose, tables, figures — produce *text* (either extracted, structured, or generated) and live in the same vector index. The alternative — a separate vision index for figures and a text index for everything else — means two retrieval paths and more moving parts. A unified index keeps the system debuggable per stream and lets retrieval be evaluated cleanly. Vision-native retrieval (ColPali-style page-image embedding) remains an option for later, once the baseline is measured.

### Local open-weights models throughout

Confidentiality forces the entire stack to run locally — no hosted APIs. This is a hard constraint, not a preference. In 2024–2025, open-weights vision-language models (Gemma 3, Qwen2-VL, InternVL, MiniCPM-V) have closed most of the gap with hosted models on document tasks, which makes local-only viable for the first time.

### Incremental, evaluation-driven build

The system is built one component at a time — text and tables first, then figure captioning, then retrieval evaluation, then deferred extensions. The principle is *make it work, then make it good.* This makes failures easy to isolate, gives milestone visibility, and lets each subsequent stage be informed by measurements on the previous one rather than guesses.

---

## What's deferred

A few cases the baseline pipeline doesn't fully handle, intentionally:

- **Multi-page tables** — a table spanning two or more pages is treated as separate tables by naive extractors. Reconstructing them as one logical table is a known-hard problem and worth treating as a focused sub-problem after the baseline works.
- **Cross-row and cross-table reasoning** — questions like *"which speeds give a turn radius under 500m?"* require synthesising across rows of one table or between two tables. Per-row chunking is weak here by design. The mitigation is to add whole-table summaries and LLM-generated table descriptions as additional retrieval candidates — but only if evaluation shows it's needed.
- **Complex technical schematics** — figures with fine annotations, dotted-line trajectories, and small labels stretch the limits of current open-weights VLMs. Caption quality on simple instructional diagrams is good; on dense schematics it's an open question and a potential area for focused improvement.
- **Scanned-page handling at scale** — layout detection and OCR for scanned pages is designed but is phase-2 work, behind the born-digital baseline.

Each of these has a measurable trigger for revisiting it: *if* evaluation shows the baseline fails on this class of query, *then* the corresponding extension gets built. This keeps the project scope honest and prevents premature complexity.

---

## Why this approach

Three sentences that capture it:

1. **The hard part of multimodal RAG is not retrieval, it's representation.** What you put into the vector store determines what's retrievable; how you chunk tables and what context you bundle with figures matters more than which embedding model you use.

2. **The simplest pipeline that handles each content type with structural respect outperforms more sophisticated systems that treat the document as one undifferentiated blob.** This is the engineering bet at the heart of this project.

3. **Build incrementally, measure honestly, and let the data tell you when to add complexity** — rather than committing to fancy architectures (vision-native retrieval, agentic loops, multi-granularity chunking) before the baseline is even running.

---

## Current status

This project is at the start of its build phase. The architecture above is the design — defended in a first project review — and the next several weeks will be spent building it one phase at a time, with each phase measured against a hand-built evaluation set before the next phase begins.

This document will evolve as the system does. Components I haven't built yet are described in the design language above; once they're working, this overview will be replaced with measurements, examples, and concrete results.

---

*Built as part of a research internship. Approach and architecture are public; data and specific evaluation results are confidential.*