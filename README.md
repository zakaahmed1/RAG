# RAG Knowledge Assistant

A production-style Retrieval-Augmented Generation (RAG) application built in Python.

The project combines document ingestion, semantic embeddings, persistent FAISS vector search and a locally hosted Hugging Face language model to answer questions using information retrieved from an external document knowledge base.

The project is being developed incrementally from an initial RAG prototype into a more complete AI engineering application with persistent indexing, retrieval evaluation, source attribution, API serving, automated testing, observability and deployment support.

---

## Current Capabilities

- Multi-document ingestion
- TXT, Markdown, PDF and DOCX support
- Recursive document chunking
- Source and page metadata preservation
- Hugging Face sentence-transformer embeddings
- Normalised embeddings for cosine-similarity scoring
- Persistent FAISS vector storage
- Semantic similarity retrieval
- Configurable similarity thresholds
- Duplicate suppression
- Source and page-level citations
- Locally hosted FLAN-T5 generation
- Grounded prompting using retrieved context
- Unsupported-question abstention
- Quantitative retrieval evaluation
- Quantitative answer evaluation
- Tune/test benchmark separation
- Automated retrieval parameter sweeps
- Direct, paraphrased, multi-document, unsupported and adversarial benchmark questions

---

## Architecture

The system separates document indexing from question answering.

### Offline Ingestion

```text
Documents
   │
   ▼
Document Loaders
   │
   ▼
Recursive Chunking
   │
   ▼
Hugging Face Embeddings
   │
   ▼
FAISS Vector Index
   │
   ▼
Persistent Local Storage
```

### Online Query Pipeline

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Similarity Threshold
      │
      ▼
Duplicate Suppression
      │
      ▼
Retrieved Evidence
      │
      ▼
Source-Aware Prompt
      │
      ▼
FLAN-T5
      │
      ├────────► Grounded Answer
      │
      └────────► Sources + Similarity Scores
```

The application does not rebuild the FAISS index every time it starts. Documents are ingested separately and the persisted index is loaded when the RAG assistant starts.

---

## Project Structure

```text
rag-assistant/
│
├── app/
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── ingest.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── search.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── evaluate.py
│       └── sweep.py
│
├── data/
│   └── documents/
│       ├── EmployeeHandbook.pdf
│       ├── ExampleCompanyPolicy.txt
│       └── ITSecurityPolicy.txt
│
├── evaluation/
│   └── questions.json
│
├── results/
│
├── storage/
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
│
├── tests/
│
├── rag_app.py
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

Generated FAISS index files are excluded from source control and can be recreated from the source documents.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | FAISS |
| Generator | `google/flan-t5-base` |
| Model runtime | Hugging Face Transformers |
| PDF loading | PyPDF |
| DOCX loading | docx2txt |
| Evaluation | Custom Python evaluation framework |

---

## Supported Document Types

Documents placed inside:

```text
data/documents/
```

can currently use the following extensions:

```text
.txt
.md
.pdf
.docx
```

The ingestion pipeline recursively scans the document directory and loads all supported files.

Metadata such as the source filename, file type, source path, page number and chunk identifier is retained alongside each chunk.

This metadata is later used for deterministic source attribution.

---

## Installation

Use Python 3.10 or later.

Clone the repository and move into the project directory:

```bash
git clone https://github.com/zakaahmed1/RAG.git
cd RAG
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
sentence-transformers
faiss-cpu
transformers
huggingface-hub
accelerate
pypdf
docx2txt
```

---

## Document Ingestion

Add documents to:

```text
data/documents/
```

Then build or rebuild the vector index:

```bash
python -m app.ingestion.ingest
```

Example output:

```text
Starting document ingestion...
Loading: EmployeeHandbook.pdf
Loading: ExampleCompanyPolicy.txt
Loading: ITSecurityPolicy.txt
Created document chunks.
Creating FAISS vector store...
Saving FAISS vector store...
Ingestion complete.
```

The persisted index is stored in:

```text
storage/faiss_index/
```

and normally contains:

```text
index.faiss
index.pkl
```

Re-run ingestion whenever the document knowledge base changes.

---

## Running the RAG Assistant

Once an index has been created, start the application with:

```bash
python rag_app.py
```

Example:

```text
Loading persistent FAISS vector store...
Loading language model...
RAG assistant ready.

Ask a question (or type 'exit' to quit):
How many days annual leave do employees receive?

--- Answer ---
25 days.

--- Sources ---
[1] EmployeeHandbook.pdf, page 4 (similarity: 0.666)
```

The assistant retrieves relevant evidence from the persisted vector store and supplies that evidence to the local language model.

---

## Retrieval

The current production-style retrieval configuration is:

```python
RETRIEVAL_MODE = "similarity"

TOP_K = 4
FETCH_K = 12

MIN_SIMILARITY = 0.425
MMR_LAMBDA_MULT = 0.7
```

`MMR_LAMBDA_MULT` remains available for experiments using Maximum Marginal Relevance, although the benchmark-selected retrieval mode is currently similarity search.

Embeddings are normalised before storage and retrieval.

For unit-normalised vectors, the FAISS squared L2 distance can be converted into cosine similarity using:

```text
cosine_similarity = 1 - squared_l2_distance / 2
```

The resulting value is used as a semantic similarity score.

A similarity score is **not** an answer-confidence probability. It measures how semantically similar a query and document chunk are.

---

## Grounded Generation

The application currently uses:

```text
google/flan-t5-base
```

through Hugging Face Transformers.

Retrieved chunks are inserted into a grounded prompt which instructs the model to:

- answer only from supplied context
- avoid outside knowledge
- avoid inventing unsupported facts
- abstain when the supplied evidence is insufficient

When no retrieved chunks meet the configured similarity threshold, the application returns:

```text
I could not find sufficient information in the supplied documents.
```

Source attribution is handled by application metadata rather than relying on the language model to invent citations.

---

## Source Attribution

Retrieved chunks retain source metadata including:

```text
file name
page number
chunk ID
similarity score
```

For PDF documents, the application can therefore display citations such as:

```text
EmployeeHandbook.pdf, page 7
```

alongside the retrieved semantic similarity score.

This makes the retrieval path inspectable and provides traceability between generated answers and source material.

---

# Evaluation and Retrieval Benchmarking

The RAG pipeline includes a custom quantitative evaluation framework.

The purpose of the framework is to separate:

```text
retrieval quality
```

from:

```text
generation quality
```

so that failures can be attributed to the correct part of the RAG pipeline.

---

## Benchmark Design

The final benchmark contains 40 labelled questions:

| Question Type | Count |
|---|---:|
| Direct supported | 12 |
| Paraphrased supported | 10 |
| Cross-policy / multi-document | 5 |
| Unsupported | 8 |
| Ambiguous / adversarial | 5 |
| **Total** | **40** |

The benchmark was split before retrieval tuning into:

```text
30 tuning questions
10 holdout test questions
```

The holdout set was not used when selecting retrieval parameters.

This helps reduce the risk of selecting a configuration that simply overfits the evaluation questions.

---

## Evaluation Metrics

The framework measures:

### Hit Rate@K

Measures whether the required evidence appears somewhere in the retrieved top-K chunks.

### Evidence Recall@K

Measures how much of the labelled evidence was retrieved.

This is particularly important for multi-document questions.

For example:

```text
Required evidence sources: 2
Retrieved required sources: 1

Evidence Recall = 0.5
```

### Mean Reciprocal Rank

MRR measures how highly the first relevant result is ranked.

A relevant result at:

```text
rank 1 → reciprocal rank 1.0
rank 2 → reciprocal rank 0.5
rank 3 → reciprocal rank 0.333
```

### Unsupported-Query Rejection Accuracy

Measures how often unsupported questions produce no chunks above the configured semantic similarity threshold.

### Answer Keyword Coverage

Uses labelled expected-answer terms to provide a lightweight deterministic measure of whether generated answers contain the expected information.

Basic written-number normalisation is included so that values such as:

```text
one hour
```

and:

```text
1 hour
```

can be evaluated consistently.

### Unsupported-Answer Abstention Accuracy

Measures whether the generator correctly refuses to answer unsupported questions.

---

## Multi-Evidence Evaluation

Benchmark entries can use:

```json
"evidence_requirement": "any"
```

when any labelled source is sufficient.

For questions requiring multiple pieces of evidence:

```json
"evidence_requirement": "all"
```

is used.

For example, a question requiring confirmation from both:

```text
EmployeeHandbook.pdf
ITSecurityPolicy.txt
```

only receives a successful Hit Rate result if evidence from both sources is retrieved.

Unsupported questions use:

```json
"evidence_requirement": "none"
```

---

## Retrieval Parameter Tuning

A retrieval-only grid search evaluated **192 retrieval configurations**.

The experiment compared:

```text
Retrieval mode:
- similarity
- MMR

TOP_K:
- 3
- 4
- 5
- 6

FETCH_K:
- 12
- 20
- 30

Similarity threshold:
- 0.300
- 0.350
- 0.375
- 0.400
- 0.425
- 0.450
- 0.475
- 0.500
```

A composite tuning score considered:

```text
Hit Rate@K
Evidence Recall@K
Unsupported-query rejection accuracy
```

with MRR retained as an additional ranking metric.

Similarity search outperformed MMR on the current benchmark, particularly for cross-policy retrieval where multiple sources were required.

The selected configuration was:

```python
RETRIEVAL_MODE = "similarity"
TOP_K = 4
FETCH_K = 12
MIN_SIMILARITY = 0.425
```

---

## Tuning-Set Results

The selected configuration achieved the following results on the 30-question tuning set:

| Metric | Result |
|---|---:|
| Hit Rate@K | **1.000** |
| Evidence Recall@K | **1.000** |
| MRR | **0.958** |
| Unsupported-query rejection | **1.000** |

Category-level retrieval achieved complete Hit Rate and Evidence Recall across:

```text
direct supported
paraphrased supported
cross-policy / multi-document
ambiguous / adversarial
```

questions.

---

## Holdout Results

After freezing the retrieval configuration, the system was evaluated against the untouched 10-question holdout set.

The configuration was not changed after observing these results.

### Retrieval

| Metric | Result |
|---|---:|
| Supported questions | 8 |
| Unsupported questions | 2 |
| Hit Rate@K | **1.000** |
| Evidence Recall@K | **1.000** |
| MRR | **0.917** |
| Unsupported-query rejection | **0.500** |

All eight supported holdout questions retrieved all required evidence.

One of the two unsupported questions retrieved semantically related company-information content above the configured threshold.

The retrieval settings were deliberately **not retuned against this holdout failure**, preserving the integrity of the holdout evaluation.

### End-to-End Generation

| Metric | Result |
|---|---:|
| Supported answer keyword coverage | **0.875** |
| Supported false-abstention rate | **0.000** |
| Unsupported answer abstention | **1.000** |

Both unsupported holdout questions ultimately produced the expected abstention behaviour.

This demonstrates a layered RAG control:

```text
Unsupported query
       │
       ▼
Similarity threshold
       │
       ├── Rejected ──► Abstain
       │
       └── Retrieved evidence
                 │
                 ▼
          Grounded generator
                 │
                 ▼
              Abstain
```

---

## Full 40-Question Benchmark

After recording the holdout results, the frozen system was evaluated descriptively across all 40 benchmark questions.

| Metric | Result |
|---|---:|
| Supported questions | 32 |
| Unsupported questions | 8 |
| Hit Rate@K | **1.000** |
| Evidence Recall@K | **1.000** |
| MRR | **0.948** |
| Unsupported-query rejection | **0.875** |
| Supported answer keyword coverage | **0.776** |
| Supported false-abstention rate | **0.000** |
| Unsupported answer abstention | **1.000** |

Retrieval therefore remained consistently strong across the complete benchmark.

---

## Generation Performance by Question Type

Generation performance varied by question complexity:

| Category | Answer Keyword Coverage |
|---|---:|
| Direct supported | **0.958** |
| Paraphrased supported | **1.000** |
| Cross-policy / multi-document | **0.567** |
| Ambiguous / adversarial | **0.100** |

This shows that the primary limitation of the current system is no longer evidence retrieval.

FLAN-T5 Base performs well on straightforward and paraphrased factual questions but is less reliable when required to:

```text
combine multiple facts
reason across several sources
reject misleading assumptions
interpret adversarial yes/no questions
```

---

## Known Generation Failure

One holdout adversarial question asked:

```text
If a stolen company laptop has already been remotely wiped,
can the employee wait until the next day to report it?
```

The retriever correctly returned:

```text
EmployeeHandbook.pdf, page 6
ITSecurityPolicy.txt
```

including the policy stating that lost or stolen devices must be reported within two hours.

However, FLAN-T5 generated:

```text
Yes.
```

This was classified as a genuine **generation-stage failure**, not a retrieval failure.

The example is intentionally retained as a regression case for future generator and prompt improvements.

The holdout configuration was not changed after observing this failure.

---

## Evaluation Commands

Evaluate the tuning set using retrieval only:

```bash
python -m app.evaluation.evaluate --retrieval-only --split tune
```

Evaluate the holdout using retrieval only:

```bash
python -m app.evaluation.evaluate --retrieval-only --split test
```

Run the full holdout evaluation:

```bash
python -m app.evaluation.evaluate --split test
```

Evaluate all 40 benchmark questions:

```bash
python -m app.evaluation.evaluate
```

Run the retrieval parameter sweep:

```bash
python -m app.evaluation.sweep
```

Evaluation outputs are written to:

```text
results/
```

with separate files for tuning, holdout and complete benchmark runs.

---

## Current Limitations

The project is intentionally still under development.

Current limitations include:

- FLAN-T5 Base can fail on complex or adversarial reasoning even when correct evidence has been retrieved.
- Multi-document synthesis is weaker than simple factual question answering.
- Similarity scores measure semantic similarity and must not be interpreted as calibrated answer-confidence probabilities.
- Semantically related unsupported questions can occasionally pass the retrieval threshold.
- The current benchmark is synthetic and relatively small.
- Benchmark results should not be interpreted as production performance guarantees.
- Answer keyword coverage is a lightweight deterministic metric and does not capture every form of semantic correctness.
- The current system uses local FAISS rather than a distributed vector database.
- The current application is command-line based.
- Authentication, API rate limiting and production security controls have not yet been implemented.
- Full observability and request tracing have not yet been implemented.

---

## Security Note

FAISS persistence uses Python pickle metadata.

The application therefore uses:

```python
allow_dangerous_deserialization=True
```

when loading the locally generated FAISS store.

The project should only load vector indexes generated locally or obtained from a trusted source.

Untrusted `.pkl` files should never be loaded.

---

## Hugging Face Authentication

The application can run without a Hugging Face access token.

Without a token, Hugging Face may display:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is not an application failure.

A Hugging Face token can optionally be configured to increase download rate limits.

Secrets should be supplied through environment variables and should never be committed to source control.

---

## Development Roadmap

The project is being developed in phases.

### Completed

```text
Phase 1
Modular RAG refactor

Phase 2
Multi-document ingestion and persistent FAISS storage

Phase 3
Scored retrieval, thresholds, source metadata and citations

Phase 4
Quantitative RAG evaluation and retrieval optimisation
```

### Planned

```text
Phase 5
FastAPI service layer

Phase 6
Streamlit user interface

Phase 7
Automated testing with pytest

Phase 8
Docker packaging

Phase 9
GitHub Actions CI/CD

Phase 10
Structured logging and observability

Phase 11
Security and responsible-AI controls

Phase 12
Generator improvements and regression evaluation
```

Future generation work will use the current adversarial failures as regression tests rather than modifying the existing holdout benchmark retrospectively.

---

## Engineering Principles

The project follows several principles intended to make RAG behaviour more inspectable and defensible:

```text
Measure retrieval separately from generation.

Preserve source metadata throughout the pipeline.

Prefer deterministic source attribution over model-generated citations.

Treat semantic similarity as retrieval evidence, not answer confidence.

Evaluate unsupported questions as well as supported questions.

Separate tuning data from holdout evaluation data.

Record real failures rather than tuning them away after observing holdout results.

Optimise retrieval empirically rather than selecting parameters by intuition.

Keep ingestion and serving paths separate.

Prefer simple, measurable architecture before adding unnecessary infrastructure.
```

---

## Repository

GitHub:

```text
https://github.com/zakaahmed1/RAG
```

---

## Status

Current development status:

```text
Phases 1–4 complete.

Persistent multi-document RAG:
Complete

Scored semantic retrieval:
Complete

Source/page attribution:
Complete

Quantitative benchmark:
Complete

Retrieval parameter optimisation:
Complete

Holdout evaluation:
Complete

FastAPI service:
Next
```