RAG Knowledge Assistant

A production-style Retrieval-Augmented Generation (RAG) application built in Python.

The project combines multi-document ingestion, semantic embeddings, persistent FAISS vector search, grounded local language-model generation, quantitative evaluation, a FastAPI backend, a Streamlit frontend, automated testing and Docker-based deployment.

The application has been developed incrementally from an initial RAG prototype into a more complete AI engineering system with measurable retrieval quality, deterministic source attribution and a separated frontend/backend architecture.

Current Capabilities

Multi-document ingestion

TXT, Markdown, PDF and DOCX support

Recursive document chunking

Source, page and chunk metadata preservation

Hugging Face sentence-transformer embeddings

Normalised embeddings for cosine-similarity scoring

Persistent FAISS vector storage

Semantic similarity retrieval

Configurable similarity thresholds

Duplicate suppression

Source and page-level citations

Locally hosted FLAN-T5 generation

Grounded prompting using retrieved context

Unsupported-question abstention

Quantitative retrieval evaluation

Quantitative answer evaluation

Tune/test benchmark separation

Automated retrieval parameter sweeps

Direct, paraphrased, multi-document, unsupported and adversarial benchmark questions

FastAPI service layer

Structured request/response schemas

API health and status endpoints

Streamlit chat interface

Source and similarity-score display in the UI

Automated pytest test suite

Real FAISS regression testing

Docker containerisation

Docker Compose orchestration

FastAPI container health checks

Persistent Hugging Face model cache

Environment-based frontend/backend configuration

Architecture

The system separates document ingestion, retrieval, generation, API serving and frontend presentation.

Offline Ingestion

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

Online Query Pipeline

User Question
      │
      ▼
Streamlit UI / API Client
      │
      ▼
FastAPI
      │
      ▼
RAG Service
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

The application does not rebuild the FAISS index every time it starts. Documents are ingested separately and the persisted index is loaded when the RAG service starts.

The FastAPI service owns the FAISS vector store and generation model. The Streamlit frontend communicates with FastAPI over HTTP rather than loading its own independent copy of the RAG pipeline.

Container Architecture

Docker Compose runs the frontend and backend as separate services using the same application image.

Browser
   │
   ▼
localhost:8501
   │
   ▼
┌─────────────────────┐
│ Streamlit Container │
└──────────┬──────────┘
           │
           │ http://api:8000
           ▼
┌─────────────────────┐
│ FastAPI Container   │
│ FAISS + FLAN-T5     │
└──────────┬──────────┘
           │
           ▼
Mounted FAISS Index

The Streamlit container receives:

RAG_API_BASE_URL=http://api:8000

through Docker Compose.

The same Streamlit code therefore works locally and in Docker without hard-coded deployment-specific URLs.

Project Structure

RAG/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── service.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py
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
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_metrics.py
│   ├── test_regression.py
│   ├── test_retrieval.py
│   └── test_service.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yaml
├── pytest.ini
├── rag_app.py
├── requirements.txt
└── README.md

Generated FAISS index files are excluded from source control and can be recreated from the source documents.

Technology Stack

Component

Technology

Language

Python

RAG framework

LangChain

Embeddings

Hugging Face Sentence Transformers

Embedding model

sentence-transformers/all-MiniLM-L6-v2

Vector database

FAISS

Generator

google/flan-t5-base

Model runtime

Hugging Face Transformers

API

FastAPI

API server

Uvicorn

Frontend

Streamlit

HTTP client

Requests

Validation

Pydantic

Testing

pytest

Coverage

pytest-cov

API test client

HTTPX / FastAPI TestClient

Containerisation

Docker

Multi-container orchestration

Docker Compose

PDF loading

PyPDF

DOCX loading

docx2txt

Evaluation

Custom Python evaluation framework

Supported Document Types

Documents placed inside:

data/documents/

can currently use the following extensions:

.txt
.md
.pdf
.docx

The ingestion pipeline recursively scans the document directory and loads all supported files.

Metadata such as source filename, file type, source path, page number and chunk identifier is retained alongside each chunk.

This metadata is later used for deterministic source attribution.

Installation

Python 3.13 is used by the current Docker image and has been used during local development.

Clone the repository:

git clone https://github.com/zakaahmed1/RAG.git
cd RAG

Install dependencies:

python -m pip install -r requirements.txt

On Windows systems where Python is exposed through the launcher instead:

py -m pip install -r requirements.txt

The main dependencies include LangChain, sentence-transformers, FAISS, Transformers, FastAPI, Uvicorn, Streamlit, pytest, PyPDF and docx2txt.

Document Ingestion

Add documents to:

data/documents/

Then build or rebuild the vector index:

python -m app.ingestion.ingest

On Windows with the Python launcher:

py -m app.ingestion.ingest

Example output:

Starting document ingestion...
Loading: EmployeeHandbook.pdf
Loading: ExampleCompanyPolicy.txt
Loading: ITSecurityPolicy.txt
Created document chunks.
Creating FAISS vector store...
Saving FAISS vector store...
Ingestion complete.

The persisted index is stored in:

storage/faiss_index/

and normally contains:

index.faiss
index.pkl

Re-run ingestion whenever the document knowledge base changes.

Running the CLI

The original CLI remains available.

Once an index has been created:

python rag_app.py

Example:

Loading persistent FAISS vector store...
Loading language model...
RAG assistant ready.

Ask a question (or type 'exit' to quit):
How many days annual leave do employees receive?

--- Answer ---
25 days.

--- Sources ---
[1] EmployeeHandbook.pdf, page 4 (similarity: 0.666)

The CLI remains useful for direct local testing of the RAG pipeline independently from FastAPI and Streamlit.

Retrieval

The current frozen retrieval configuration is:

RETRIEVAL_MODE = "similarity"

TOP_K = 4
FETCH_K = 12

MIN_SIMILARITY = 0.425
MMR_LAMBDA_MULT = 0.7

MMR_LAMBDA_MULT remains available for experiments using Maximum Marginal Relevance, although the benchmark-selected production-style retrieval mode is similarity search.

Embeddings are normalised before storage and retrieval.

For unit-normalised vectors, FAISS squared L2 distance can be converted into cosine similarity using:

cosine_similarity = 1 - squared_l2_distance / 2

The resulting value is used as a semantic similarity score.

A similarity score is not an answer-confidence probability. It measures how semantically similar a query and document chunk are.

Grounded Generation

The application currently uses:

google/flan-t5-base

through Hugging Face Transformers.

Retrieved chunks are inserted into a grounded prompt which instructs the model to:

answer only from supplied context

avoid outside knowledge

avoid inventing unsupported facts

abstain when the supplied evidence is insufficient

When evidence is insufficient, the application uses the shared abstention response:

I could not find sufficient information in the supplied documents.

Source attribution is handled by application metadata rather than relying on the language model to generate citations.

Source Attribution

Retrieved chunks retain source metadata including:

file name
page number
chunk ID
similarity score

For PDF documents, the application can therefore display citations such as:

EmployeeHandbook.pdf, page 7

alongside semantic similarity scores.

This makes the retrieval path inspectable and provides traceability between generated answers and source material.

Evaluation and Retrieval Benchmarking

The RAG pipeline includes a custom quantitative evaluation framework.

The framework deliberately separates:

retrieval quality

from:

generation quality

so that failures can be attributed to the correct part of the RAG pipeline.

Benchmark Design

The final benchmark contains 40 labelled questions:

Question Type

Count

Direct supported

12

Paraphrased supported

10

Cross-policy / multi-document

5

Unsupported

8

Ambiguous / adversarial

5

Total

40

The benchmark was split before retrieval tuning into:

30 tuning questions
10 holdout test questions

The holdout set was not used when selecting retrieval parameters.

This reduces the risk of selecting a configuration that simply overfits the benchmark.

Evaluation Metrics

Hit Rate@K

Measures whether the required evidence appears somewhere in the retrieved top-K chunks.

A result counts only when its source/page metadata and labelled supporting text all match.

Evidence Recall@K

Measures how much of the labelled evidence was retrieved.

This is particularly important for multi-document questions.

For example:

Required evidence sources: 2
Retrieved required sources: 1

Evidence Recall = 0.5

Mean Reciprocal Rank

MRR measures how highly the first relevant result is ranked.

rank 1 → reciprocal rank 1.0
rank 2 → reciprocal rank 0.5
rank 3 → reciprocal rank 0.333

Unsupported-Query Rejection Accuracy

Measures how often unsupported questions produce no chunks above the configured semantic similarity threshold.

Answer Keyword Coverage

Uses labelled expected-answer terms to provide a lightweight deterministic measure of whether generated answers contain the expected information.

Terms and phrases are matched on normalized token boundaries, so a label such as `1` does not incorrectly match `10`, `15` or `150`.

Basic written-number normalisation is included so that values such as:

one hour

and:

1 hour

can be evaluated consistently.

Unsupported-Answer Abstention Accuracy

Measures whether the generator correctly refuses to answer unsupported questions.

Multi-Evidence Evaluation

Each supported benchmark entry defines `required_evidence` as independently required groups. Every group contains one or more `alternatives`.

For example, one factual question may accept the same supporting statement from either the handbook or security policy. Those sources are alternatives inside one group. A cross-policy comparison instead uses two groups, because evidence from both documents is independently required.

Each alternative records:

- source file
- page, when applicable
- exact supporting text that must occur in the retrieved chunk

Unsupported questions use an empty `required_evidence` list.

Evaluation summaries also record run provenance, including the Git commit, dirty-worktree state, timestamp, dataset hash, Python version, model revisions, chunking settings, retrieval settings and generation parameters.

Retrieval Parameter Tuning

A retrieval-only grid search evaluated 192 retrieval configurations.

The experiment compared:

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

A composite tuning score considered:

Hit Rate@K
Evidence Recall@K
Unsupported-query rejection accuracy

with MRR retained as an additional ranking metric.

Similarity search outperformed MMR on the current benchmark, particularly for cross-policy retrieval where multiple sources were required.

The selected configuration was:

RETRIEVAL_MODE = "similarity"
TOP_K = 4
FETCH_K = 12
MIN_SIMILARITY = 0.425

Tuning-Set Results

The selected configuration achieved the following results on the 30-question tuning set:

Metric

Result

Hit Rate@K

1.000

Evidence Recall@K

1.000

MRR

0.958

Unsupported-query rejection

1.000

Category-level retrieval achieved complete Hit Rate and Evidence Recall across direct, paraphrased, cross-policy and adversarial supported questions.

Holdout Results

After freezing the retrieval configuration, the system was evaluated against the untouched 10-question holdout set.

The configuration was not changed after observing these results.

Retrieval

Metric

Result

Supported questions

8

Unsupported questions

2

Hit Rate@K

1.000

Evidence Recall@K

1.000

MRR

0.917

Unsupported-query rejection

0.500

All eight supported holdout questions retrieved all required evidence.

One of the two unsupported questions retrieved semantically related company-information content above the configured threshold.

The retrieval settings were deliberately not retuned against this holdout failure, preserving the integrity of the holdout evaluation.

End-to-End Generation

Metric

Result

Supported answer keyword coverage

0.875

Supported false-abstention rate

0.000

Unsupported answer abstention

1.000

Both unsupported holdout questions ultimately produced the expected abstention behaviour.

This demonstrates a layered RAG control:

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

Full 40-Question Benchmark

After recording the holdout results, the frozen system was evaluated descriptively across all 40 benchmark questions.

Metric

Result

Supported questions

32

Unsupported questions

8

Hit Rate@K

1.000

Evidence Recall@K

1.000

MRR

0.948

Unsupported-query rejection

0.875

Supported answer keyword coverage

0.776

Supported false-abstention rate

0.000

Unsupported answer abstention

1.000

Retrieval therefore remained consistently strong across the complete benchmark.

Generation Performance by Question Type

Generation performance varied by question complexity:

Category

Answer Keyword Coverage

Direct supported

0.958

Paraphrased supported

1.000

Cross-policy / multi-document

0.567

Ambiguous / adversarial

0.100

This shows that the primary limitation of the current system is no longer evidence retrieval.

FLAN-T5 Base performs well on straightforward and paraphrased factual questions but is less reliable when required to:

combine multiple facts
reason across several sources
reject misleading assumptions
interpret adversarial yes/no questions

Known Generation Failure

One holdout adversarial question asked:

If a stolen company laptop has already been remotely wiped,
can the employee wait until the next day to report it?

The retriever correctly returned:

EmployeeHandbook.pdf, page 6
ITSecurityPolicy.txt

including the policy stating that lost or stolen devices must be reported within two hours.

However, FLAN-T5 generated:

Yes.

This was classified as a genuine generation-stage failure, not a retrieval failure.

The example is intentionally retained as a regression case for future generator and prompt improvements.

The holdout configuration was not changed after observing this failure.

Evaluation Commands

Evaluate the tuning set using retrieval only:

python -m app.evaluation.evaluate --retrieval-only --split tune

Evaluate the holdout using retrieval only:

python -m app.evaluation.evaluate --retrieval-only --split test

Run the full holdout evaluation:

python -m app.evaluation.evaluate --split test

Evaluate all 40 benchmark questions:

python -m app.evaluation.evaluate

Run the retrieval parameter sweep:

python -m app.evaluation.sweep

Evaluation outputs are written to:

results/

with separate files for tuning, holdout and complete benchmark runs.

FastAPI Service

The RAG pipeline is exposed through a FastAPI service.

The backend loads the persisted FAISS vector store and FLAN-T5 model once during application startup and reuses them across requests.

This avoids repeatedly loading heavyweight resources.

Start FastAPI locally with:

python -m uvicorn app.api.app:app --host 127.0.0.1 --port 8000

API Endpoints

GET  /health
GET  /status
POST /query
GET  /docs
GET  /redoc

GET /health

Basic liveness check.

Example response:

{
  "status": "ok"
}

GET /status

Returns service readiness and active RAG configuration.

Example:

{
  "ready": true,
  "retrieval_mode": "similarity",
  "top_k": 4,
  "fetch_k": 12,
  "min_similarity": 0.425,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "generator_model": "google/flan-t5-base"
}

POST /query

Example request:

{
  "question": "How many days annual leave do employees receive?"
}

Example response:

{
  "question": "How many days annual leave do employees receive?",
  "answer": "25 days.",
  "retrieval_rejected": false,
  "retrieved_count": 4,
  "sources": [
    {
      "file": "EmployeeHandbook.pdf",
      "page": 4,
      "chunk_id": 8,
      "similarity": 0.68
    }
  ]
}

Blank or whitespace-only questions are rejected with HTTP 422.

Swagger documentation is automatically available at:

http://127.0.0.1:8000/docs

Streamlit Frontend

The project includes a Streamlit chat interface that communicates with FastAPI over HTTP.

Start FastAPI first, then in a second terminal run:

python -m streamlit run app/ui/streamlit_app.py

The UI is normally available at:

http://localhost:8501

The frontend provides:

chat-style question and answer interaction

API online/offline state

backend readiness state

active retrieval configuration

embedding and generator model information

source file and page display

chunk IDs

similarity scores

retrieval-rejection messaging

conversation display history

clear-conversation control

graceful API connection error handling

The conversation history is display state only. Each RAG query remains independent; the system does not currently implement multi-turn conversational memory.

Environment-Based API Configuration

The Streamlit application uses:

API_BASE_URL = os.getenv(
    "RAG_API_BASE_URL",
    "http://127.0.0.1:8000",
)

When running locally, no environment variable is required.

When running in Docker Compose:

RAG_API_BASE_URL=http://api:8000

is supplied automatically.

This allows the same frontend code to work in different environments without changing the source code.

Automated Testing

The project uses pytest for unit, API and integration testing.

The current suite contains 22 automated tests.

Run the fast suite:

python -m pytest -m "not integration" -v

Current fast-suite result:

21 passed
1 integration test deselected

Run the complete suite:

python -m pytest -v

Current complete-suite result:

22 passed

The complete suite includes a real FAISS retrieval regression test.

Test Coverage Areas

The test suite covers:

evaluation metrics
number normalisation
evidence matching
evidence recall
Hit Rate@K behaviour
MRR behaviour
abstention detection
retrieval similarity conversion
retrieval threshold filtering
empty-query validation
service-layer abstention
service-layer generation
service readiness checks
FastAPI health endpoint
FastAPI status endpoint
FastAPI query endpoint
FastAPI validation
real persisted-FAISS retrieval regression

AA003 Regression Test

The known adversarial AA003 example is retained as a retrieval regression case.

The integration test verifies that the correct evidence from:

EmployeeHandbook.pdf, page 6
ITSecurityPolicy.txt

continues to be retrieved.

The generation failure itself remains documented for future Phase 12 work.

Coverage Report

Run:

python -m pytest -m "not integration" --cov=app --cov-report=term-missing

Coverage is interpreted as a diagnostic metric rather than a target to maximise artificially. Command-line runners, evaluation scripts and Streamlit execution paths are intentionally less represented in the fast unit suite.

Docker

The application is containerised using Docker and Docker Compose.

The image uses:

FROM python:3.13-slim

FastAPI and Streamlit use the same built application image but run with different commands.

Build

Ensure the FAISS index has already been created:

python -m app.ingestion.ingest

Then build the application image:

docker compose build

Start

Run both services:

docker compose up

Or run them in the background:

docker compose up -d

Once healthy:

FastAPI:
http://localhost:8000

Streamlit:
http://localhost:8501

Inspect Services

docker compose ps

The API service includes a health check against:

http://127.0.0.1:8000/health

The UI service waits for the API service to become healthy before starting.

Logs

FastAPI logs:

docker compose logs -f api

Streamlit logs:

docker compose logs -f ui

All logs:

docker compose logs -f

Stop

docker compose down

The Docker Compose configuration uses a named Hugging Face cache volume so downloaded models can survive normal container recreation.

Avoid:

docker compose down -v

unless the Hugging Face cache volume should also be deleted.

Docker Storage Strategy

Application code is built into the Docker image.

The FAISS index is mounted into the API container as read-only runtime data:

./storage/faiss_index:/app/storage/faiss_index:ro

Hugging Face model downloads are stored in a named Docker volume:

huggingface_cache

The deployment separation is therefore:

Application code       → Docker image
FAISS runtime data     → read-only host mount
Model cache            → Docker named volume
Configuration          → environment variables

This avoids baking the generated FAISS index or downloaded model cache directly into the application image.

Running Tests in Docker

The fast test suite can also be executed inside the Linux container:

docker compose run --rm api python -m pytest -m "not integration" -v

The full suite can be run with:

docker compose run --rm api python -m pytest -v

This provides an additional environment-independence check for a project developed primarily on Windows.

Current Limitations

Current limitations include:

FLAN-T5 Base can fail on complex or adversarial reasoning even when correct evidence has been retrieved.

Multi-document synthesis is weaker than simple factual question answering.

Similarity scores measure semantic similarity and must not be interpreted as calibrated answer-confidence probabilities.

Semantically related unsupported questions can occasionally pass the retrieval threshold.

The current benchmark is synthetic and relatively small.

Benchmark results should not be interpreted as production performance guarantees.

Answer keyword coverage is a lightweight deterministic metric and does not capture every form of semantic correctness.

The system uses local FAISS rather than a distributed vector database.

Authentication and API rate limiting have not yet been implemented.

Structured production logging, tracing and monitoring have not yet been implemented.

Prompt-injection and malicious-document defences have not yet been fully hardened.

The application is containerised but is not yet deployed to a managed cloud production environment.

Real production SLAs, multi-user load testing and operational ownership are outside the current project scope.

Security Note

FAISS persistence uses Python pickle metadata.

The application therefore uses:

allow_dangerous_deserialization=True

when loading the locally generated FAISS store.

The project should only load vector indexes generated locally or obtained from a trusted source.

Untrusted .pkl files should never be loaded.

The Docker configuration mounts the FAISS index read-only into the API container.

Hugging Face Authentication

The application can run without a Hugging Face access token.

Without a token, Hugging Face may display:

Warning: You are sending unauthenticated requests to the HF Hub.

This is not an application failure.

A Hugging Face token can optionally be configured to increase download rate limits.

Secrets should be supplied through environment variables and should never be committed to source control.

Development Roadmap

The project is being developed in phases.

Completed

Phase 1
Modular RAG refactor

Phase 2
Multi-document ingestion and persistent FAISS storage

Phase 3
Scored retrieval, thresholds, source metadata and citations

Phase 4
Quantitative RAG evaluation and retrieval optimisation

Phase 5
FastAPI service layer

Phase 6
Streamlit user interface

Phase 7
Automated testing with pytest

Phase 8
Docker containerisation and Docker Compose orchestration

Remaining

Phase 9
GitHub Actions CI/CD

Phase 10
Structured logging and observability

Phase 11
Security and responsible-AI hardening

Phase 12
Generator improvements and regression evaluation

Future generation work will use current adversarial failures as regression tests rather than modifying the existing holdout benchmark retrospectively.

Engineering Principles

The project follows several principles intended to make RAG behaviour more inspectable and defensible:

Measure retrieval separately from generation.

Preserve source metadata throughout the pipeline.

Prefer deterministic source attribution over model-generated citations.

Treat semantic similarity as retrieval evidence, not answer confidence.

Evaluate unsupported questions as well as supported questions.

Separate tuning data from holdout evaluation data.

Record real failures rather than tuning them away after observing holdout results.

Optimise retrieval empirically rather than selecting parameters by intuition.

Keep ingestion and serving paths separate.

Load heavyweight runtime resources once and reuse them.

Keep frontend and backend responsibilities separated.

Use environment configuration rather than hard-coded deployment URLs.

Use automated tests to protect deterministic application behaviour.

Keep generated vector indexes outside source control.

Prefer measurable architecture before adding unnecessary infrastructure.

Repository

GitHub:

https://github.com/zakaahmed1/RAG

Status

Current development status:

Phases 1–8 complete.

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
Complete

Streamlit frontend:
Complete

Automated pytest suite:
Complete

Docker containerisation:
Complete

GitHub Actions CI/CD:
Next

The project currently represents a production-style RAG application architecture. It should not be interpreted as a fully operated production service until deployment, access controls, monitoring, production security controls and operational ownership are added
