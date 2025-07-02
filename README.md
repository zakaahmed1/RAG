# RAG App – Retrieval-Augmented Generation

A simple RAG application built in Python which utilises Generative AI models to answer customer queries using an external data source.

## 🚀 Features

- 🔎 Vector-based semantic search using FAISS or ChromaDB
- 📖 Document ingestion and chunking (e.g., PDFs, Markdown, HTML)
- 🧠 LLM integration using Hugging Face models
- 🤖 Retrieval-Augmented Q&A with context injection

---

## 🛠️ Tech Stack

- Python
- LangChain
- Hugging Face Transformers
- FAISS
- dotenv for managing environment variables

---

## Usage

1. **Install dependencies**

   Use Python 3.10 or later and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**

   Execute the main script which will ingest `ExampleCompanyPolicy.txt` and start an interactive question loop:

   ```bash
   python rag_app.py
   ```

3. **Ask questions**

   When prompted, type a question about the document. Type `exit` to quit.

   ```text
   Ask a question (or type 'exit' to quit): What is the company policy on remote work?
   ```

   The app retrieves relevant document sections, sends them to a lightweight LLM, and prints the generated answer.
