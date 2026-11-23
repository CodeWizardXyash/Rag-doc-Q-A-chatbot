# RAG Document Q&A Chatbot

Upload a PDF and ask questions about it. Answers are generated **only** from the document and come with **page citations**. If the answer isn't in the PDF, the bot says so instead of guessing.

**Stack:** Python · FastAPI · ChromaDB · Google Gemini (or OpenAI) · Streamlit · Docker

## How it works

```
 PDF ──► pypdf extract ──► chunk (overlap) ──► embed ──► ChromaDB
                                                            │
 Question ──► embed ──► top-k similarity search ◄───────────┘
                              │
                 grounded prompt (context + rules)
                              │
                         LLM answer + [Page N] citations
```

## Project structure

```
app/
  main.py                 FastAPI app: /health, /upload, /ask, /documents
  config.py               Settings from env vars (pydantic-settings)
  schemas.py              Request/response + internal models
  exceptions.py           Domain errors -> HTTP status codes
  services/
    pdf_processor.py      Text extraction + chunking
    vector_store.py       ChromaDB: embeddings, upsert, semantic search
    rag_engine.py         Grounded prompt + LLM call + citations
frontend/app.py           Streamlit chat UI
tests/test_api.py         pytest suite (no API key needed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                 # then add your GEMINI_API_KEY (or OPENAI_API_KEY)
```

## Run

```bash
# Terminal 1: backend  (API docs at http://localhost:8000/docs)
uvicorn app.main:app --reload

# Terminal 2: frontend (http://localhost:8501)
streamlit run frontend/app.py
```

## Docker (backend)

```bash
docker build -t rag-qa-api .
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/code/data" rag-qa-api
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + number of indexed chunks |
| POST | `/upload` | Upload a PDF (multipart `file`) -> `doc_id` |
| POST | `/ask` | `{"question": "...", "doc_id": null}` -> answer + sources |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Remove a document |

Errors: `400` invalid/empty/corrupt PDF · `404` unknown document · `413` file too large · `415` not a PDF · `422` no extractable text · `501` a core RAG step is still a TODO · `502` LLM provider failure.

## Tests

```bash
pytest -v
```

## Core RAG pieces

| # | Piece | File |
|---|-------|------|
| 1 | Chunking with overlap | `app/services/pdf_processor.py` |
| 2 | Embeddings + upsert to ChromaDB | `app/services/vector_store.py` |
| 3 | Semantic similarity search | `app/services/vector_store.py` |
| 4 | Grounded, citation-enforcing prompt | `app/services/rag_engine.py` |

## Known limitations

- Scanned PDFs (images) are rejected; OCR is not implemented.
- No authentication or rate limiting; add both before exposing publicly.
