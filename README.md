# Insurance Hybrid RAG Chatbot

A modular FastAPI application that answers insurance questions using the existing Hybrid RAG pipeline:

`PDF -> Unstructured API -> chunk_by_title() -> filter chunks under 50 characters -> BGE dense retrieval + Pinecone hosted sparse retrieval -> RRF -> BGE cross-encoder reranking -> Groq answer`

Conversation memory is maintained per session with LangChain. Voice requests are transcribed with Whisper and use the same retrieval and answer flow as text chat.

## Live deploy

https://insurance-rag-chatbot-production.up.railway.app/

## Project structure

```text
src/
  ingestion/      Unstructured PDF loading helpers
  embeddings/     BAAI/bge-base-en-v1.5 adapter
  vectorstores/   Separate dense and sparse Pinecone access
  retrieval/      Dense/sparse orchestration and reciprocal-rank fusion
  reranker/       BAAI/bge-reranker-base adapter
  memory/         Session-scoped LangChain memory
  llm/            Groq answer generation
  voice/          Whisper transcription
  services/       Chat application service
config/           YAML configuration and global settings
evaluation/       Reproducible retrieval/answer metric scripts
notebooks/        Original experimentation notebook
static/           Existing frontend assets
tests/            Pytest coverage
```

## Requirements

- Python 3.10+
- Pinecone, Groq, and Unstructured API keys
- Docker Desktop only if running the container image

## Local setup

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set these values in `.env`:

```dotenv
PINECONE_API_KEY=...
GROQ_API_KEY=...
UNSTRUCTURED_API_KEY=...
```

## Build the indexes

Place PDF documents in the configured `Data/insurance_documents` directory, then run:

```powershell
python main.py
```

The command creates the configured separate Pinecone dense and hosted sparse indexes when absent, assigns matching deterministic IDs to both, and uploads the chunk records.

## Retrieval Top-K settings

`config/config.yaml` controls each retrieval stage independently:

- `dense_top_k: 20` — dense Pinecone results
- `sparse_top_k: 20` — hosted sparse Pinecone results
- `rrf_top_k: 10` — fused RRF candidates sent to the cross encoder
- `final_top_k: 5` — reranked documents sent to the LLM

## Run the API

```powershell
python app.py
```

The retained frontend is available at `http://localhost:8000/`, interactive API documentation at `http://localhost:8000/docs`, and health status at `http://localhost:8000/health`.

## API endpoints

- `POST /chat` — JSON body: `{"question": "...", "session_id": "optional"}`
- `POST /voice-chat` — multipart `audio` file with an optional session ID
- `POST /reset-memory` — JSON body: `{"session_id": "..."}`
- `GET /health`

## Docker

Docker Compose is not needed because this project runs as one API container and
uses managed external services. Build and run the image directly:

```powershell
docker build -t insurance-rag-chatbot .
docker run --rm -p 8000:8000 --env-file .env insurance-rag-chatbot
```

## Validation

```powershell
python -m pytest tests -q
python -m evaluation.run
```

The evaluation module reports Context Precision, Context Recall, Faithfulness, and Answer Relevancy for labelled samples. These are lightweight keyword/token-overlap approximations (see `evaluation/metrics.py`), not LLM-judged RAGAS metrics — treat them as a fast offline sanity check, not a substitute for a real RAGAS evaluation. Use your own labelled retrieval outputs for live quality evaluation.
