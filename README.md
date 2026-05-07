# RAG-Powered Document Q&A System

Python + LangChain + OpenAI + FastAPI + Pinecone + Docker (Lambda-compatible).

## Features

- End-to-end RAG pipeline over private documents
- Pinecone vector search for low-latency retrieval
- FastAPI inference API with SSE streaming endpoint
- Session-based conversation memory (SQLite)
- API key auth + in-memory rate limiting
- Dockerized runtime compatible with AWS Lambda container images
- Offline evaluation script for answer relevance benchmarking

## Project Structure

- `app/main.py`: FastAPI app and routes
- `app/rag_service.py`: Ingestion, retrieval, prompt building, streaming generation
- `app/vectorstore.py`: Pinecone index bootstrap
- `app/config.py`: Environment-driven settings
- `scripts/evaluate.py`: Relevance benchmark utility

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure env:

```bash
copy .env.example .env
```

4. Fill `.env` with:
   - `APP_API_KEY` (required for protected routes)
   - OpenAI and Pinecone keys

## Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

## Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest ^
  -H "x-api-key: change-me" ^
  -H "Content-Type: application/json" ^
  -d "{\"source_dir\":\"./docs\",\"glob_pattern\":\"**/*.*\"}"
```

## Ask a Question (Non-streaming)

```bash
curl -X POST http://localhost:8000/ask ^
  -H "x-api-key: change-me" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the SLA?\",\"session_id\":\"demo-1\"}"
```

## Ask a Question (Streaming SSE)

```bash
curl -N -X POST http://localhost:8000/ask/stream ^
  -H "x-api-key: change-me" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Summarize section 2\",\"session_id\":\"demo-1\"}"
```

## Docker

```bash
docker compose up --build
```

## AWS Lambda Deployment (Container)

1. Build and push image to ECR.
2. Create Lambda function from container image.
3. Set environment variables from `.env`.
4. Attach API Gateway (HTTP API) for public inference endpoint.

You can also deploy via AWS SAM:

```bash
sam build -t deployment/sam-template.yaml
sam deploy --guided
```

Monitoring starter dashboard JSON is available at `deployment/cloudwatch-dashboard.json`.

## Evaluation

Create a JSONL file:

```json
{"query":"What is refund policy?","expected":"..."}
```

Run:

```bash
python scripts/evaluate.py --dataset eval.jsonl --api-url http://localhost:8000/ask
```

This computes an approximate cosine-style lexical relevance score for quick internal benchmarking.

## Tests and CI

Run tests locally:

```bash
pytest -q
```

GitHub Actions CI config is in `.github/workflows/ci.yml`.
