import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mangum import Mangum

from app.errors import register_exception_handlers
from app.rag_service import RAGService
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.security import require_api_key

rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global rag_service
    if os.getenv("SKIP_RAG_INIT") != "1":
        rag_service = RAGService()
    yield


app = FastAPI(
    title="RAG Document Q&A API",
    version="1.0.0",
    lifespan=lifespan,
)
register_exception_handlers(app)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, _: str = Depends(require_api_key)) -> IngestResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    count = rag_service.ingest_directory(request.source_dir, request.glob_pattern)
    return IngestResponse(chunks_indexed=count, namespace=rag_service.settings.pinecone_namespace)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, _: str = Depends(require_api_key)) -> AskResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    stream, sources = rag_service.stream_answer(
        query=request.query,
        session_id=request.session_id,
        top_k=request.top_k,
        history=request.history,
    )
    answer = "".join(list(stream))
    return AskResponse(answer=answer, sources=sources)


@app.post("/ask/stream")
def ask_stream(request: AskRequest, _: str = Depends(require_api_key)) -> StreamingResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    token_stream, sources = rag_service.stream_answer(
        query=request.query,
        session_id=request.session_id,
        top_k=request.top_k,
        history=request.history,
    )

    async def event_stream() -> AsyncIterator[str]:
        for token in token_stream:
            payload = {"type": "token", "content": token}
            yield f"data: {json.dumps(payload)}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


handler = Mangum(app)
