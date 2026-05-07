import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum

from app.errors import register_exception_handlers
from app.rag_service import RAGService
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    UploadResponse,
)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), _: str = Depends(require_api_key)
) -> UploadResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    uploads_dir = Path("docs") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / file.filename
    content = await file.read()
    target_path.write_bytes(content)

    chunks = rag_service.ingest_file(str(target_path))
    source_id = rag_service.source_id_for_path(target_path)
    return UploadResponse(
        filename=file.filename,
        source_id=source_id,
        chunks_indexed=chunks,
        namespace=rag_service.settings.pinecone_namespace,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(_: str = Depends(require_api_key)) -> DocumentListResponse:
    base_dir = Path("docs")
    if not base_dir.exists():
        return DocumentListResponse(documents=[])

    supported = {".txt", ".md", ".pdf", ".docx"}
    docs = [
        path.as_posix()
        for path in sorted(base_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in supported
    ]
    return DocumentListResponse(documents=docs)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, _: str = Depends(require_api_key)) -> AskResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    answer_text, sources = rag_service.answer(
        query=request.query,
        session_id=request.session_id,
        top_k=request.top_k,
        source_filters=request.source_filters,
        history=request.history,
    )
    return AskResponse(answer=answer_text, sources=sources)


@app.post("/ask/stream")
def ask_stream(request: AskRequest, _: str = Depends(require_api_key)) -> StreamingResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    token_stream, sources = rag_service.stream_answer(
        query=request.query,
        session_id=request.session_id,
        top_k=request.top_k,
        source_filters=request.source_filters,
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
