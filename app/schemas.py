from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_dir: str = Field(..., description="Directory containing source documents.")
    glob_pattern: str = Field("**/*.*", description="Glob to select files.")


class IngestResponse(BaseModel):
    chunks_indexed: int
    namespace: str


class UploadResponse(BaseModel):
    filename: str
    source_id: str
    chunks_indexed: int
    namespace: str


class DocumentListResponse(BaseModel):
    documents: List[str]


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str
    session_id: str = "default-session"
    top_k: Optional[int] = None
    source_filters: List[str] = Field(default_factory=list)
    history: List[Message] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
