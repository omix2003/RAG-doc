from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_dir: str = Field(..., description="Directory containing source documents.")
    glob_pattern: str = Field("**/*.*", description="Glob to select files.")


class IngestResponse(BaseModel):
    chunks_indexed: int
    namespace: str


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str
    session_id: str = "default-session"
    top_k: Optional[int] = None
    history: List[Message] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
