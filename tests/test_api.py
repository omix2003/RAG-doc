import os

os.environ["SKIP_RAG_INIT"] = "1"
os.environ["APP_API_KEY"] = "test-key"

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


class FakeRAGService:
    class SettingsObj:
        pinecone_namespace = "default"

    def __init__(self) -> None:
        self.settings = self.SettingsObj()

    def ingest_directory(self, source_dir: str, glob_pattern: str) -> int:
        if source_dir == "bad":
            raise ValueError("Source directory not found: bad")
        return 3

    def stream_answer(self, query: str, session_id: str, top_k=None, history=None):
        def gen():
            yield "hello "
            yield "world"

        return gen(), ["docs/resume.txt"]


client = TestClient(app)


def setup_function() -> None:
    main_module.rag_service = FakeRAGService()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_required() -> None:
    response = client.post("/ask", json={"query": "hi", "session_id": "s1"})
    assert response.status_code == 401


def test_ingest_success() -> None:
    response = client.post(
        "/ingest",
        headers={"x-api-key": "test-key"},
        json={"source_dir": "./docs", "glob_pattern": "**/*.txt"},
    )
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 3


def test_ingest_validation_error() -> None:
    response = client.post(
        "/ingest",
        headers={"x-api-key": "test-key"},
        json={"source_dir": "bad", "glob_pattern": "**/*.txt"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_ask_success() -> None:
    response = client.post(
        "/ask",
        headers={"x-api-key": "test-key"},
        json={"query": "summarize", "session_id": "s1", "history": []},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "hello world"
    assert payload["sources"] == ["docs/resume.txt"]
