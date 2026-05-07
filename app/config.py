from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_api_key: str = "change-me"
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    memory_db_path: str = "data/memory.db"

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    pinecone_api_key: str
    pinecone_index_name: str = "rag-doc-qa"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
