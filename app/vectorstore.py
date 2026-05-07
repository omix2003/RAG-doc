from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings


def ensure_index_exists() -> None:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing_indexes = set(pc.list_indexes().names())
    if settings.pinecone_index_name in existing_indexes:
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
