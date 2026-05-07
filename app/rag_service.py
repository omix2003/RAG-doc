from pathlib import Path
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import get_settings
from app.memory_store import SQLiteMemoryStore
from app.schemas import Message
from app.vectorstore import ensure_index_exists


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        ensure_index_exists()

        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embed_model,
            api_key=self.settings.openai_api_key,
        )
        self.vectorstore = PineconeVectorStore(
            index_name=self.settings.pinecone_index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.settings.pinecone_api_key,
            namespace=self.settings.pinecone_namespace,
        )
        self.llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=0,
            api_key=self.settings.openai_api_key,
            streaming=True,
        )
        self.memory_store = SQLiteMemoryStore(self.settings.memory_db_path)
        self.supported_extensions = {".txt", ".md", ".pdf", ".docx"}

    def _load_documents(self, source_dir: str, glob_pattern: str) -> List[Document]:
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Source directory not found: {source_dir}")

        documents: List[Document] = []
        for file_path in source_path.glob(glob_pattern):
            if not file_path.is_file():
                continue
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                continue

            if extension in {".txt", ".md"}:
                loader = TextLoader(str(file_path), autodetect_encoding=True)
            elif extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = Docx2txtLoader(str(file_path))

            documents.extend(loader.load())

        return documents

    def ingest_directory(self, source_dir: str, glob_pattern: str) -> int:
        docs = self._load_documents(source_dir, glob_pattern)
        if not docs:
            raise ValueError(
                "No supported files found. Supported extensions: .txt, .md, .pdf, .docx"
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No text chunks were produced from the selected documents.")
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def _build_prompt(self, query: str, context_docs: List[Document], history: List[Message]) -> str:
        context = "\n\n".join(doc.page_content for doc in context_docs)
        history_text = "\n".join(f"{m.role}: {m.content}" for m in history[-8:])
        return (
            "You are an enterprise retrieval assistant. Answer only using the provided context.\n"
            "If context is insufficient, say so clearly.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def stream_answer(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        history: List[Message] | None = None,
    ) -> tuple[Iterable[str], List[str]]:
        effective_history = history if history else self.memory_store.get_history(session_id)
        docs = self.vectorstore.similarity_search(query, k=top_k or self.settings.top_k)
        source_names = list(
            dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs)
        )
        prompt = self._build_prompt(query, docs, effective_history)

        stream = self.llm.stream(prompt)

        def token_generator() -> Iterable[str]:
            answer_tokens: List[str] = []
            for chunk in stream:
                text = chunk.content or ""
                if text:
                    answer_tokens.append(text)
                    yield text
            final_answer = "".join(answer_tokens)
            self.memory_store.append_messages(
                session_id,
                [
                    Message(role="user", content=query),
                    Message(role="assistant", content=final_answer),
                ],
            )

        return token_generator(), source_names
