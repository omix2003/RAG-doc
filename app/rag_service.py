from pathlib import Path
import re
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

    def _normalize_source_id(self, file_path: Path) -> str:
        try:
            return file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return file_path.as_posix()

    def source_id_for_path(self, file_path: str | Path) -> str:
        return self._normalize_source_id(Path(file_path))

    def _build_retrieval_filter(self, source_filters: List[str]) -> dict | None:
        if not source_filters:
            return None
        normalized = [Path(src).as_posix() for src in source_filters]
        windows_style = [src.replace("/", "\\") for src in normalized]
        absolute_posix = [str((Path.cwd() / src).resolve()).replace("\\", "/") for src in normalized]
        absolute_windows = [path.replace("/", "\\") for path in absolute_posix]

        all_variants = list(
            dict.fromkeys(normalized + windows_style + absolute_posix + absolute_windows)
        )

        # Support both new metadata (`source_id`) and older ingested docs (`source`).
        return {
            "$or": [
                {"source_id": {"$in": all_variants}},
                {"source": {"$in": all_variants}},
            ]
        }

    def _retrieve_docs(
        self, query: str, top_k: int, source_filters: List[str] | None = None
    ) -> List[Document]:
        # MMR reduces near-duplicate chunks, which improves answer readability.
        filter_query = self._build_retrieval_filter(source_filters or [])
        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=max(top_k * 4, 12),
            lambda_mult=0.6,
            filter=filter_query,
        )

    def _dedupe_context(self, docs: List[Document]) -> str:
        seen: set[str] = set()
        unique_lines: List[str] = []
        for doc in docs:
            for raw_line in doc.page_content.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                key = re.sub(r"\s+", " ", line).lower()
                if key in seen:
                    continue
                seen.add(key)
                unique_lines.append(line)
        return "\n".join(unique_lines)

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

            loaded_docs = loader.load()
            source_id = self._normalize_source_id(file_path)
            for doc in loaded_docs:
                doc.metadata["source_id"] = source_id
            documents.extend(loaded_docs)

        return documents

    def _load_single_file(self, file_path: Path) -> List[Document]:
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(
                "Unsupported file type. Supported extensions: .txt, .md, .pdf, .docx"
            )

        if extension in {".txt", ".md"}:
            loader = TextLoader(str(file_path), autodetect_encoding=True)
        elif extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = Docx2txtLoader(str(file_path))

        docs = loader.load()
        source_id = self._normalize_source_id(file_path)
        for doc in docs:
            doc.metadata["source_id"] = source_id
        return docs

    def _index_documents(self, docs: List[Document]) -> int:
        if not docs:
            raise ValueError("No documents to index.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No text chunks were produced from the selected documents.")
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def ingest_directory(self, source_dir: str, glob_pattern: str) -> int:
        docs = self._load_documents(source_dir, glob_pattern)
        if not docs:
            raise ValueError(
                "No supported files found. Supported extensions: .txt, .md, .pdf, .docx"
            )
        return self._index_documents(docs)

    def ingest_file(self, file_path: str) -> int:
        docs = self._load_single_file(Path(file_path))
        return self._index_documents(docs)

    def _build_prompt(self, query: str, context_docs: List[Document], history: List[Message]) -> str:
        context = self._dedupe_context(context_docs)
        history_text = "\n".join(f"{m.role}: {m.content}" for m in history[-8:])
        return (
            "You are an enterprise retrieval assistant. Answer only using the provided context.\n"
            "If context is partial, provide a best-effort answer with clear assumptions.\n"
            "Only say context is insufficient when there is no usable candidate/job information at all.\n\n"
            "Formatting rules:\n"
            "- Return clean, readable output with no duplicated words or characters.\n"
            "- Use concise bullet points where appropriate.\n"
            "- Keep field labels simple and professional.\n\n"
            "For ATS/evaluation requests:\n"
            "- Provide a numeric score whenever candidate details exist in context.\n"
            "- If job description is missing, assume a generic Python Backend Fresher role and state that assumption.\n"
            "- Include: Overall Score, Strengths, Gaps, Missing Keywords, and 5 actionable improvements.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def _clean_answer_text(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n")
        cleaned = re.sub(r"\b([A-Za-z]{2,})\1\b", r"\1", cleaned)
        cleaned = re.sub(r"\b(\w+)([\s,.:;/-]+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\*{2,}", "", cleaned)
        cleaned = re.sub(r"\s+([,.:;])", r"\1", cleaned)
        # Promote inline bullets into separate lines when model returns one long sentence.
        cleaned = re.sub(r"\s+-\s+", "\n- ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        normalized_lines: List[str] = []
        for raw_line in cleaned.split("\n"):
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                continue

            if line.startswith(("-", "*")):
                line = "- " + line[1:].strip()

            words = line.split(" ")
            deduped_words: List[str] = []
            for word in words:
                if deduped_words and word.lower() == deduped_words[-1].lower():
                    continue
                deduped_words.append(word)
            normalized_lines.append(" ".join(deduped_words))

        cleaned = "\n".join(normalized_lines).strip()
        return cleaned

    def answer(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        source_filters: List[str] | None = None,
        history: List[Message] | None = None,
    ) -> tuple[str, List[str]]:
        effective_history = (
            self.memory_store.get_history(session_id) if history is None else history
        )
        docs = self._retrieve_docs(query, top_k or self.settings.top_k, source_filters)
        source_names = list(
            dict.fromkeys(
                doc.metadata.get("source_id") or doc.metadata.get("source", "unknown")
                for doc in docs
            )
        )
        prompt = self._build_prompt(query, docs, effective_history)
        result = self.llm.invoke(prompt)
        final_answer = self._clean_answer_text(result.content or "")
        self.memory_store.append_messages(
            session_id,
            [
                Message(role="user", content=query),
                Message(role="assistant", content=final_answer),
            ],
        )
        return final_answer, source_names

    def stream_answer(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        source_filters: List[str] | None = None,
        history: List[Message] | None = None,
    ) -> tuple[Iterable[str], List[str]]:
        effective_history = (
            self.memory_store.get_history(session_id) if history is None else history
        )
        docs = self._retrieve_docs(query, top_k or self.settings.top_k, source_filters)
        source_names = list(
            dict.fromkeys(
                doc.metadata.get("source_id") or doc.metadata.get("source", "unknown")
                for doc in docs
            )
        )
        prompt = self._build_prompt(query, docs, effective_history)
        result = self.llm.invoke(prompt)
        cleaned_answer = self._clean_answer_text(result.content or "")

        def token_generator() -> Iterable[str]:
            for i in range(0, len(cleaned_answer), 32):
                yield cleaned_answer[i : i + 32]
            self.memory_store.append_messages(
                session_id,
                [
                    Message(role="user", content=query),
                    Message(role="assistant", content=cleaned_answer),
                ],
            )

        return token_generator(), source_names
