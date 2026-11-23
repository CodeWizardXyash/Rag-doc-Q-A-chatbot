"""ChromaDB wrapper. Setup is done for you; embedding/upsert (TODO #2) and
similarity search (TODO #3) are yours."""

import logging

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings
from app.exceptions import LLMServiceError
from app.schemas import DocumentChunk, DocumentInfo, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store holding chunks of all uploaded documents."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        # embedding_function=None: WE compute embeddings (TODO #2), Chroma just stores them.
        # cosine distance: Chroma returns distance = 1 - cosine_similarity.
        self._collection: Collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    # ----------------------------------------------------------------------- #
    # Provided: embedding client plumbing (lazy, so tests don't need API keys)
    # ----------------------------------------------------------------------- #
    def _require_api_key(self) -> str:
        key = self.settings.active_api_key
        if not key:
            raise LLMServiceError(
                f"No API key configured for provider '{self.settings.llm_provider}'. "
                "Set it in your .env file."
            )
        return key

    def _get_embedding_client(self):
        """Return an SDK client for the configured provider.

        - gemini -> `google.genai.Client`
        - openai -> `openai.OpenAI`
        """
        api_key = self._require_api_key()
        if self.settings.llm_provider == "gemini":
            from google import genai

            return genai.Client(api_key=api_key)
        from openai import OpenAI

        return OpenAI(api_key=api_key)

    # ----------------------------------------------------------------------- #
    # YOUR TODO #2: embeddings + upsert
    # ----------------------------------------------------------------------- #
    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Turn texts into embedding vectors using the configured provider.

        Args:
            texts: Strings to embed (chunk texts, or a single user question).
            is_query: True when embedding a search query, False for documents.
                Gemini embeddings accept a task type that improves retrieval:
                "RETRIEVAL_QUERY" vs "RETRIEVAL_DOCUMENT".

        Returns:
            One vector per input text, in the same order.

        TODO (step by step):
            1. `client = self._get_embedding_client()`.
            2. Gemini: `client.models.embed_content(model=self.settings.gemini_embedding_model,
               contents=texts, config=types.EmbedContentConfig(task_type=...))`
               (`from google.genai import types`). Vectors live in
               `response.embeddings[i].values`.
               OpenAI: `client.embeddings.create(model=self.settings.openai_embedding_model,
               input=texts)`. Vectors live in `response.data[i].embedding`.
            3. BATCH the calls (e.g. 100 texts at a time); providers limit batch size
               and a 300-page PDF produces hundreds of chunks.
            4. Wrap provider errors and raise `LLMServiceError` with a friendly message
               (never leak the raw exception or the API key to the client).
            5. Sanity check: `len(vectors) == len(texts)`.
        """
        raise NotImplementedError("TODO #2: implement embed_texts in vector_store.py")

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed chunks and upsert them (with metadata) into ChromaDB.

        Returns:
            Number of chunks stored.

        TODO (step by step):
            1. Return 0 early if `chunks` is empty.
            2. `embeddings = self.embed_texts([c.text for c in chunks])`.
            3. Build parallel lists: ids, documents (texts), metadatas.
               Metadata MUST include: doc_id, filename, page_number, chunk_index
               (Chroma metadata values must be str/int/float/bool; no None, no lists).
            4. `self._collection.upsert(ids=..., embeddings=..., documents=..., metadatas=...)`.
               Use upsert (not add) so re-uploading the same document is idempotent.
            5. Upsert in batches of a few hundred if the list is large.
            6. Log how many chunks were stored and return the count.
        """
        raise NotImplementedError("TODO #2: implement add_chunks in vector_store.py")

    # ----------------------------------------------------------------------- #
    # YOUR TODO #3: semantic similarity search
    # ----------------------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the `top_k` chunks most semantically similar to `query`.

        Args:
            query: The user's question.
            top_k: How many chunks to return.
            doc_id: If given, only search inside that document.

        Returns:
            `RetrievedChunk`s sorted best-first. Empty list if nothing matches.

        TODO (step by step):
            1. Embed the query with `self.embed_texts([query], is_query=True)[0]`.
               It MUST use the same model as the documents, or the vectors are
               not comparable.
            2. If `doc_id` is set, build `where={"doc_id": doc_id}`.
            3. `result = self._collection.query(query_embeddings=[vec], n_results=top_k,
               where=where, include=["documents", "metadatas", "distances"])`.
               Note: results are nested one level deep (one list per query),
               so use `result["documents"][0]`, etc.
            4. Convert distance to a similarity score: `score = 1 - distance`
               (collection uses cosine distance).
            5. Build and return `RetrievedChunk` objects.
            6. OPTIONAL: drop chunks with a very low score (e.g. < 0.3) so the LLM
               isn't fed irrelevant text. Tune the threshold on your own PDFs.
            7. Guard: if the collection is empty, `query` may error. Check
               `self.count()` first and return [].
        """
        raise NotImplementedError("TODO #3: implement search in vector_store.py")

    # ----------------------------------------------------------------------- #
    # Provided: housekeeping
    # ----------------------------------------------------------------------- #
    def count(self) -> int:
        """Total chunks stored across all documents."""
        return self._collection.count()

    def list_documents(self) -> list[DocumentInfo]:
        """Distinct documents with their chunk counts."""
        data = self._collection.get(include=["metadatas"])
        docs: dict[str, DocumentInfo] = {}
        for meta in data["metadatas"] or []:
            doc_id = str(meta["doc_id"])
            if doc_id not in docs:
                docs[doc_id] = DocumentInfo(
                    doc_id=doc_id, filename=str(meta["filename"]), num_chunks=0
                )
            docs[doc_id].num_chunks += 1
        return sorted(docs.values(), key=lambda d: d.filename.lower())

    def document_exists(self, doc_id: str) -> bool:
        found = self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return bool(found["ids"])

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})
