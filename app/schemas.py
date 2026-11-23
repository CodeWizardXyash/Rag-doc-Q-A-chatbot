"""Pydantic models: API request/response shapes and internal data containers."""

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Internal containers (used between services)
# --------------------------------------------------------------------------- #
class PageText(BaseModel):
    """Raw text extracted from one PDF page."""

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str


class DocumentChunk(BaseModel):
    """A slice of a document, ready to be embedded and stored."""

    chunk_id: str = Field(..., description="Unique, deterministic id, e.g. '<doc_id>-p3-c0'")
    doc_id: str
    filename: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    text: str


class RetrievedChunk(BaseModel):
    """A chunk returned by semantic search, with its similarity information."""

    text: str
    doc_id: str
    filename: str
    page_number: int
    score: float = Field(..., description="Similarity score; higher = more similar")


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    environment: str
    llm_provider: str
    indexed_chunks: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    message: str = "Document processed successfully"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_id: str | None = Field(
        default=None,
        description="Restrict the search to one uploaded document. None = search all.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    """One piece of evidence shown to the user under an answer."""

    filename: str
    page_number: int
    snippet: str = Field(..., description="Short excerpt of the supporting chunk")
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str


class ErrorResponse(BaseModel):
    detail: str
