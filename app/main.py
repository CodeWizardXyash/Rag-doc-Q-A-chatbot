"""FastAPI application: /health, /upload, /ask (+ document management)."""

import logging
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.exceptions import (
    AppError,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidFileError,
    UnsupportedMediaTypeError,
)
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    UploadResponse,
)
from app.services.pdf_processor import process_pdf
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Upload a PDF, then ask questions answered strictly from its content.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Dependencies (override these in tests with `app.dependency_overrides`)
# --------------------------------------------------------------------------- #
@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings())


def get_rag_engine(
    cfg: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> RAGEngine:
    return RAGEngine(cfg, store)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(NotImplementedError)
async def not_implemented_handler(_: Request, exc: NotImplementedError) -> JSONResponse:
    """While you are still writing the core RAG pieces, fail loudly but cleanly."""
    return JSONResponse(
        status_code=501,
        content={"detail": f"Core RAG step not implemented yet: {exc}"},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(
    cfg: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    return HealthResponse(
        app=cfg.app_name,
        environment=cfg.environment,
        llm_provider=cfg.llm_provider,
        indexed_chunks=store.count(),
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["documents"],
)
def upload_pdf(
    file: UploadFile = File(...),
    cfg: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> UploadResponse:
    """Validate a PDF, extract + chunk it, embed it and store it in ChromaDB."""
    filename = (file.filename or "document.pdf").strip()

    if not filename.lower().endswith(".pdf"):
        raise UnsupportedMediaTypeError("Only .pdf files are supported.")

    # Read at most limit+1 bytes so a huge upload can't exhaust memory.
    data = file.file.read(cfg.max_upload_bytes + 1)
    if len(data) > cfg.max_upload_bytes:
        raise FileTooLargeError(f"File exceeds the {cfg.max_upload_mb} MB limit.")
    if not data:
        raise InvalidFileError("The uploaded file is empty.")
    if not data.startswith(b"%PDF-"):
        raise InvalidFileError("The file does not look like a valid PDF.")

    doc_id = uuid.uuid4().hex
    total_pages, chunks = process_pdf(data, doc_id, filename, cfg)  # may raise TODO #1
    stored = store.add_chunks(chunks)  # may raise TODO #2

    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    (cfg.upload_dir / f"{doc_id}.pdf").write_bytes(data)

    logger.info("Indexed %s as %s (%d pages, %d chunks)", filename, doc_id, total_pages, stored)
    return UploadResponse(
        doc_id=doc_id, filename=filename, num_pages=total_pages, num_chunks=stored
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    tags=["chat"],
)
def ask(
    body: AskRequest,
    engine: RAGEngine = Depends(get_rag_engine),
    store: VectorStore = Depends(get_vector_store),
) -> AskResponse:
    """Answer a question strictly from the uploaded document(s), with page citations."""
    if body.doc_id and not store.document_exists(body.doc_id):
        raise DocumentNotFoundError(f"Document '{body.doc_id}' was not found.")
    return engine.answer(body.question, doc_id=body.doc_id, top_k=body.top_k)


@app.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(store: VectorStore = Depends(get_vector_store)) -> DocumentListResponse:
    return DocumentListResponse(documents=store.list_documents())


@app.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
def delete_document(
    doc_id: str,
    cfg: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> None:
    if not store.document_exists(doc_id):
        raise DocumentNotFoundError(f"Document '{doc_id}' was not found.")
    store.delete_document(doc_id)
    (cfg.upload_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
