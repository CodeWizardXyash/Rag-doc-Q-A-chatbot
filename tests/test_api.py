"""Basic API tests. They use a fake vector store, so no API key, Chroma data or network is needed."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.config import Settings, get_settings
from app.main import app, get_vector_store


class FakeVectorStore:
    """Stands in for VectorStore so tests never touch ChromaDB or an LLM."""

    def __init__(self) -> None:
        self.chunks = []

    def count(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def document_exists(self, doc_id: str) -> bool:
        return any(c.doc_id == doc_id for c in self.chunks)

    def list_documents(self):
        return []


@pytest.fixture()
def client(tmp_path):
    settings = Settings(upload_dir=tmp_path / "uploads", max_upload_mb=1)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def blank_pdf_bytes() -> bytes:
    """A valid PDF with one blank page (no extractable text)."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 0


def test_upload_rejects_non_pdf_extension(client):
    response = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_upload_rejects_fake_pdf(client):
    response = client.post(
        "/upload", files={"file": ("fake.pdf", b"this is not a pdf", "application/pdf")}
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client):
    big = b"%PDF-" + b"0" * (1024 * 1024 + 1)  # limit is 1 MB in this fixture
    response = client.post("/upload", files={"file": ("big.pdf", big, "application/pdf")})
    assert response.status_code == 413


def test_upload_rejects_pdf_without_text(client):
    response = client.post(
        "/upload", files={"file": ("blank.pdf", blank_pdf_bytes(), "application/pdf")}
    )
    assert response.status_code == 422


def test_ask_validates_question_length(client):
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ask_unknown_document_returns_404(client):
    response = client.post("/ask", json={"question": "What is this about?", "doc_id": "nope"})
    assert response.status_code == 404
