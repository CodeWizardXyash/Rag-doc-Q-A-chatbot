"""PDF text extraction (done for you) + chunking strategy (YOUR TODO #1)."""

import logging
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings
from app.exceptions import InvalidFileError, NoExtractableTextError
from app.schemas import DocumentChunk, PageText

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Provided: extraction
# --------------------------------------------------------------------------- #
def extract_text_by_page(pdf_bytes: bytes) -> list[PageText]:
    """Extract text from every page of a PDF, keeping 1-based page numbers.

    Pages with no extractable text are skipped (they would only add noise).

    Raises:
        InvalidFileError: the file is corrupt or password-protected.
        NoExtractableTextError: no page contains text (e.g. a scanned PDF).
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            # Many "encrypted" PDFs just use an empty user password.
            if reader.decrypt("") == 0:
                raise InvalidFileError("PDF is password-protected.")
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(page_number=index, text=text))
        total_pages = len(reader.pages)
    except InvalidFileError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        logger.warning("Could not parse PDF: %s", exc)
        raise InvalidFileError("Could not read the PDF; the file may be corrupt.") from exc

    if not pages:
        raise NoExtractableTextError(
            f"No extractable text found in {total_pages} page(s). "
            "Scanned PDFs need OCR, which is not supported yet."
        )
    return pages


def count_pages(pdf_bytes: bytes) -> int:
    """Total number of pages (including empty ones)."""
    return len(PdfReader(BytesIO(pdf_bytes)).pages)


# --------------------------------------------------------------------------- #
# YOUR TODO #1: chunking strategy
# --------------------------------------------------------------------------- #
def chunk_pages(
    pages: list[PageText],
    doc_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Split page text into overlapping chunks that keep their page number.

    Why overlap? A sentence that straddles a chunk boundary would otherwise be
    cut in half and neither chunk would retrieve well. Overlap repeats the tail
    of chunk N at the head of chunk N+1.

    Args:
        pages: Output of `extract_text_by_page`.
        doc_id: Unique id of the document (used to build chunk ids).
        filename: Original filename (stored as metadata for citations).
        chunk_size: Target max characters per chunk (e.g. 1000).
        chunk_overlap: Characters shared between consecutive chunks (e.g. 200).

    Returns:
        A list of `DocumentChunk`. Each chunk MUST carry the page_number it came
        from, because the final answer has to cite pages.

    TODO (step by step):
        1. Validate: `chunk_overlap < chunk_size`, otherwise you loop forever.
        2. Loop over `pages`. Chunk each page separately so a chunk never spans
           two pages (this keeps page citations exact). Trade-off to think about:
           a sentence that continues onto the next page gets split. Is that
           acceptable for this project? Write down your decision in a comment.
        3. Inside a page, slide a window: start = 0, then
           start += chunk_size - chunk_overlap until start >= len(text).
        4. BONUS (better quality): don't cut mid-word/sentence. Try to end each
           window at the last ". " or "\n" or " " inside the window
           (`text.rfind(...)`). Fall back to a hard cut if none is found.
        5. Strip whitespace and skip empty chunks.
        6. Build a deterministic id, e.g. f"{doc_id}-p{page_number}-c{chunk_index}".
           Deterministic ids make re-uploading idempotent when you use `upsert`.
        7. Return the list. Log how many chunks you produced (`logger.info`).

    Test it alone first (no API key needed), e.g. in a Python shell:
        chunks = chunk_pages([PageText(page_number=1, text="word " * 500)],
                             "doc1", "a.pdf", 200, 50)
        assert all(len(c.text) <= 200 for c in chunks)
    """
    raise NotImplementedError("TODO #1: implement chunk_pages in pdf_processor.py")


# --------------------------------------------------------------------------- #
# Provided: orchestration
# --------------------------------------------------------------------------- #
def process_pdf(
    pdf_bytes: bytes,
    doc_id: str,
    filename: str,
    settings: Settings,
) -> tuple[int, list[DocumentChunk]]:
    """Extract + chunk a PDF. Returns (total_pages, chunks)."""
    total_pages = count_pages(pdf_bytes)
    pages = extract_text_by_page(pdf_bytes)
    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        filename=filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return total_pages, chunks
