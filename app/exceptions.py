"""Domain exceptions. `main.py` maps each of these to a clean HTTP response."""


class AppError(Exception):
    """Base class for expected, user-facing errors."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidFileError(AppError):
    """Uploaded file is not a usable PDF (wrong type, empty, corrupt, encrypted)."""

    status_code = 400


class UnsupportedMediaTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class NoExtractableTextError(AppError):
    """PDF opened fine but has no text layer (e.g. a scanned document)."""

    status_code = 422


class DocumentNotFoundError(AppError):
    status_code = 404


class LLMServiceError(AppError):
    """Upstream LLM / embedding provider failed or is misconfigured."""

    status_code = 502
