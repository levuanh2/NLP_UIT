"""Application exception hierarchy."""


class LegalRAGError(Exception):
    """Base exception for expected Legal RAG failures."""


class ConfigurationError(LegalRAGError):
    """Raised when application configuration is invalid."""


class UnsupportedDocumentError(LegalRAGError):
    """Raised when no parser supports an input context file."""


class DocumentParseError(LegalRAGError):
    """Raised when a context JSON file cannot be validated or parsed."""


class ModelNotLoadedError(LegalRAGError):
    """Raised when inference is requested before local model loading."""


class SubmissionValidationError(LegalRAGError):
    """Raised when an invalid submission cannot be written."""
