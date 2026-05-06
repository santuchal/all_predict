"""Custom exceptions used by all_predict."""


class AllPredictError(Exception):
    """Base exception for package-specific failures."""


class DataValidationError(AllPredictError):
    """Raised when user input is invalid or incomplete."""


class InvalidTaskError(AllPredictError):
    """Raised when the task cannot be inferred or is unsupported."""
