"""Custom exceptions for the FIRM benchmark scaffold."""


class PlaceholderTaskError(NotImplementedError):
    """Raised when a task family is registered but its scene is not implemented."""
