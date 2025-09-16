"""
Custom exceptions for the workflow engine.
"""


class WorkflowValidationError(Exception):
    """Raised when workflow validation fails."""
    pass


class NodeExecutionError(Exception):
    """Raised when node execution fails."""
    pass