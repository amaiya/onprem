"""
OnPrem Workflow Engine

This package provides a YAML-based workflow execution engine for document processing pipelines.
It allows users to define automated document processing workflows that connect different types 
of nodes (Loader, TextSplitter, Storage, Processor, Exporter) in a pipeline.
"""

# Import base classes and exceptions
from .base import (
    BaseNode, LoaderNode, TextSplitterNode, StorageNode, QueryNode, 
    ProcessorNode, DocumentProcessor, ResultProcessor, AggregatorProcessor, DocumentTransformerNode, ExporterNode,
    NODE_TYPES
)
from .exceptions import WorkflowValidationError, NodeExecutionError

# Import main engine and convenience functions
from .engine import WorkflowEngine
from .convenience import load_workflow, execute_workflow

# Import node registry
from .registry import NODE_REGISTRY

# Import CLI
from . import cli

__all__ = [
    # Base classes
    "BaseNode", "LoaderNode", "TextSplitterNode", "StorageNode", "QueryNode",
    "ProcessorNode", "DocumentProcessor", "ResultProcessor", "AggregatorProcessor", "DocumentTransformerNode", "ExporterNode",
    
    # Type system
    "NODE_TYPES",
    
    # Exceptions
    "WorkflowValidationError", "NodeExecutionError",
    
    # Main engine
    "WorkflowEngine",
    
    # Convenience functions
    "load_workflow", "execute_workflow",
    
    # Registry
    "NODE_REGISTRY",
    
    # CLI module
    "cli",
]


# CLI forwarding for backward compatibility
if __name__ == "__main__":
    import sys
    sys.exit(cli.main())