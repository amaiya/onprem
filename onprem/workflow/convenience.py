"""
Convenience functions for workflow execution.

This module provides simple functions to load and execute workflows without
directly interacting with the WorkflowEngine class.
"""

from typing import Dict, Any
from .engine import WorkflowEngine


def load_workflow(yaml_path: str) -> WorkflowEngine:
    """Convenience function to load a workflow from YAML file."""
    engine = WorkflowEngine()
    engine.load_workflow_from_yaml(yaml_path)
    return engine


def execute_workflow(yaml_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Convenience function to load and execute a workflow from YAML file."""
    engine = load_workflow(yaml_path)
    return engine.execute(verbose=verbose)