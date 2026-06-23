"""Agent pipeline using PatchPal for sandboxed autonomous execution."""

from .base import AgentExecutor

# Keep backward compatibility - Agent is now an alias for AgentExecutor
Agent = AgentExecutor

__all__ = ['Agent', 'AgentExecutor']
