"""Agent pipeline module with smolagents dependency checking."""

def _check_smolagents():
    """Check if smolagents is installed and raise helpful error if not."""
    try:
        import smolagents
    except ImportError:
        raise ImportError(
            "You must install the agent dependencies to use this pipeline. "
            "Run: pip install onprem[agent]"
        )

# Check on module import
_check_smolagents()