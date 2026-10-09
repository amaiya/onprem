"""Lazy-loading package for OnPrem pipelines.

Pipelines are imported on first access (PEP 562) rather than eagerly, so that
importing a lightweight pipeline (e.g., `Extractor`, `Summarizer`, `RAGPipeline`)
does not pull in the heavy, optional dependencies required by other pipelines
(e.g., `torch` for the classifiers via `[local]`, or `patchpal` for the agent
via `[agent]`). Those dependencies are only required when the corresponding
pipeline is actually imported/used.
"""

# Map each public attribute to the submodule that provides it.
_LAZY_IMPORTS = {
    "Extractor": "onprem.pipelines.extractor.base",
    "Summarizer": "onprem.pipelines.summarizer",
    "FewShotClassifier": "onprem.pipelines.classifier",
    "SKClassifier": "onprem.pipelines.classifier",
    "HFClassifier": "onprem.pipelines.classifier",
    "AgentExecutor": "onprem.pipelines.agent.base",
    "RAGPipeline": "onprem.pipelines.rag",
    "KVRouter": "onprem.pipelines.rag",
    "CategorySelection": "onprem.pipelines.rag",
    "Guider": "onprem.pipelines.guider",
}

# Map each pipeline to the optional extra required to use it (if any), so that a
# missing dependency surfaces an actionable "pip install onprem[...]" message
# instead of a bare ModuleNotFoundError for a transitive dependency.
_PIPELINE_EXTRAS = {
    "FewShotClassifier": "local",
    "SKClassifier": "sklearn",
    "HFClassifier": "local",
    "AgentExecutor": "agent",
    "Guider": "guidance",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    """Import and return a pipeline attribute on first access (PEP 562)."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        extra = _PIPELINE_EXTRAS.get(name)
        if extra is not None:
            raise ImportError(
                f"Using `{name}` requires extra dependencies. "
                f"Install them with: pip install onprem[{extra}]"
            ) from e
        raise
    attr = getattr(module, name)
    # Cache on the package so subsequent lookups skip __getattr__.
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))
