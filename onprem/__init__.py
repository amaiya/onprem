__version__ = "0.12.8"

# reference: https://stackoverflow.com/questions/74918614/error-importing-seaborn-module-attributeerror/76760670#76760670
import numpy as np
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

# Some older nvidia/cuda base images force this need on us.
# Try to appease chromadb expectation before it is loaded.
import sqlite3
if sqlite3.sqlite_version_info < (3, 35, 0):
    import importlib
    try:
        importlib.import_module("pysqlite3")
    except:
        raise ImportError(
            "Please install pysqlite3-binary: pip install pysqlite3-binary"
        )
    from sys import modules
    modules["sqlite3"] = modules.pop("pysqlite3")

from onprem.llm import LLM
