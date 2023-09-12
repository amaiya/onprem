__version__ = "0.0.17"

# reference: https://stackoverflow.com/questions/74918614/error-importing-seaborn-module-attributeerror/76760670#76760670
import numpy as np
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

from .core import LLM
