"""Layer C runtime helpers.

Sprint 22 starts Layer C with a single concrete runtime primitive:
``BatchExecutor``. It is intentionally eager and synchronous. Cache,
async execution, streaming, and profiling remain future runtime
increments.
"""

from latent_anything.runtime.batch_executor import BatchExecutor as BatchExecutor

__all__ = ["BatchExecutor"]
