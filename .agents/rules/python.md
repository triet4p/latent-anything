# Python Rules

**Always use [uv](https://docs.astral.sh/uv/) as the package manager.** Never use `pip`, `poetry`, or `conda`.

## Common uv commands

**Initialize a new sub-project:**
```bash
uv init <folder> -p 3.13
```

**Add or remove a package:**
```bash
uv add <package>
uv remove <package>
```

**Run a Python file** (prefer `-m` module style from the sub-project root):
```bash
uv run python -m src.main
# or, if module-style is not applicable:
uv run python src/main.py
```

## Standalone utility scripts

For scripts that are independent of any sub-project, do **not** use the sub-project's `uv` environment or `.venv`. Instead, use uv's inline script metadata ([PEP 723](https://peps.python.org/pep-0723/)):

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests<3",
#     "rich",
# ]
# ///

import requests
from rich import print

def main():
    response = requests.get("https://api.github.com")
    print(f"[bold green]Status Code:[/bold green] {response.status_code}")

if __name__ == "__main__":
    main()
```

## Code style

- **Linter / formatter:** `ruff` — run `ruff check` and `ruff format` before committing. Never leave lint errors unresolved.
- **Type checker:** `pyright` in strict mode. All public APIs must pass without errors.
- **Minimum Python version:** 3.12.

## Type annotations

- All function signatures (parameters + return type) must be fully annotated. No bare `Any` unless genuinely unavoidable — add a comment explaining why.
- Prefer `typing.Protocol` over abstract base classes for defining interfaces (lighter, structural, cross-language friendly).
- Use `collections.abc` types (`Sequence`, `Mapping`, `Callable`) in annotations, not their `typing` equivalents (deprecated since 3.9).
- For public APIs, tensor/array parameters must be typed as `numpy.ndarray`, not `torch.Tensor` — PyTorch must not leak into the public surface.

## Naming

Follow standard Python conventions:

| Kind | Convention | Example |
|---|---|---|
| Module / package | `snake_case` | `latent_space.py` |
| Class | `PascalCase` | `ModelAdapter` |
| Function / method / variable | `snake_case` | `encode_batch` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_CACHE_SIZE` |
| Private | leading `_` | `_compute_hash` |

## Testing

- Use `pytest` for all tests. Group tests by module under `tests/`, mirroring `src/` structure.
- Use `hypothesis` for property-based tests on core primitives (`LatentSpace`, `Trajectory`).
- Test file naming: `test_<module>.py`.
- Each test function name should read as a sentence: `test_trajectory_slice_returns_new_instance`.

## Approved packages

Use only the packages listed below for each concern. Do not introduce alternatives without discussion.

| Concern | Package(s) |
|---|---|
| Public API arrays | `numpy` — the only array type exposed in public signatures |
| Internal ML compute | `torch` — internal only, never leaked to callers |
| Config / validation | `pydantic` v2 — not Hydra, not dataclasses |
| Shape manipulation | `einops` |
| Async execution | `asyncio` (stdlib) |
| Disk cache | `diskcache` (default) or custom sqlite backend |
| Serialization | `pyarrow` — chosen for Rust interop readiness |
| Visualization | `plotly`, `matplotlib`, `anywidget` |
| Dimensionality reduction | `scikit-learn`, `umap-learn`, `pacmap` |
| Documentation | `mkdocs-material` |
| Cross-language plugins (Phase 7+) | `PyO3` + `maturin`, Arrow IPC or gRPC |

## Async

- The API is **async-primary.** Expose `async def` for all I/O-bound operations.
- Provide a `run_sync()` convenience wrapper for use in notebooks and scripts.
- Do not mix blocking calls inside `async def` functions — use `asyncio.to_thread` for CPU-bound or legacy-blocking code.
