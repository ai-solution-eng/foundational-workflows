"""Tests that ingest-time skip/caption warnings actually reach the UI path.

The API records ingest warnings (e.g. "Omitting audio document: no ASR
model…") into a per-request list via ``rag_system._ingest_warnings``.  The
collector is a contextvar, and the API hands the ingest work to a
``run_in_executor`` thread pool — which does NOT propagate contextvars by
default, so without a fix the warnings silently vanish before the UI can
return them.

These tests assert that ``api_server._submit_with_context`` (the wrapper used
at every ingest call site) keeps the collector visible across the
executor → background-event-loop → worker chain.

Run::

    python tests/full_pipeline/test_ingest_warnings.py     # standalone
    pytest tests/full_pipeline/test_ingest_warnings.py     # under pytest
"""

import asyncio
import os
import sys

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.api_server import _submit_with_context
from multimodal_rag.rag_system import _ingest_warnings, _record_ingest_warning
from multimodal_rag.utils.general_tools import sync_wrapper_safe


def _record_via_rag_path() -> None:
    """Replicate a warning raised deep inside the ingest pipeline: it runs in
    the background event loop (via sync_wrapper_safe) from a worker thread."""

    async def _inner() -> None:
        _record_ingest_warning("SKIP-CAPTION: no ASR/embedder support")

    sync_wrapper_safe(_inner, {})


def test_submit_with_context_propagates_ingest_warnings() -> None:
    warnings: list[str] = []

    async def main() -> list[str]:
        token = _ingest_warnings.set(warnings)
        await _submit_with_context(None, _record_via_rag_path)
        _ingest_warnings.reset(token)
        return warnings

    captured = asyncio.run(main())
    assert captured == ["SKIP-CAPTION: no ASR/embedder support"], (
        "ingest warnings must reach the request's list through _submit_with_context"
    )


def test_submit_with_context_returns_callable_value() -> None:
    async def main() -> int:
        result = await _submit_with_context(None, lambda: 42)
        return result

    assert asyncio.run(main()) == 42


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__} ... OK")
        except Exception:
            failed += 1
            print(f"  {fn.__name__} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
