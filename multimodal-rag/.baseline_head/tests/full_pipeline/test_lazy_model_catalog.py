"""Regression test for the v3.3.0 G2 crashloop (PEP 562 lazy catalog).

The production image deliberately excludes pcai_models.py (.dockerignore:
it hardcodes PCAI-internal credentials).  utils/__init__.py used to have a
name-blind __getattr__ that tried to import the catalog for ANY attribute
miss - so "from multimodal_rag.utils import bm25" (a submodule import that
consults __getattr__ before the submodule attribute exists) crashed every
container at startup.

This file simulates the image condition by blocking the catalog module via a
meta-path finder, then asserts:

  * the full startup import chain (dataset_manager / rag_system / api_server)
    succeeds,
  * accessing a catalog slug raises a clear, actionable AttributeError,
  * an unknown attribute raises a plain AttributeError (module-like behaviour),
  * the bm25 lane stays importable and functional.

Run:

    python tests/full_pipeline/test_lazy_model_catalog.py    # standalone
    pytest tests/full_pipeline/test_lazy_model_catalog.py    # under pytest
"""

import importlib.abc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

CATALOG = "multimodal_rag.utils.pcai_models"


class _BlockCatalog(importlib.abc.MetaPathFinder):
    """Make the catalog module unimportable, like the production image."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == CATALOG:
            raise ImportError("pcai_models.py is docker-ignored (simulated)")


def _with_blocked_catalog(fn):
    blocker = _BlockCatalog()
    saved_meta = sys.meta_path[:]
    saved = sys.modules.pop(CATALOG, None)
    utils = sys.modules.get("multimodal_rag.utils")
    saved_attr = {}
    if utils is not None:
        # RESET (not pop) the module globals - __getattr__ reads them via
        # bare global lookups, so the keys must always exist.
        for key in ("__pcai_models", "__pcai_models_missing"):
            saved_attr[key] = utils.__dict__.get(key)
            utils.__dict__[key] = False if key.endswith("missing") else None
    sys.meta_path.insert(0, blocker)
    try:
        return fn()
    finally:
        sys.meta_path[:] = saved_meta
        if saved is not None:
            sys.modules[CATALOG] = saved
        if utils is not None:
            utils.__dict__.update(saved_attr)


def test_startup_imports_without_catalog():
    def run():
        import multimodal_rag.api_server
        import multimodal_rag.dataset_manager
        import multimodal_rag.rag_system  # noqa: F401

    _with_blocked_catalog(run)


def test_catalog_slug_raises_actionable_error():
    def run():
        import multimodal_rag.utils as u

        try:
            u.qwen3_vl_8B  # noqa: B018 - deliberate attribute-access probe
            raise AssertionError("expected AttributeError")
        except AttributeError as e:
            assert "pcai_models" in str(e) and "MODEL_" in str(e), str(e)

    _with_blocked_catalog(run)


def test_unknown_attribute_is_plain_attributeerror():
    def run():
        import multimodal_rag.utils as u

        try:
            u.definitely_not_a_thing  # noqa: B018
            raise AssertionError("expected AttributeError")
        except AttributeError as e:
            assert "no attribute" in str(e)

    _with_blocked_catalog(run)


def test_bm25_lane_importable_and_functional_without_catalog():
    def run():
        import multimodal_rag.utils.bm25 as bm25_lane

        tf = bm25_lane.term_counts("error code E1234 in module foo")
        weights = bm25_lane.bm25_query_weights(tf, bm25_lane._new_stats())
        vec = bm25_lane.to_sparse_vector(weights)
        assert len(vec.indices) == len(vec.values) > 0

    _with_blocked_catalog(run)


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
    print()
    print("All tests passed!" if not failed else str(failed) + " test(s) failed")
    sys.exit(1 if failed else 0)
