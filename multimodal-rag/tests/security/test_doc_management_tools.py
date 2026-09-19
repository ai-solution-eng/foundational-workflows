"""Wave-5 F2: MCP document-management tools (previously REST-only).

``dataset_add_documents`` / ``dataset_delete_documents`` /
``dataset_replace_document`` are thin wrappers over the SAME DatasetManager
calls the REST endpoints use — the tests pin both the behaviour (guards,
validation, events) and the parity with the REST twins' result shapes:

* dataset-existence guard → the REST 404 message as a ToolError;
* the password gate via ``_check_unlocked_or_password`` (REST
  ``_require_dataset_password`` parity);
* the D15 dataset ACL (deny without confirming existence);
* local ingest paths validated against the ``MEDIA_ALLOW_PATH_PREFIXES``
  allowlist (same policy as the MCP media-read tools — an ingested
  ``/etc/passwd`` would be readable back via search);
* upload-history events recorded by the SAME helper the REST endpoints use;
* result shapes: texts → ``{"status","stored_ids","count"}`` (POST
  /documents twin); paths/URLs → the batch job terminal shape
  ``{"status","file_count","files"}``; delete → ``{"status","deleted","count"}``
  (the single-doc REST twin extended to the batch).

Everything runs OFFLINE: real DatasetManager + local-mode Qdrant + the
conftest stub embedder.

Run::

    pytest tests/security/test_doc_management_tools.py -q
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp
from multimodal_rag.dataset_manager import (
    DatasetManager,
    _load_upload_history,
)
from multimodal_rag.utils.media_paths import MediaRefError  # noqa: F401 — parity guard

qdrant_client = pytest.importorskip("qdrant_client")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_prefixes(tmp_path, monkeypatch):
    """Point the media-path allowlist at a temp tree (module-global attrs —
    the same pattern as test_security_kernel)."""
    datasets = tmp_path / "datasets"
    staging = tmp_path / "staging"
    datasets.mkdir()
    staging.mkdir()
    import multimodal_rag.utils.media_paths as mp

    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", (str(datasets), str(staging)))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)
    return tmp_path


@pytest.fixture
def dm(base_path, embedder, allow_prefixes, monkeypatch):
    """Real offline DatasetManager with one dataset, injected as the MCP
    server's shared manager. DATA_PATH is pointed at the temp root so the
    upload-history log (a DATA_PATH sidecar, like the REST endpoints write
    it) lands in the sandbox."""
    monkeypatch.setenv("DATA_PATH", base_path)
    manager = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    manager.create_dataset("docs")
    mcp._dm = manager
    yield manager
    mcp._dm = None


def _call(tool, **kwargs):
    """Invoke an MCP tool coroutine (the tools are thin async wrappers)."""
    return asyncio.run(tool(**kwargs))


def _payload(result: str) -> dict:
    return json.loads(result)


def _stage(allow_prefixes: Path, name: str, text: str) -> str:
    p = allow_prefixes / "staging" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


def _stored_ids(dm: DatasetManager, name: str) -> list[str]:
    return [doc_id for doc_id, _ in dm.list_documents(name, 1000)]


# ---------------------------------------------------------------------------
# dataset_add_documents
# ---------------------------------------------------------------------------


def test_add_documents_texts_matches_rest_twin_shape(dm):
    """texts → the POST /api/datasets/{name}/documents result shape."""
    out = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["alpha doc", "beta doc"]))
    assert out["status"] == "ok"
    assert out["documents"]["status"] == "ok"
    assert out["documents"]["count"] == 2
    assert len(out["documents"]["stored_ids"]) == 2
    # really stored in the dataset
    assert len(_stored_ids(dm, "docs")) == 2


def test_add_documents_local_path_within_allowlist(dm, allow_prefixes):
    staged = _stage(allow_prefixes, "export.txt", "line one for the staged export\nline two for the staged export")
    out = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", paths=[staged]))
    batch = out["files"]
    # the batch-files/batch-urls terminal (job-result) shape
    assert batch["status"] == "ok"
    assert batch["file_count"] == 1
    assert batch["files"][0]["file"] == "export.txt"
    assert batch["files"][0]["chunks"] >= 1
    assert len(_stored_ids(dm, "docs")) >= 1


def test_add_documents_url_goes_through_the_rest_batch_path(dm, monkeypatch):
    """http(s)/s3 paths route to add_urls_batch (the batch-urls twin)."""
    calls = {}

    def _fake_add_urls(dataset, urls, **kwargs):
        calls["urls"] = list(urls)
        return {"status": "ok", "file_count": 1, "files": [{"file": "x.txt", "chunks": 1}]}

    monkeypatch.setattr(dm, "add_urls_batch", _fake_add_urls)
    out = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", paths=["https://example.com/x.txt"]))
    assert calls["urls"] == ["https://example.com/x.txt"]
    assert out["files"]["file_count"] == 1


def test_add_documents_local_path_outside_allowlist_refused(dm, allow_prefixes):
    evil = str(allow_prefixes / "etc" / "passwd")
    with pytest.raises(mcp.ToolError, match="allowed ingest prefixes"):
        _call(mcp.dataset_add_documents, dataset_name="docs", paths=[evil])
    assert _stored_ids(dm, "docs") == []


def test_add_documents_missing_local_file_refused(dm, allow_prefixes):
    missing = str(allow_prefixes / "staging" / "nope.txt")
    with pytest.raises(mcp.ToolError, match="does not exist"):
        _call(mcp.dataset_add_documents, dataset_name="docs", paths=[missing])


def test_add_documents_missing_dataset_is_the_rest_404_parity(dm):
    with pytest.raises(mcp.ToolError, match="Dataset 'nope' not found."):
        _call(mcp.dataset_add_documents, dataset_name="nope", texts=["x"])


def test_add_documents_password_gate_parity(dm):
    dm.set_password("docs", "hunter2")
    with pytest.raises(mcp.ToolError, match="password protected"):
        _call(mcp.dataset_add_documents, dataset_name="docs", texts=["x"])
    out = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["x"], password="hunter2"))
    assert out["status"] == "ok"


def test_add_documents_requires_a_source(dm):
    with pytest.raises(mcp.ToolError, match="at least one"):
        _call(mcp.dataset_add_documents, dataset_name="docs")


def test_add_documents_records_upload_history_like_rest(dm, allow_prefixes):
    """Same events/audit: the SAME helper the REST batch endpoints call."""
    staged = _stage(allow_prefixes, "hist.txt", "history body")
    _call(mcp.dataset_add_documents, dataset_name="docs", paths=[staged])
    entries = [e for e in _load_upload_history() if e["dataset"] == "docs"]
    assert entries and entries[-1]["file"] == "hist.txt"
    assert entries[-1]["source"] == "files"
    assert entries[-1]["status"] == "ok"


# ---------------------------------------------------------------------------
# dataset_delete_documents
# ---------------------------------------------------------------------------


def test_delete_documents_by_ids(dm):
    ids = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["one", "two"]))["documents"][
        "stored_ids"
    ]
    out = _payload(_call(mcp.dataset_delete_documents, dataset_name="docs", doc_ids=[ids[0]]))
    assert out["status"] == "ok"
    assert out["deleted"] == [ids[0]]
    assert out["count"] == 1
    assert _stored_ids(dm, "docs") == [ids[1]]


def test_delete_documents_requires_exactly_one_selector(dm):
    with pytest.raises(mcp.ToolError, match="exactly one"):
        _call(mcp.dataset_delete_documents, dataset_name="docs")
    with pytest.raises(mcp.ToolError, match="exactly one"):
        _call(
            mcp.dataset_delete_documents,
            dataset_name="docs",
            doc_ids=["a"],
            filter={"source_prefix": "s3://b/"},
        )


def test_delete_documents_filter_source_prefix(dm):
    """filter={"source_prefix"} deletes exactly the matching sources — the
    S3-sync prune semantic, server-side MatchPrefix scroll. Seeded through
    the REAL ingest path (a raw dict doc's top-level ``source`` key becomes
    the stored ``metadata.source``, exactly as URL ingests record it)."""
    ids = dm.add_documents(
        "docs",
        [
            {"text": "keeper doc", "source": "s3://bucket/keep/a.txt"},
            {"text": "report one", "source": "s3://bucket/reports/1.pdf"},
            {"text": "report two", "source": "s3://bucket/reports/2.pdf"},
        ],
    )
    out = _payload(
        _call(
            mcp.dataset_delete_documents,
            dataset_name="docs",
            filter={"source_prefix": "s3://bucket/reports/"},
        )
    )
    assert out["status"] == "ok"
    assert sorted(out["deleted"]) == sorted(ids[1:])
    assert out["count"] == 2
    assert out["truncated"] is False
    assert _stored_ids(dm, "docs") == [ids[0]]


def test_delete_documents_filter_requires_known_keys(dm):
    with pytest.raises(mcp.ToolError, match="source_prefix"):
        _call(mcp.dataset_delete_documents, dataset_name="docs", filter={"evil": "x"})


# ---------------------------------------------------------------------------
# dataset_replace_document
# ---------------------------------------------------------------------------


def test_replace_document_roundtrip(dm, allow_prefixes):
    ids = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["old version"]))["documents"][
        "stored_ids"
    ]
    staged = _stage(allow_prefixes, "replacement.txt", "new version body")
    out = _payload(_call(mcp.dataset_replace_document, dataset_name="docs", doc_id=ids[0], path=staged))
    assert out["status"] == "ok"
    assert out["replaced"] == ids[0]
    assert out["deleted"] == ids[0]
    assert out["added"]["stored_ids"]
    after = _stored_ids(dm, "docs")
    assert ids[0] not in after
    assert set(out["added"]["stored_ids"]) <= set(after)


def test_replace_document_unknown_doc_refused_no_silent_add(dm, allow_prefixes):
    staged = _stage(allow_prefixes, "r.txt", "body")
    with pytest.raises(mcp.ToolError, match="not found in dataset"):
        _call(mcp.dataset_replace_document, dataset_name="docs", doc_id=uuid.uuid4().hex, path=staged)
    assert _stored_ids(dm, "docs") == []


def test_replace_document_ingest_failure_loses_nothing(dm, allow_prefixes, monkeypatch):
    """Add-first ordering: a failed ingest leaves the old document intact."""
    ids = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["precious"]))["documents"]["stored_ids"]

    def _boom(*a, **k):
        raise RuntimeError("embedder exploded")

    monkeypatch.setattr(dm, "add_file", _boom)
    staged = _stage(allow_prefixes, "boom.txt", "body")
    with pytest.raises(mcp.ToolError, match="embedder exploded"):
        _call(mcp.dataset_replace_document, dataset_name="docs", doc_id=ids[0], path=staged)
    assert _stored_ids(dm, "docs") == [ids[0]]


def test_replace_document_delete_failure_is_honest_partial(dm, allow_prefixes, monkeypatch):
    ids = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["v1"]))["documents"]["stored_ids"]
    staged = _stage(allow_prefixes, "v2.txt", "v2 body")

    def _boom(*a, **k):
        raise RuntimeError("qdrant delete failed")

    monkeypatch.setattr(dm, "delete_document", _boom)
    out = _payload(_call(mcp.dataset_replace_document, dataset_name="docs", doc_id=ids[0], path=staged))
    assert out["status"] == "partial"
    assert out["added"]["stored_ids"]
    assert out["deleted"] is None
    assert "delete failed" in out["error"]


# ---------------------------------------------------------------------------
# D15 ACL guards (deny-first, existence-oracle-free)
# ---------------------------------------------------------------------------


@pytest.fixture
def acl_client(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "alice:key-a;bob:key-b")
    monkeypatch.setenv("RAG_DATASET_ACLS", "alice:docs")
    yield
    monkeypatch.delenv("RAG_API_KEY_CLIENTS", raising=False)
    monkeypatch.delenv("RAG_DATASET_ACLS", raising=False)


def test_acl_denied_dataset_refused_for_every_doc_tool(dm, acl_client):
    """A key with no grant for 'docs' is denied by all three tools — without
    an existence oracle (the denial fires before the existence check)."""
    import multimodal_rag.utils.clients_registry as cr

    ident = cr.Identity(kind="client", name="bob", datasets=frozenset())
    token = cr.set_current_identity(ident)
    try:
        with pytest.raises(mcp.ToolError, match="not permitted for this API key"):
            _call(mcp.dataset_add_documents, dataset_name="docs", texts=["x"])
        with pytest.raises(mcp.ToolError, match="not permitted for this API key"):
            _call(mcp.dataset_delete_documents, dataset_name="docs", doc_ids=["a"])
        with pytest.raises(mcp.ToolError, match="not permitted for this API key"):
            _call(mcp.dataset_replace_document, dataset_name="docs", doc_id="a", path="/x")
        # denial does not confirm existence: an unknown dataset gets the SAME error
        with pytest.raises(mcp.ToolError, match="not permitted for this API key"):
            _call(mcp.dataset_add_documents, dataset_name="never-created", texts=["x"])
    finally:
        cr.reset_current_identity(token)


def test_acl_granted_key_can_manage(dm, acl_client):
    import multimodal_rag.utils.clients_registry as cr

    token = cr.set_current_identity(cr.Identity(kind="client", name="alice", datasets=frozenset({"docs"})))
    try:
        out = _payload(_call(mcp.dataset_add_documents, dataset_name="docs", texts=["allowed"]))
        assert out["status"] == "ok"
    finally:
        cr.reset_current_identity(token)
