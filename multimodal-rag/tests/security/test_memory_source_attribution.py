"""Writer-identity attribution for add_memory (2026-10-17).

THE BUG THIS PINS
-----------------
add_memory built its document base as {"source": "opencode:memory"} and then
carried the caller's metadata through with a "not in base" guard.  Since
source was ALWAYS in base, the caller's own metadata["source"] was silently
dropped on every write.  Result: every memory in every store recorded
"opencode:memory" — opencode's memories correctly, and the gateway
distiller's ("gateway:distiller") and DSH's ("dsh:memory") as a lie.

The label was therefore not a writer identity but a constant, which defeats
exactly what a provenance field is for: telling writers apart, filtering by
writer, auditing who wrote what.

THE CONTRACT AFTER THE FIX
--------------------------
* the caller's metadata["source"] is preserved verbatim;
* a write that declares nothing is stored as "unknown:memory" — never as
  some other client's name;
* surrounding whitespace is stripped and an empty/blank source is treated as
  absent (a blank label is not an attribution);
* source is not ALSO carried through as a stray metadata key.

These tests drive the real add_memory tool body with a DatasetManager
double, so they cover the actual argument plumbing rather than a helper.

Run::

    pytest tests/security/test_memory_source_attribution.py -q
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp


class _CapturingDM:
    """DatasetManager double: records the documents handed to add_documents."""

    def __init__(self) -> None:
        self.docs: list[dict] = []

    def get_dataset(self, name: str, sync_count: bool = True) -> dict:
        return {"name": name, "document_count": len(self.docs)}

    def has_password(self, name: str) -> bool:
        return False

    def add_documents(self, name: str, docs: list) -> list:
        self.docs.extend(docs)
        return [f"id-{i}" for i in range(len(docs))]

    def delete_session_memories(self, *a, **k) -> int:  # pragma: no cover - unused here
        return 0


@pytest.fixture()
def captured(monkeypatch):
    """Install the double and neutralise the ACL/password gates."""
    dm = _CapturingDM()
    monkeypatch.setattr(mcp, "_dm", dm, raising=False)
    monkeypatch.setattr(mcp, "get_manager", lambda: dm, raising=False)
    monkeypatch.setattr(mcp, "_require_dataset_acl", lambda name: None, raising=False)
    monkeypatch.setattr(mcp, "_require_dataset_write", lambda name: None, raising=False)
    monkeypatch.setattr(mcp, "_resolve_and_unlock", lambda dm_, name, pw: None, raising=False)
    monkeypatch.setenv("MEMORY_DATASET", "test-memory")
    return dm


def _write(metadata):
    asyncio.run(mcp.add_memory(text="a durable fact", metadata=metadata, dataset_name="test-memory"))


def test_gateway_distiller_source_is_preserved(captured):
    """The exact regression: the distiller's label must survive the write."""
    _write({"kind": "project-summary", "source": "gateway:distiller", "session_id": "s1"})
    assert captured.docs, "the write reached the store"
    assert captured.docs[0]["source"] == "gateway:distiller"


def test_dsh_source_is_preserved(captured):
    _write({"kind": "session_history", "source": "dsh:memory"})
    assert captured.docs[0]["source"] == "dsh:memory"


def test_opencode_source_is_preserved(captured):
    _write({"kind": "session_history", "source": "opencode:memory"})
    assert captured.docs[0]["source"] == "opencode:memory"


def test_absent_source_is_labelled_unattributed(captured):
    """A writer that names nobody must not inherit another client's name."""
    _write({"kind": "note"})
    assert captured.docs[0]["source"] == "unknown:memory"
    assert captured.docs[0]["source"] != "opencode:memory"


def test_no_metadata_at_all_is_unattributed(captured):
    _write(None)
    assert captured.docs[0]["source"] == "unknown:memory"


@pytest.mark.parametrize("blank", ["", "   ", "\t", None])
def test_blank_source_is_treated_as_absent(captured, blank):
    """A blank label is not an attribution — fall back, never store ""."""
    _write({"source": blank})
    assert captured.docs[0]["source"] == "unknown:memory"


def test_source_whitespace_is_stripped(captured):
    _write({"source": "  gateway:distiller  "})
    assert captured.docs[0]["source"] == "gateway:distiller"


def test_source_not_duplicated_as_stray_metadata(captured):
    """`source` is consumed as the writer identity — it must not ALSO ride
    along as an arbitrary metadata key."""
    _write({"source": "dsh:memory"})
    doc = captured.docs[0]
    assert doc["source"] == "dsh:memory"
    # It appears once: as the canonical top-level field, not as a second key.
    assert sum(1 for k in doc if k == "source") == 1


def test_other_metadata_still_carried_through(captured):
    """The fix must not narrow the existing carry-through behaviour."""
    _write({"kind": "decision", "tags": ["auth"], "session_id": "s9", "git_branch": "main"})
    doc = captured.docs[0]
    assert doc["memory_kind"] == "decision"
    assert doc["memory_tags"] == ["auth"]
    assert doc["session_id"] == "s9"
    assert doc["git_branch"] == "main"


def test_every_chunk_carries_the_source(captured, monkeypatch):
    """Long memories split into chunks; attribution must survive the split.

    The splitter packs whole sections at the session-history heading
    vocabulary, so the fixture body uses those headings.
    """
    monkeypatch.setenv("MEMORY_MAX_TOKENS", "24")
    body = "\n".join(
        f"### User\nquestion number {i} with enough words to consume budget\n"
        f"### Assistant\nanswer number {i} which is also reasonably long\n"
        for i in range(8)
    )
    asyncio.run(
        mcp.add_memory(
            text=body,
            metadata={"kind": "session_history", "source": "gateway:distiller", "session_id": "s2"},
            dataset_name="test-memory",
        )
    )
    assert len(captured.docs) > 1, "the memory actually split"
    assert {d["source"] for d in captured.docs} == {"gateway:distiller"}
