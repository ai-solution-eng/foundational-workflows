"""Regression tests: RAG_UNLOCK_MAX_TTL=0 must not reject positive TTLs.

Live defect (G2, found 2026-10-14): with RAG_UNLOCK_MAX_TTL=0 the MCP
unlock_dataset tool rejected EVERY positive TTL with the nonsense message
"TTL must be between 60 and 0 seconds", leaving ttl=0 as the only working
unlock.  The REST endpoint was already correct (api_server.py guards the
comparison with "ttl_max and ..."); the MCP mirror was not.

ttl_max == 0 is the deployment's "no-expiry ALLOWED" opt-in sentinel
(_unlock_ttl_max returns 0 to mean "ttl=0 is permitted"), NOT a zero-second
cap.  These tests pin that reading for both surfaces, so the two
implementations cannot drift apart again.

Run:

    pytest tests/security/test_unlock_ttl_max.py -q
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("RAG_UNLOCK_MAX_TTL", raising=False)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 86400),        # unset -> bounded default
        ("", 86400),          # blank -> default
        ("0", 0),             # explicit opt-in: no-expiry ALLOWED
        ("3600", 3600),       # explicit bound
        ("abc", 86400),       # garbage -> default (never a crash)
        ("-5", 86400),        # negative -> default
    ],
)
def test_unlock_ttl_max_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("RAG_UNLOCK_MAX_TTL", raising=False)
    else:
        monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", raw)
    assert mcp._unlock_ttl_max() == expected
    assert api._unlock_ttl_max() == expected


def test_mcp_and_rest_agree_on_zero_sentinel(monkeypatch):
    """Both surfaces must read ttl_max==0 the same way (this is the bug)."""
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "0")
    assert mcp._unlock_ttl_max() == 0
    assert api._unlock_ttl_max() == 0


def test_positive_ttl_is_valid_when_max_is_zero(monkeypatch):
    """The predicate must ACCEPT a positive TTL under the 0 sentinel.

    Mirrors both implementations exactly; if this fails, the deployment-wide
    "only ttl=0 can unlock" regression is back.
    """
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "0")
    ttl_max = mcp._unlock_ttl_max()

    def rejects(ttl: int) -> bool:
        return ttl < 60 or bool(ttl_max and ttl > ttl_max)

    for ttl in (60, 1800, 86400, 10**6):
        assert rejects(ttl) is False, "valid ttl=" + str(ttl) + " rejected under the 0 sentinel"

    assert ttl_max == 0  # ttl=0 allowed only because of this

    for ttl in (1, 59):
        assert rejects(ttl) is True


def test_bounded_deployment_still_caps(monkeypatch):
    """With a real bound, an over-max TTL is still refused."""
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "3600")
    ttl_max = mcp._unlock_ttl_max()
    assert ttl_max == 3600

    def rejects(ttl: int) -> bool:
        return ttl < 60 or bool(ttl_max and ttl > ttl_max)

    assert rejects(3600) is False   # exactly at the cap
    assert rejects(3601) is True    # over the cap
    assert rejects(1800) is False


def test_source_uses_the_guarded_comparison():
    """Guard the shipped expression, not just the predicate.

    The bug was a missing guard on one line; predicate tests cannot see which
    expression the shipped code uses.
    """
    src = inspect.getsource(mcp.unlock_dataset)
    assert "ttl_max and ttl > ttl_max" in src, (
        "mcp_server.unlock_dataset must guard the cap with 'ttl_max and ...' "
        "(ttl_max==0 is the no-expiry sentinel, not a zero-second cap)"
    )
    assert "ttl < 60 or ttl > ttl_max" not in src, "unguarded cap comparison is back"
