#!/usr/bin/env python3
"""Offline unit tests for stmem/stmem.py -- the ST-MEM reaper core.

Task: task-1 (ST-MEM unit tests: envelope + reaper selection); task-6 updated
the four behavior-pinning tests to the FIXED stmem.py contract.

Everything here runs OFFLINE: stmem._req is replaced with an in-memory
stand-in and an autouse fixture turns any real urlopen into a hard failure,
so no socket is ever opened and no cluster is touched.

The module is imported by path (importlib.util.spec_from_file_location)
because stmem.py is a standalone stdlib-only CLI, not a package member.

Coverage (acceptance criteria in the task):
  1. _parse_envelope     -- valid / invalid JSON / no "stmm" key / non-dict
                            JSON / leading whitespace / no newline after the
                            envelope / empty string.
  2. _find_text          -- str payload / dict payload / no envelope / decoy
                            containing '"stmm"' (the real envelope must win).
  3. cmd_reap selection  -- expired / not-yet-expired / superseded /
                            unparseable-or-missing created_at (fail-closed) /
                            missing ttl_hours -> --ttl-hours fallback /
                            STMM_DRY_RUN=1.
  4. iter_documents      -- two pages then stop on an empty next_cursor; a
                            REPEATING cursor also stops (with a WARNING).
  5. cmd_ensure          -- 409/200 -> 0; 400/5xx/URLError -> 1, including the
                            "does not exist" and "existing" substring traps.
  6. _req transport      -- F8: BOTH X-API-Key and X-RAG-Api-Key are sent when
                            the key is set, and no auth header leaks when it is
                            not; a non-JSON 2xx body (N2) is a URLError.
  7. TTL range (N1)      -- 1e8 / 1e9 / inf / nan and a year-9999 created_at
                            never abort a sweep.

REVIEW FINDINGS (raised by task-1, all FIXED in stmem.py on 2026-10-11; this
suite now pins the FIXED contract, so every assertion below describes the
INTENDED behavior -- a regression turns the test red):
  F1 iter_documents (was stmem.py:72-81) looped forever on a repeating
     next_cursor. NOW a seen-cursor set stops pagination and writes a WARNING
     to stderr. Pinned by
     test_iter_documents_terminates_on_repeating_cursor.
  F2 cmd_reap (was stmem.py:145-147) let a TypeError from a malformed
     created_at / ttl_hours abort the WHOLE sweep. NOW every per-entry error
     is contained: the malformed entry is KEPT (fail-closed), healthy expired
     entries are still reaped, and the sweep reports them as
     skipped_unparseable. Pinned by
     test_reap_non_string_created_at_does_not_abort_the_sweep and
     test_reap_non_numeric_ttl_hours_does_not_abort_the_sweep.
  F3 The expiry parse accepted only %Y-%m-%dT%H:%M:%SZ. NOW zone-bearing
     ISO-8601 variants (offset, fractional seconds) parse and DO expire,
     while naive / date-only values stay KEPT (ambiguous -> fail-closed).
     Pinned by test_reap_expires_offset_and_fractional_created_at and
     test_reap_keeps_naive_created_at.
  F4 _find_text returned the first dict value merely CONTAINING '"stmm"', so a
     decoy could shadow the real envelope. NOW the value that actually PARSES
     as an envelope wins; the substring scan is only a fallback. Pinned by
     test_find_text_prefers_real_envelope_over_decoy.
  F8 _req sent only X-API-Key while the baseline middleware reads
     X-RAG-Api-Key / Authorization, so auth silently failed on one of them.
     NOW both are sent. PINNED HERE (task-8) by
     test_req_sends_both_api_key_headers / test_req_omits_auth_headers_when_unset
     -- reverting the fix turns them red.
  N1 An out-of-range ttl_hours (1e8+) passed the old type check, reached the
     timedelta arithmetic and raised an uncaught OverflowError that killed the
     whole sweep. NOW _ttl_hours range-checks (_MAX_TTL_HOURS) and the
     arithmetic is contained. PINNED HERE (task-8) by test_ttl_hours_range_check
     and test_reap_out_of_range_ttl_does_not_abort_the_sweep.
  N2 A non-JSON 2xx body raised an uncaught JSONDecodeError; it is now a
     URLError, and cmd_ensure's 400 handler no longer matches the broad
     "exist" substring (which also matched "does not exist"). PINNED HERE
     (task-8) by test_req_non_json_2xx_raises_urlerror and
     test_cmd_ensure_400_errors_are_failures.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

#: Newline without a backslash escape in this source file (keeps the test
#: corpus trivially greppable and avoids escape-in-escape confusion).
NL = chr(10)
TAB = chr(9)

STMEM_PATH = Path(__file__).resolve().parents[1] / "stmem.py"


def _load_stmem():
    """Import stmem.py by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("stmem_under_test", STMEM_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {STMEM_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stmem = _load_stmem()

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: Deterministic clock for every expiry test (cmd_reap calls datetime.now(utc)).
FROZEN_NOW = datetime.datetime(2026, 10, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)

_MISSING = object()


class _FrozenDatetime(datetime.datetime):
    """datetime whose now() is pinned, so expiry math is exact."""

    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW if tz is not None else FROZEN_NOW.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any real HTTP attempt is a test failure (defence in depth)."""

    def _boom(*args, **kwargs):
        raise AssertionError("test attempted real network I/O via urlopen")

    monkeypatch.setattr(stmem.urllib.request, "urlopen", _boom)


@pytest.fixture
def frozen_now(monkeypatch):
    """Pin stmem's clock to FROZEN_NOW for the duration of the test."""
    monkeypatch.setattr(stmem, "datetime", _FrozenDatetime)
    return FROZEN_NOW


def envelope_text(
    *,
    created_at=_MISSING,
    ttl_hours=_MISSING,
    supersedes=None,
    run_id="run-1",
    agent="agent-a",
    step=1,
    epistemic="observed",
    body="a self-contained fact",
):
    """Build an stmm/v1 entry exactly as cmd_write lays it out."""
    env = {
        "stmm": 1,
        "schema": "stmm/v1",
        "run_id": run_id,
        "agent": agent,
        "step": step,
        "epistemic": epistemic,
        "supersedes": supersedes or [],
    }
    if created_at is not _MISSING:
        env["created_at"] = created_at
    if ttl_hours is not _MISSING:
        env["ttl_hours"] = ttl_hours
    return json.dumps(env) + NL + body


def doc(doc_id, text, *, as_dict_payload=False):
    """One document entry in the shape GET .../documents returns."""
    payload = {"text": text} if as_dict_payload else text
    return {"id": doc_id, "payload": payload}


def page(docs, next_cursor=None):
    return {"documents": list(docs), "next_cursor": next_cursor}


class FakeStore:
    """In-memory stand-in for stmem._req; records every call."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []
        self.gets = []
        self.deleted = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == "GET":
            self.gets.append(path)
            if not self._pages:
                return page([])
            return self._pages.pop(0)
        if method == "DELETE":
            doc_id = path.rsplit("/", 1)[-1]
            self.deleted.append(doc_id)
            return {"status": "ok", "deleted": doc_id}
        raise AssertionError(f"unexpected method {method!r}")


def reap_args(ttl_hours=24, dry_run=False):
    return argparse.Namespace(ttl_hours=ttl_hours, dry_run=dry_run)


def run_reap(monkeypatch, docs, *, ttl_hours=24, dry_run=False, env_dry_run=None):
    """Run cmd_reap against an in-memory store; return the store for asserts."""
    store = FakeStore([page(docs)])
    monkeypatch.setattr(stmem, "_req", store)
    if env_dry_run is None:
        monkeypatch.delenv("STMM_DRY_RUN", raising=False)
    else:
        monkeypatch.setenv("STMM_DRY_RUN", env_dry_run)
    stmem.cmd_reap(reap_args(ttl_hours=ttl_hours, dry_run=dry_run))
    return store


# Expiry fixtures relative to FROZEN_NOW (2026-10-11T12:00:00Z), ttl 24h:
EXPIRED_25H_AGO = "2026-10-10T11:00:00Z"  # expires 2026-10-11T11:00Z -> expired
FRESH_1H_AGO = "2026-10-11T11:00:00Z"  # expires 2026-10-12T11:00Z -> kept
EXPIRES_EXACTLY_NOW = "2026-10-10T12:00:00Z"  # expires == now -> kept (< is strict)


# ---------------------------------------------------------------------------
# 1. _parse_envelope
# ---------------------------------------------------------------------------


def test_parse_envelope_valid():
    text = envelope_text(created_at=FRESH_1H_AGO, ttl_hours=24, supersedes=["old-1"], body="fact body")
    env, body = stmem._parse_envelope(text)
    assert env is not None
    assert env["stmm"] == 1 and env["schema"] == "stmm/v1"
    assert env["supersedes"] == ["old-1"] and env["ttl_hours"] == 24
    assert body == "fact body"  # remainder is the fact, newline consumed


def test_parse_envelope_multiline_body_preserved():
    text = envelope_text(created_at=FRESH_1H_AGO) + NL + "second line" + NL + "third line"
    env, body = stmem._parse_envelope(text)
    assert env is not None
    assert body == "a self-contained fact" + NL + "second line" + NL + "third line"


def test_parse_envelope_invalid_json_first_line():
    text = '{"stmm" not json at all' + NL + "body"
    env, body = stmem._parse_envelope(text)
    assert env is None
    assert body == text  # original text returned untouched


def test_parse_envelope_json_without_stmm_key():
    text = '{"run_id": "r1", "ttl_hours": 24}' + NL + "body"
    env, body = stmem._parse_envelope(text)
    assert env is None and body == text


def test_parse_envelope_wrong_stmm_version_is_rejected():
    # Reaches the isinstance/version guard (prefilter passes, JSON parses).
    text = '{"stmm": 2, "schema": "stmm/v1"}' + NL + "body"
    env, body = stmem._parse_envelope(text)
    assert env is None and body == text


@pytest.mark.parametrize(
    "text",
    [
        "[1, 2, 3]" + NL + "body",
        '"stmm"' + NL + "body",
        "42" + NL + "body",
        '{"other": 1}' + NL + "body",
    ],
)
def test_parse_envelope_non_dict_json(text):
    env, body = stmem._parse_envelope(text)
    assert env is None and body == text


def test_parse_envelope_leading_whitespace_before_envelope():
    text = "   " + TAB + envelope_text(created_at=FRESH_1H_AGO, body="indented")
    env, body = stmem._parse_envelope(text)
    assert env is not None, "lstrip() prefilter should still accept the envelope"
    assert body == "indented"


def test_parse_envelope_no_newline_after_envelope():
    # Envelope only, no fact line: still parses; the body is the empty string.
    text = json.dumps({"stmm": 1, "schema": "stmm/v1", "created_at": FRESH_1H_AGO})
    env, body = stmem._parse_envelope(text)
    assert env is not None and body == ""


def test_parse_envelope_empty_string():
    env, body = stmem._parse_envelope("")
    assert env is None and body == ""


def test_parse_envelope_whitespace_only():
    text = "   " + NL + TAB + NL
    env, body = stmem._parse_envelope(text)
    assert env is None and body == text


@pytest.mark.parametrize("value", [None, 42, b'{"stmm": 1}' + NL.encode() + b"x", {"stmm": 1}, ["x"]])
def test_parse_envelope_non_string_input(value):
    env, body = stmem._parse_envelope(value)
    assert env is None and body == value


# ---------------------------------------------------------------------------
# 2. _find_text
# ---------------------------------------------------------------------------


def test_find_text_str_payload():
    text = envelope_text(created_at=FRESH_1H_AGO)
    assert stmem._find_text(text) == text


def test_find_text_dict_payload_containing_envelope():
    text = envelope_text(created_at=FRESH_1H_AGO)
    assert stmem._find_text({"text": text, "source": "agent"}) == text
    assert stmem._find_text({"meta": {"nested": True}, "content": text}) == text


def test_find_text_dict_without_envelope():
    assert stmem._find_text({"text": "plain document, no envelope", "source": "x"}) is None
    assert stmem._find_text({}) is None


def test_find_text_dict_with_non_string_values():
    assert stmem._find_text({"n": 1, "b": True, "none": None, "lst": ['"stmm"']}) is None


@pytest.mark.parametrize("value", [None, 42, 3.5, b'{"stmm": 1}', ["x"]])
def test_find_text_non_str_non_dict(value):
    assert stmem._find_text(value) is None


def test_find_text_decoy_containing_stmm_is_returned_then_rejected():
    """The '"stmm"' substring test is a cheap prefilter, not a validator.

    A decoy string that merely CONTAINS '"stmm"' is handed back by _find_text;
    _parse_envelope is what rejects it (so nothing downstream trusts it).
    """
    decoy = 'note: the key "stmm" appears in this text but this is not an envelope'
    assert stmem._find_text({"text": decoy}) == decoy  # prefilter is substring-only
    env, body = stmem._parse_envelope(decoy)
    assert env is None and body == decoy  # real validation rejects it


def test_find_text_prefers_real_envelope_over_decoy():
    """FINDING F4 (fixed): a decoy must not shadow the real envelope.

    _find_text now prefers the value that actually PARSES as an envelope; the
    '"stmm"' substring scan is only a fallback for payloads where nothing
    parses. Dict order must no longer decide.
    """
    real = envelope_text(created_at=FRESH_1H_AGO)
    decoy = 'this note quotes the envelope key "stmm" verbatim'
    payload = {"notes": decoy, "text": real}  # decoy FIRST in dict order
    assert stmem._find_text(payload) == real  # the REAL envelope wins
    env, _ = stmem._parse_envelope(stmem._find_text(payload))
    assert env is not None, "the entry must survive the decoy"


# ---------------------------------------------------------------------------
# 3. cmd_reap selection
# ---------------------------------------------------------------------------


def test_reap_deletes_expired_entry(monkeypatch, capsys, frozen_now):
    store = run_reap(monkeypatch, [doc("expired-1", envelope_text(created_at=EXPIRED_25H_AGO, ttl_hours=24))])
    assert store.deleted == ["expired-1"]
    out = capsys.readouterr().out
    assert "entries=1 to_delete=1" in out
    assert "(expired)" in out


def test_reap_keeps_not_yet_expired_entry(monkeypatch, capsys, frozen_now):
    store = run_reap(monkeypatch, [doc("fresh-1", envelope_text(created_at=FRESH_1H_AGO, ttl_hours=24))])
    assert store.deleted == []
    assert "entries=1 to_delete=0" in capsys.readouterr().out


def test_reap_keeps_entry_expiring_exactly_now(monkeypatch, frozen_now):
    """Boundary: expiry comparison is strict (expires < now), so == now is kept."""
    store = run_reap(monkeypatch, [doc("edge-1", envelope_text(created_at=EXPIRES_EXACTLY_NOW, ttl_hours=24))])
    assert store.deleted == []


def test_reap_deletes_superseded_entry(monkeypatch, capsys, frozen_now):
    docs = [
        doc("old-1", envelope_text(created_at=FRESH_1H_AGO)),
        doc("new-1", envelope_text(created_at=FRESH_1H_AGO, supersedes=["old-1"])),
    ]
    store = run_reap(monkeypatch, docs)
    assert store.deleted == ["old-1"]  # the superseding entry itself stays
    out = capsys.readouterr().out
    assert "to_delete=1" in out and "(superseded)" in out


def test_reap_superseded_takes_precedence_over_expired(monkeypatch, capsys, frozen_now):
    docs = [
        doc("old-1", envelope_text(created_at=EXPIRED_25H_AGO)),  # also expired
        doc("new-1", envelope_text(created_at=FRESH_1H_AGO, supersedes=["old-1"])),
    ]
    store = run_reap(monkeypatch, docs)
    assert store.deleted == ["old-1"]
    assert "(superseded)" in capsys.readouterr().out  # if/elif order


def test_reap_ignores_dangling_supersedes(monkeypatch, frozen_now):
    """Invariant 2: a supersedes id not present in this sweep deletes nothing."""
    store = run_reap(monkeypatch, [doc("live-1", envelope_text(created_at=FRESH_1H_AGO, supersedes=["ghost-id"]))])
    assert store.deleted == []


def test_reap_supersede_wins_over_malformed_expiry(monkeypatch, capsys, frozen_now):
    """A live supersede is an explicit instruction: it beats a malformed expiry.

    The superseded entry has ttl_hours=null, which on its own would make it
    unparseable -> KEPT. Because a live entry names it in supersedes, it must
    be deleted anyway (and the live entry itself must survive).
    """
    docs = [
        doc("old-malformed", envelope_text(created_at=FRESH_1H_AGO, ttl_hours=None)),
        doc("new-live", envelope_text(created_at=FRESH_1H_AGO, supersedes=["old-malformed"])),
    ]
    store = run_reap(monkeypatch, docs)
    assert store.deleted == ["old-malformed"]
    out = capsys.readouterr().out
    assert "to_delete=1" in out and "(superseded)" in out


def test_reap_keeps_entry_with_missing_created_at(monkeypatch, capsys, frozen_now):
    """Fail-closed invariant 1: no created_at -> KEEP."""
    store = run_reap(monkeypatch, [doc("no-date", envelope_text(ttl_hours=0))])
    assert store.deleted == []
    assert "to_delete=0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "created_at",
    [
        "yesterday",  # unparseable
        "2026-10-10",  # date only
        "2026-10-10T11:00:00",  # no zone
        "",  # empty
    ],
)
def test_reap_keeps_entry_with_unparseable_created_at(monkeypatch, frozen_now, created_at):
    store = run_reap(monkeypatch, [doc("bad-date", envelope_text(created_at=created_at, ttl_hours=0))])
    assert store.deleted == []


def test_reap_non_string_created_at_does_not_abort_the_sweep(monkeypatch, capsys, frozen_now):
    """FINDING F2 (fixed): a JSON null created_at must not abort the sweep.

    Regression guard: before the fix a TypeError escaped cmd_reap and NOTHING
    was reaped. The sweep must now complete: the malformed entry is KEPT
    (fail-closed) while the healthy expired entry is still deleted. No
    except-clause here on purpose -- any escaping exception fails the test.
    """
    docs = [
        doc("null-date", envelope_text(created_at=None, ttl_hours=0)),
        doc("expired-good", envelope_text(created_at=EXPIRED_25H_AGO)),
    ]
    store = run_reap(monkeypatch, docs)  # must not raise
    assert "null-date" not in store.deleted  # fail-closed for the malformed entry
    assert store.deleted == ["expired-good"]  # healthy entries still reaped
    assert "skipped_unparseable=1" in capsys.readouterr().out


@pytest.mark.parametrize("bad_ttl", [None, "24"])
def test_reap_non_numeric_ttl_hours_does_not_abort_the_sweep(monkeypatch, capsys, frozen_now, bad_ttl):
    """FINDING F2 (fixed): a non-numeric ttl_hours must not abort the sweep.

    Regression guard: timedelta(hours=None / "24") used to raise TypeError out
    of cmd_reap, abandoning every other entry. The value is NOT guessed
    (fail-closed -> the entry is KEPT) but the healthy expired entry must
    still be reaped. No except-clause here on purpose.
    """
    docs = [
        doc("bad-ttl", envelope_text(created_at=EXPIRED_25H_AGO, ttl_hours=bad_ttl)),
        doc("expired-good", envelope_text(created_at=EXPIRED_25H_AGO)),
    ]
    store = run_reap(monkeypatch, docs)  # must not raise
    assert "bad-ttl" not in store.deleted
    assert store.deleted == ["expired-good"]
    assert "skipped_unparseable=1" in capsys.readouterr().out


@pytest.mark.parametrize("created_at", ["2026-10-09T11:00:00+00:00", "2026-10-09T11:00:00.000Z"])
def test_reap_expires_offset_and_fractional_created_at(monkeypatch, frozen_now, created_at):
    """FINDING F3 (fixed): zone-bearing ISO-8601 variants parse and DO expire.

    Both values are 2026-10-09T11:00:00Z = 49h before the frozen clock, with
    ttl_hours=24, so the entry must be deleted.
    """
    store = run_reap(monkeypatch, [doc("iso-1", envelope_text(created_at=created_at, ttl_hours=24))])
    assert store.deleted == ["iso-1"]


@pytest.mark.parametrize("created_at", ["2026-10-10", "2026-10-10T11:00:00"])
def test_reap_keeps_naive_created_at(monkeypatch, capsys, frozen_now, created_at):
    """Naive / date-only timestamps are ambiguous -> fail-closed: entry KEPT.

    ttl_hours=0 makes both of these immediately expired IF the zone were
    known, so this is the strictest form of the fail-closed check.
    """
    store = run_reap(monkeypatch, [doc("naive-1", envelope_text(created_at=created_at, ttl_hours=0))])
    assert store.deleted == []
    assert "skipped_unparseable=1" in capsys.readouterr().out


def test_reap_missing_ttl_hours_falls_back_to_argument(monkeypatch, capsys, frozen_now):
    """No ttl_hours in the envelope -> the --ttl-hours argument decides."""
    text = envelope_text(created_at=EXPIRED_25H_AGO)  # 25h old, no ttl_hours key
    store = run_reap(monkeypatch, [doc("ttl-fallback", text)], ttl_hours=24)
    assert store.deleted == ["ttl-fallback"]
    assert "(expired)" in capsys.readouterr().out


def test_reap_missing_ttl_hours_respects_larger_argument(monkeypatch, frozen_now):
    text = envelope_text(created_at=EXPIRED_25H_AGO)  # same entry, 48h argument
    store = run_reap(monkeypatch, [doc("ttl-fallback", text)], ttl_hours=48)
    assert store.deleted == []


def test_reap_dry_run_env_deletes_nothing(monkeypatch, capsys, frozen_now):
    docs = [
        doc("expired-1", envelope_text(created_at=EXPIRED_25H_AGO)),
        doc("old-1", envelope_text(created_at=FRESH_1H_AGO)),
        doc("new-1", envelope_text(created_at=FRESH_1H_AGO, supersedes=["old-1"])),
    ]
    store = run_reap(monkeypatch, docs, env_dry_run="1")
    assert store.deleted == []
    assert not [c for c in store.calls if c[0] == "DELETE"]
    out = capsys.readouterr().out
    assert "to_delete=2" in out  # selection still reported
    assert out.count("DRY_RUN would delete") == 2


def test_reap_dry_run_flag_deletes_nothing(monkeypatch, capsys, frozen_now):
    store = run_reap(monkeypatch, [doc("expired-1", envelope_text(created_at=EXPIRED_25H_AGO))], dry_run=True)
    assert store.deleted == []
    assert "DRY_RUN would delete" in capsys.readouterr().out


def test_reap_mixed_sweep_deletes_only_the_right_ids(monkeypatch, frozen_now):
    docs = [
        doc("expired-1", envelope_text(created_at=EXPIRED_25H_AGO)),
        doc("old-1", envelope_text(created_at=FRESH_1H_AGO)),
        doc("new-1", envelope_text(created_at=FRESH_1H_AGO, supersedes=["old-1"])),
        doc("fresh-1", envelope_text(created_at=FRESH_1H_AGO)),
        doc("no-date", envelope_text(ttl_hours=0)),
        doc("plain", "no envelope here at all"),
    ]
    store = run_reap(monkeypatch, docs)
    assert sorted(store.deleted) == ["expired-1", "old-1"]
    assert len(store.gets) == 1  # one listing page, no re-fetch


# ---------------------------------------------------------------------------
# 4. iter_documents pagination
# ---------------------------------------------------------------------------


def test_iter_documents_two_pages_then_stops(monkeypatch):
    store = FakeStore(
        [
            page([doc("a", "text-a"), doc("b", "text-b")], next_cursor="CURSOR-1"),
            page([doc("c", "text-c")], next_cursor=None),
        ]
    )
    monkeypatch.setattr(stmem, "_req", store)

    seen = list(stmem.iter_documents())

    assert [i for i, _ in seen] == ["a", "b", "c"]
    assert [p for _, p in seen] == ["text-a", "text-b", "text-c"]
    assert len(store.gets) == 2, "must stop after the page with no next_cursor"
    assert "limit=500" in store.gets[0]
    assert "cursor=CURSOR-1" in store.gets[1], "cursor must be fed back on page 2"


@pytest.mark.parametrize("final_cursor", [None, ""])
def test_iter_documents_stops_on_empty_next_cursor(monkeypatch, final_cursor):
    store = FakeStore([page([doc("only", "x")], next_cursor=final_cursor)])
    monkeypatch.setattr(stmem, "_req", store)
    assert list(stmem.iter_documents()) == [("only", "x")]
    assert len(store.gets) == 1


def test_iter_documents_missing_documents_key_stops(monkeypatch):
    store = FakeStore([{"next_cursor": None}])
    monkeypatch.setattr(stmem, "_req", store)
    assert list(stmem.iter_documents()) == []


def test_iter_documents_terminates_on_repeating_cursor(monkeypatch, capsys):
    """FINDING F1 (fixed): a repeating next_cursor STOPS pagination.

    stmem.py keeps a seen-cursor set, so a stuck token cannot loop forever:
    the page is read once more, then the generator returns and writes a
    WARNING to stderr. The in-call cap below is a safety net -- if this ever
    regresses to the pre-fix infinite loop, the test fails instead of hanging.
    """
    calls = {"n": 0}

    def _req(method, path, body=None):
        assert method == "GET"
        calls["n"] += 1
        assert calls["n"] <= 3, "pagination did not stop on the repeating cursor"
        return {"documents": [doc("dup-1", "x")], "next_cursor": "STUCK-CURSOR"}

    monkeypatch.setattr(stmem, "_req", _req)

    seen = list(stmem.iter_documents())  # must terminate on its own

    assert [i for i, _ in seen] == ["dup-1", "dup-1"]  # read twice, then stop
    assert calls["n"] == 2
    err = capsys.readouterr().err
    assert "WARNING" in err and "repeating next_cursor" in err and "STUCK-CURSOR" in err


# ---------------------------------------------------------------------------
# 5. cmd_ensure (idempotent dataset bootstrap) -- task-8
# ---------------------------------------------------------------------------


def http_error(code, body=b"", *, readable=True):
    """Build an HTTPError; readable=False models a body that cannot be read."""
    fp = io.BytesIO(body) if readable else None
    return urllib.error.HTTPError("http://stmem.invalid/api/datasets", code, "err", {}, fp)


def run_ensure(monkeypatch, *, returns=None, raises=None):
    """Call cmd_ensure with _req stubbed; return (exit_code, calls)."""
    calls = []

    def _req(method, path, body=None):
        calls.append((method, path, body))
        if raises is not None:
            raise raises
        return returns

    monkeypatch.setattr(stmem, "_req", _req)
    return stmem.cmd_ensure(None), calls


def test_cmd_ensure_posts_the_dataset(monkeypatch, capsys):
    code, calls = run_ensure(monkeypatch, returns={"status": "created"})
    assert code == 0
    assert len(calls) == 1
    method, path, body = calls[0]
    assert method == "POST" and path == "/api/datasets"
    assert body["name"] == stmem.DATASET
    assert "created dataset" in capsys.readouterr().out


def test_cmd_ensure_200_is_success(monkeypatch):
    code, _ = run_ensure(monkeypatch, returns={})
    assert code == 0


def test_cmd_ensure_409_is_success(monkeypatch, capsys):
    """409 == the dataset already exists: idempotent success, exit 0."""
    code, _ = run_ensure(monkeypatch, raises=http_error(409, b'{"detail":"exists"}'))
    assert code == 0
    assert "already exists" in capsys.readouterr().out


def test_cmd_ensure_409_with_unreadable_body_is_still_success(monkeypatch):
    """409 is decided by the STATUS, so an unreadable body must not change it."""
    code, _ = run_ensure(monkeypatch, raises=http_error(409, readable=False))
    assert code == 0


def test_cmd_ensure_400_already_exists_is_success(monkeypatch):
    """A 400 is the documented fallback for 'already exists' bodies."""
    code, _ = run_ensure(monkeypatch, raises=http_error(400, b'{"detail":"dataset already exists"}'))
    assert code == 0


@pytest.mark.parametrize(
    "body",
    [
        b'{"detail":"dataset does not exist"}',  # N2: the broad "exist" trap
        b'{"detail":"Invalid dataset name"}',
        b'{"detail":"conflicts with existing resource"}',  # N2: "existing" trap
    ],
)
def test_cmd_ensure_400_errors_are_failures(monkeypatch, capsys, body):
    """A 400 that does NOT say the dataset exists is a failure (exit 1)."""
    code, _ = run_ensure(monkeypatch, raises=http_error(400, body))
    assert code == 1
    assert "FAILED to ensure dataset" in capsys.readouterr().err


@pytest.mark.parametrize("status", [500, 502, 503])
def test_cmd_ensure_5xx_is_failure(monkeypatch, capsys, status):
    code, _ = run_ensure(monkeypatch, raises=http_error(status, b"boom"))
    assert code == 1
    assert "FAILED to ensure dataset" in capsys.readouterr().err


def test_cmd_ensure_urlerror_is_failure(monkeypatch, capsys):
    code, _ = run_ensure(monkeypatch, raises=urllib.error.URLError("connection refused"))
    assert code == 1
    assert "FAILED to ensure dataset" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6. _req transport: auth headers (F8) + non-JSON 2xx body (N2) -- task-8
# ---------------------------------------------------------------------------


class FakeHTTPResponse:
    """Minimal urlopen() result: a context manager whose read() yields bytes."""

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def capture_request(monkeypatch, body=b"{}"):
    """Replace urlopen with a capturer; return the dict it fills in."""
    captured = {}

    def _urlopen(req, timeout=None):
        captured["request"] = req
        captured["timeout"] = timeout
        return FakeHTTPResponse(body)

    monkeypatch.setattr(stmem.urllib.request, "urlopen", _urlopen)
    return captured


def sent_headers(captured):
    """Lower-cased header map of the captured Request (urllib capitalizes)."""
    return {k.lower(): v for k, v in captured["request"].headers.items()}


def test_req_sends_both_api_key_headers(monkeypatch):
    """F8: the live tree reads X-API-Key, the baseline middleware reads
    X-RAG-Api-Key. BOTH must be sent or auth silently fails on one of them."""
    monkeypatch.setattr(stmem, "BASE", "http://stmem.invalid")
    monkeypatch.setattr(stmem, "API_KEY", "key-abc")
    monkeypatch.setattr(stmem, "PASSWORD", "pw-xyz")
    captured = capture_request(monkeypatch, body=b'{"ok": true}')

    assert stmem._req("GET", "/api/datasets") == {"ok": True}

    headers = sent_headers(captured)
    assert headers["x-api-key"] == "key-abc"
    assert headers["x-rag-api-key"] == "key-abc"
    assert headers["x-dataset-password"] == "pw-xyz"
    assert headers["content-type"] == "application/json"
    assert captured["request"].get_method() == "GET"
    assert captured["request"].full_url == "http://stmem.invalid/api/datasets"


def test_req_omits_auth_headers_when_unset(monkeypatch):
    """No credentials configured => no auth header may be sent at all."""
    monkeypatch.setattr(stmem, "BASE", "http://stmem.invalid")
    monkeypatch.setattr(stmem, "API_KEY", None)
    monkeypatch.setattr(stmem, "PASSWORD", None)
    captured = capture_request(monkeypatch)

    stmem._req("GET", "/api/datasets")

    headers = sent_headers(captured)
    assert "x-api-key" not in headers
    assert "x-rag-api-key" not in headers
    assert "x-dataset-password" not in headers


def test_req_empty_body_returns_empty_dict(monkeypatch):
    """A blank 2xx body is normalised to {}, not a JSON error."""
    monkeypatch.setattr(stmem, "BASE", "http://stmem.invalid")
    monkeypatch.setattr(stmem, "API_KEY", None)
    capture_request(monkeypatch, body=b"   ")
    assert stmem._req("GET", "/api/datasets") == {}


def test_req_non_json_2xx_raises_urlerror(monkeypatch):
    """N2: an HTML gateway page on a 200 must surface as a transport error,
    not an uncaught JSONDecodeError (which callers do not catch)."""
    monkeypatch.setattr(stmem, "BASE", "http://stmem.invalid")
    monkeypatch.setattr(stmem, "API_KEY", None)
    capture_request(monkeypatch, body=b"<html>502 Bad Gateway</html>")

    with pytest.raises(urllib.error.URLError):
        stmem._req("GET", "/api/datasets")


# ---------------------------------------------------------------------------
# 7. TTL range containment (N1) -- task-8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (24, 24.0),
        (876000, 876000.0),  # the documented maximum (_MAX_TTL_HOURS)
        (876001, None),  # one past the boundary
        (1e8, None),  # the value that used to raise OverflowError
        (1e9, None),
        (float("inf"), None),
        (float("-inf"), None),
        (float("nan"), None),
        (-1, None),
        (True, None),  # bool is not a TTL
        ("24", None),
        (None, None),
    ],
)
def test_ttl_hours_range_check(value, expected):
    """N1: only a finite, in-range number is a TTL; everything else is None."""
    assert stmem._ttl_hours(value, 24) == expected


@pytest.mark.parametrize("bad_ttl", [1e8, 1e9, float("inf"), float("nan")])
def test_reap_out_of_range_ttl_does_not_abort_the_sweep(monkeypatch, capsys, frozen_now, bad_ttl):
    """N1: an out-of-range TTL must not kill the sweep.

    Before the fix, 1e8 reached the timedelta arithmetic and raised an uncaught
    OverflowError that aborted everything -- the healthy expired entry below was
    never reaped. The malformed entry is now KEPT (fail-closed).
    """
    docs = [
        doc("bad-ttl", envelope_text(created_at=FRESH_1H_AGO, ttl_hours=bad_ttl)),
        doc("expired-good", envelope_text(created_at=EXPIRED_25H_AGO)),
    ]
    store = run_reap(monkeypatch, docs)  # must not raise
    assert "bad-ttl" not in store.deleted
    assert store.deleted == ["expired-good"]
    assert "skipped_unparseable=1" in capsys.readouterr().out


def test_reap_year_9999_created_at_does_not_abort_the_sweep(monkeypatch, capsys, frozen_now):
    """N1: created_at + ttl overflowing datetime.max is contained, not fatal."""
    docs = [
        doc("far-future", envelope_text(created_at="9999-12-31T23:59:59Z", ttl_hours=24)),
        doc("expired-good", envelope_text(created_at=EXPIRED_25H_AGO)),
    ]
    store = run_reap(monkeypatch, docs)  # must not raise
    assert "far-future" not in store.deleted
    assert store.deleted == ["expired-good"]
    assert "skipped_unparseable=1" in capsys.readouterr().out
