"""PCAI_OIDC_ENV_PREFIX — the import-time env namespace for the shared OIDC modules.

2026-11-10 audit (P2-17): the prefix switch shipped with ZERO coverage.  It is
the one import-time setting on this surface (every other OIDC knob is read per
request), and it is consumed by TWO sibling modules that must agree, so a
regression here silently disables OIDC for a prefixed consumer (SQLhandler) or
makes the two modules read different variable names.

The constants are resolved at import, so the namespace tests run in a
subprocess with a scrubbed environment; the alias-precedence tests run
in-process by patching the module's _ENV_PREFIX.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from multimodal_rag.utils import oidc_identity  # noqa: E402

_OIDC_PREFIXES = ("OIDC_", "RAG_OIDC_")


def _scrubbed_env(**extra: str) -> dict[str, str]:
    """Inherit the environment minus every OIDC knob, then apply *extra*."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_OIDC_PREFIXES) and k != "PCAI_OIDC_ENV_PREFIX"}
    env["PYTHONPATH"] = str(SRC)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _run(code: str, **extra: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_scrubbed_env(**extra),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


_PROBE = """
from multimodal_rag.utils import oidc_identity, oidc_sso
print(oidc_identity.ENABLED_ENV)
print(oidc_identity.ISSUER_ENV)
print(oidc_identity.FORCE_MIN_INTERVAL_ENV)
print(oidc_sso.ENABLED_ENV)
print(oidc_sso.CLIENT_ID_ENV)
print(oidc_sso.ISSUER_ENV)
"""


def test_default_namespace_is_unprefixed():
    """With no prefix the names are app-agnostic OIDC_* (the fleet default)."""
    enabled, issuer, force, sso_enabled, sso_client, sso_issuer = _run(_PROBE).splitlines()
    assert enabled == "OIDC_ENABLED"
    assert issuer == "OIDC_ISSUER"
    assert force == "JWKS_FORCE_MIN_INTERVAL"
    assert sso_enabled == "OIDC_SSO_ENABLED"
    assert sso_client == "OIDC_SSO_CLIENT_ID"
    assert sso_issuer == "OIDC_ISSUER"


def test_prefix_namespaces_every_variable():
    """PCAI_OIDC_ENV_PREFIX=SQLHANDLER gives SQLHANDLER_OIDC_* / SQLHANDLER_OIDC_SSO_*."""
    out = _run(_PROBE, PCAI_OIDC_ENV_PREFIX="SQLHANDLER")
    enabled, issuer, force, sso_enabled, sso_client, sso_issuer = out.splitlines()
    assert enabled == "SQLHANDLER_OIDC_ENABLED"
    assert issuer == "SQLHANDLER_OIDC_ISSUER"
    # The force-interval knob deliberately has no OIDC_ segment.
    assert force == "SQLHANDLER_JWKS_FORCE_MIN_INTERVAL"
    assert sso_enabled == "SQLHANDLER_OIDC_SSO_ENABLED"
    assert sso_client == "SQLHANDLER_OIDC_SSO_CLIENT_ID"
    # oidc_sso's discovery default must read the SAME issuer variable.
    assert sso_issuer == "SQLHANDLER_OIDC_ISSUER"


def test_prefix_is_uppercased_and_underscore_suffixed():
    """A lower-case / padded prefix is normalized, not used verbatim."""
    out = _run(_PROBE, PCAI_OIDC_ENV_PREFIX="  sqlhandler  ")
    assert out.splitlines()[0] == "SQLHANDLER_OIDC_ENABLED"


def test_prefix_disables_the_legacy_rag_alias():
    """A namespaced consumer must NOT silently pick up another app's RAG_OIDC_*.

    The alias exists for the pre-rename RAG chart, which only ever deployed the
    unprefixed namespace; honouring it under a prefix would let two apps on one
    node configure each other.
    """
    code = """
from multimodal_rag.utils import oidc_identity, oidc_sso
print(oidc_identity._flag_set())
print(oidc_sso._flag_set())
"""
    out = _run(code, PCAI_OIDC_ENV_PREFIX="SQLHANDLER", RAG_OIDC_ENABLED="1", RAG_OIDC_SSO_ENABLED="1")
    assert out.splitlines() == ["False", "False"]


def test_no_prefix_still_reads_the_legacy_rag_alias(monkeypatch):
    """Without a prefix the deprecated RAG_OIDC_* names still activate the resolver."""
    monkeypatch.setattr(oidc_identity, "_ENV_PREFIX", "")
    monkeypatch.delenv("OIDC_ENABLED", raising=False)
    monkeypatch.setenv("RAG_OIDC_ENABLED", "1")
    assert oidc_identity._flag_set() is True


def test_canonical_name_wins_over_the_legacy_alias(monkeypatch):
    """The canonical OIDC_* value is read first; the alias is only a fallback."""
    monkeypatch.setattr(oidc_identity, "_ENV_PREFIX", "")
    monkeypatch.setenv("OIDC_ENABLED", "0")
    monkeypatch.setenv("RAG_OIDC_ENABLED", "1")
    assert oidc_identity._flag_set() is False


@pytest.mark.parametrize("var", ["NO_PROXY", "no_proxy"])
def test_bracketed_ipv6_no_proxy_does_not_break_import(var):
    """2026-11-10 audit P1-9: a bracketed IPv6 NO_PROXY entry made the package unimportable.

    httpx2 parses each NO_PROXY entry as a URL pattern and raises
    InvalidURL on "[::1]".  The package now normalizes it at import, so a
    plain import must succeed and leave an httpx-parseable value behind.
    """
    code = f"""
import os
import multimodal_rag  # noqa: F401  -- the import itself is the assertion
import httpx2
client = httpx2.Client()
client.close()
print(os.environ.get({var!r}, ""))
"""
    env = _scrubbed_env()
    env[var] = "127.0.0.1,localhost,::1,[::1]"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[" not in proc.stdout.strip()
