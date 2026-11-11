"""Zone W4 regression tests: fleet tooling, operator scripts, Open WebUI filter.

Every test here pins a defect from documentation/AUDIT-2026-11-10.md (P1-7,
P1-8, P2-11, P2-15, P2-16, P3-6, P3-10, P3-11, P3-12).  Each one FAILS on the
pre-remediation code.

The two fleet tools and the scripts are not installed packages, so they are
loaded by path.  Nothing here touches the real hardlink mesh, the real
pcai_utils tree or the network: every tree is synthetic and every HTTP call is
either stubbed or intercepted.
"""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_module(name: str, rel_path: str) -> Any:
    """Import a repo file by path (tools/scripts are not installed packages)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None, f"cannot load {rel_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# P1-7 -- fleet_reconcile mirror-pair reconcile must REPAIR, never PROMOTE
# ===========================================================================


def _mirror_tree(tmp_path: Path, mirror_text: str) -> tuple[Any, Path, Path, Path]:
    """A synthetic repo + mirror pair whose mirror alone moved.

    Returns (fleet_reconcile module, repo_copy, mirror_path, state_path).
    """
    fleet = load_module("fleet_reconcile_under_test", "src/multimodal_rag/utils/fleet_reconcile.py")
    repo = tmp_path / "repo"
    dest = tmp_path / "dest"
    rel = Path("src") / "mypkg" / "utils" / "myutil.py"
    repo_copy = repo / rel
    mirror = dest / rel
    for path, text in ((repo_copy, "REPO-CONTENT\n"), (mirror, mirror_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (repo / "hardlink_config.json").write_text(
        json.dumps({"source": str(repo), "dest": str(dest), "ignore": []}), encoding="utf-8"
    )
    state = tmp_path / "state.json"
    key = str(repo_copy.relative_to(tmp_path))
    baseline = fleet.sha_of(repo_copy)
    state.write_text(json.dumps({key: baseline, f"{key} (mirror)": baseline}), encoding="utf-8")
    return fleet, repo_copy, mirror, state


def test_mirror_only_change_repairs_from_the_repo_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-7: a mirror that moved alone must never redefine fleet content.

    Pre-fix, reconcile_file promoted the MIRROR bytes into the repo copy (fleet
    truth), so a Syncthing reversion of a delivery-tree file propagated STALE
    content to every linked repo.
    """
    fleet, repo_copy, mirror, state = _mirror_tree(tmp_path, "STALE-PEER-REVERSION\n")
    repo = tmp_path / "repo"
    rel = repo_copy.relative_to(repo)
    monkeypatch.setattr(fleet, "HPE", tmp_path)
    monkeypatch.setattr(fleet, "STATE", state)
    monkeypatch.setattr(fleet, "EXCEPTIONS", tmp_path / "exceptions.json")
    monkeypatch.setattr(fleet, "expected_copies", lambda _repo: [("myutil.py", repo / rel)])

    reconciler = fleet.Reconciler(run=True, build=False, verbose=True)
    assert reconciler.run_fleet([repo]) == 0, "mirror-only move must reconcile cleanly"

    assert repo_copy.read_text(encoding="utf-8") == "REPO-CONTENT\n", (
        "the repo copy (fleet truth) was overwritten by mirror content"
    )
    assert mirror.read_text(encoding="utf-8") == "REPO-CONTENT\n", "the mirror was not repaired"
    assert os.stat(repo_copy).st_ino == os.stat(mirror).st_ino, "the mirror was not re-linked"


def test_repo_copy_still_promotes_into_the_fleet_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The normal direction is untouched: a repo copy still promotes."""
    fleet, repo_copy, mirror, state = _mirror_tree(tmp_path, "REPO-CONTENT\n")
    monkeypatch.setattr(fleet, "HPE", tmp_path)
    monkeypatch.setattr(fleet, "STATE", state)
    key = str(repo_copy.relative_to(tmp_path))
    mirror.write_text("NEW-REPO-WORK\n", encoding="utf-8")
    reconciler = fleet.Reconciler(run=True, build=False, verbose=True)
    assert reconciler.reconcile_file(key, repo_copy, [mirror]) is True
    assert repo_copy.read_text(encoding="utf-8") == "NEW-REPO-WORK\n"


# ===========================================================================
# P2-11 -- promote() must preserve the source's mode (exec bit)
# ===========================================================================


def test_promote_preserves_the_executable_bit(tmp_path: Path) -> None:
    """P2-11: promoting a 100755 mesh file used to chmod it to 0o644 fleet-wide."""
    fleet = load_module("fleet_reconcile_promote_under_test", "src/multimodal_rag/utils/fleet_reconcile.py")
    source = tmp_path / "bump_version.sh"
    source.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    source.chmod(0o755)
    copy = tmp_path / "copy.sh"
    copy.write_text("#!/bin/sh\necho new\n", encoding="utf-8")
    copy.chmod(0o644)

    assert fleet.promote(copy, source) == "promoted"

    assert source.read_text(encoding="utf-8") == "#!/bin/sh\necho new\n"
    assert stat.S_IMODE(source.stat().st_mode) == 0o755, "the exec bit was stripped by promote()"


# ===========================================================================
# P1-8 -- hardlinker --prune must honour the ignore list, hidden dirs and .git
# ===========================================================================


def _prune_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "a.txt").write_text("a\n", encoding="utf-8")
    (source / "keep.txt").write_text("k\n", encoding="utf-8")
    (dest / "pkg").mkdir(parents=True)
    (dest / "pkg" / "a.txt").write_text("old\n", encoding="utf-8")
    (dest / "stale.txt").write_text("stale\n", encoding="utf-8")
    (dest / ".git").mkdir()
    (dest / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (dest / ".venv").mkdir()
    (dest / ".venv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (dest / "debug.log").write_text("noise\n", encoding="utf-8")
    config = tmp_path / "hardlink_config.json"
    config.write_text(
        json.dumps(
            {
                "source": str(source),
                "dest": str(dest),
                "ignore": ["*.log"],
                "dry_run": True,
                "base_ignores": False,
            }
        ),
        encoding="utf-8",
    )
    return source, dest, config


def _run_hardlinker(monkeypatch: pytest.MonkeyPatch, config: Path, extra: list[str]) -> None:
    hardlinker = load_module("hardlinker_under_test", "hardlinker.py")
    argv = ["hardlinker.py", "-c", str(config), "--run", "--prune", "--force", "--no-charts", "--no-secret-scan"]
    monkeypatch.setattr(sys, "argv", argv + extra)
    hardlinker.main()


def test_prune_force_keeps_git_ignored_and_hidden_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-8: --run --prune --force deleted dest/.git, dest/.venv and ignored files."""
    _source, dest, config = _prune_tree(tmp_path)

    _run_hardlinker(monkeypatch, config, [])

    assert not (dest / "stale.txt").exists(), "a genuine orphan must still be pruned"
    assert (dest / ".git" / "HEAD").is_file(), "git history was deleted by --prune --force"
    assert (dest / ".venv" / "pyvenv.cfg").is_file(), "a hidden tree was deleted by --prune --force"
    assert (dest / "debug.log").is_file(), "an ignore-listed file was deleted by --prune --force"
    assert (dest / "pkg" / "a.txt").read_text(encoding="utf-8") == "a\n"


def test_prune_refuses_git_even_when_hidden_dirs_are_walked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-8 (defence in depth): a .git path is refused even with --no-hidden-skip."""
    _source, dest, config = _prune_tree(tmp_path)

    _run_hardlinker(monkeypatch, config, ["--no-hidden-skip"])

    assert (dest / ".git" / "HEAD").is_file(), "the .git hard refusal is gone"


# ===========================================================================
# P2-15 -- configure-oidc-rag.sh: CA verification + password on stdin
# ===========================================================================

_CURL_STUB = """#!/usr/bin/env python3
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["W4_CURL_ARGV_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(argv) + "\\n")
stdin = sys.stdin.buffer.read()
with open(os.environ["W4_CURL_STDIN_LOG"], "ab") as fh:
    fh.write(stdin + b"\\n----\\n")
url = next((a for a in argv if a.startswith("http")), "")
if url.endswith("/token"):
    print(json.dumps({"access_token": "ADMIN-TOKEN"}))
elif "clientId=" in url:
    print(json.dumps([{"id": "CLIENT-ID"}]))
elif "/clients/" in url:
    print(json.dumps({"secret": "CLIENT-SECRET", "redirectUris": []}))
else:
    print("")
"""

_KUBECTL_STUB = """#!/usr/bin/env python3
import sys

args = " ".join(sys.argv[1:])
if "ezapp-manager-parameters" in args:
    print("example.test")
elif "admin-pass" in args:
    print("S3CRET-ADMIN-PW")
"""


def _oidc_stub_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("curl", _CURL_STUB), ("kubectl", _KUBECTL_STUB)):
        stub = bindir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    argv_log = tmp_path / "curl-argv.jsonl"
    stdin_log = tmp_path / "curl-stdin.bin"
    argv_log.write_text("", encoding="utf-8")
    stdin_log.write_bytes(b"")
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HOME": str(home),
        "W4_CURL_ARGV_LOG": str(argv_log),
        "W4_CURL_STDIN_LOG": str(stdin_log),
    }
    return env, argv_log


def _run_oidc_script(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "configure-oidc-rag.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=120,
    )


def test_oidc_script_keeps_the_admin_password_out_of_curl_argv(tmp_path: Path) -> None:
    """P2-15: the Keycloak admin password transited in curl's argv (/proc-readable)."""
    env, argv_log = _oidc_stub_env(tmp_path)
    ca = tmp_path / "pcai-aie-root-ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    env["RAG_CA_BUNDLE"] = str(ca)

    proc = _run_oidc_script(tmp_path, env)

    assert proc.returncode == 0, proc.stderr
    invocations = [json.loads(line) for line in argv_log.read_text(encoding="utf-8").splitlines()]
    assert invocations, "the script made no curl calls"
    leaked = [argv for argv in invocations if any("S3CRET-ADMIN-PW" in arg for arg in argv)]
    assert not leaked, "the admin password leaked into curl's argv"
    stdin_payload = (tmp_path / "curl-stdin.bin").read_text(encoding="utf-8", errors="replace")
    assert "password=S3CRET-ADMIN-PW" in stdin_payload, "the password was not fed on stdin"


def test_oidc_script_verifies_tls_with_the_platform_ca(tmp_path: Path) -> None:
    """P2-15: every credential-carrying call must verify TLS when a bundle exists."""
    env, argv_log = _oidc_stub_env(tmp_path)
    ca = tmp_path / "pcai-aie-root-ca.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    env["RAG_CA_BUNDLE"] = str(ca)

    proc = _run_oidc_script(tmp_path, env)

    assert proc.returncode == 0, proc.stderr
    invocations = [json.loads(line) for line in argv_log.read_text(encoding="utf-8").splitlines()]
    assert invocations
    for argv in invocations:
        assert "-k" not in argv, f"curl ran with verification disabled: {argv}"
        assert "--cacert" in argv and str(ca) in argv, f"curl did not use the CA bundle: {argv}"


def test_oidc_script_warns_loudly_when_no_ca_bundle_exists(tmp_path: Path) -> None:
    """P2-15: the -k fallback must be loud, never silent."""
    env, argv_log = _oidc_stub_env(tmp_path)

    proc = _run_oidc_script(tmp_path, env)

    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stderr and "DISABLED" in proc.stderr
    invocations = [json.loads(line) for line in argv_log.read_text(encoding="utf-8").splitlines()]
    assert invocations and all("-k" in argv for argv in invocations)


# ===========================================================================
# P2-16 -- .dockerignore must exclude the whole non-build context
# ===========================================================================


def _dockerignored(rel_path: str, patterns: set[str]) -> bool:
    """Emulate Docker's .dockerignore match for the pattern shapes used here."""
    parts = rel_path.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    for pattern in patterns:
        for prefix in prefixes:
            if fnmatch.fnmatch(prefix, pattern) or fnmatch.fnmatch(prefix.rsplit("/", 1)[-1], pattern):
                return True
    return False


def test_dockerignore_excludes_the_whole_build_context() -> None:
    """P2-16: a one-line .dockerignore shipped .git, tests and helm*/local secrets."""
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    patterns = {line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}

    for required in (
        ".git",
        ".scratch",
        ".venv",
        "**/__pycache__",
        "tests",
        "documentation",
        "helm*/local",
        "uploads_for_test_dataset",
        "qdrant_storage",
    ):
        assert required in patterns, f".dockerignore lost the {required!r} exclusion"

    for excluded in (
        ".git/config",
        ".scratch/audit-2026-11-10/dirty-work-snapshot.tgz",
        ".venv/lib/python3.14/site-packages/x.py",
        "tests/full_pipeline/test_hybrid_search.py",
        "documentation/AUDIT-2026-11-10.md",
        "helm-scale-large/local/secrets.g2.yaml",
        "uploads_for_test_dataset/big.bin",
        "qdrant_storage/raft/wal.bin",
        "src/multimodal_rag/utils/__pycache__/bm25.cpython-314.pyc",
    ):
        assert _dockerignored(excluded, patterns), f"{excluded} still reaches the builder"

    # docker/Dockerfile only COPYs src/ and docker/requirements.txt: never exclude them.
    for needed in (
        "src/multimodal_rag/api_server.py",
        "src/multimodal_rag/utils/bm25.py",
        "docker/requirements.txt",
        "docker/Dockerfile",
    ):
        assert not _dockerignored(needed, patterns), f"{needed} must stay in the build context"


# ===========================================================================
# P3-6 -- media-proxy: bounded caches + loopback-only posture
# ===========================================================================


def test_media_proxy_memory_cache_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-6: the RAM cache was an unbounded dict (a long session grew forever)."""
    proxy = load_module("media_proxy_under_test", "scripts/media-proxy.py")
    monkeypatch.setattr(proxy, "MEMORY_CACHE_MAX_ITEMS", 3)
    monkeypatch.setattr(proxy, "MEMORY_CACHE_MAX_BYTES", 64)
    store = proxy.MediaStore(gateway=object(), cache_dir=tmp_path)

    for index in range(10):
        store._memory_put(f"k{index}", (b"x" * 16, "text/plain"))

    assert len(store._memory) <= 3, "the memory cache is not item-capped"
    assert store._memory_bytes <= 64, "the memory cache is not byte-capped"
    assert "k9" in store._memory and "k0" not in store._memory, "the LRU did not evict oldest-first"


def test_media_proxy_prunes_the_disk_cache_at_startup(tmp_path: Path) -> None:
    """P3-6: the disk cache was never pruned."""
    proxy = load_module("media_proxy_prune_under_test", "scripts/media-proxy.py")
    for index in range(5):
        path = tmp_path / f"f{index}.bin"
        path.write_bytes(b"x" * 100)
        os.utime(path, (1000 + index, 1000 + index))

    removed = proxy.prune_cache_dir(tmp_path, 250)

    assert removed == 3
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 250
    assert not (tmp_path / "f0.bin").exists() and not (tmp_path / "f2.bin").exists()
    assert (tmp_path / "f4.bin").exists(), "the newest entry must survive"


def test_media_proxy_refuses_a_non_loopback_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-6: --host 0.0.0.0 silently defeated the documented loopback posture."""
    proxy = load_module("media_proxy_host_under_test", "scripts/media-proxy.py")
    assert proxy.is_loopback_host("127.0.0.1")
    assert proxy.is_loopback_host("localhost")
    assert proxy.is_loopback_host("::1")
    assert not proxy.is_loopback_host("0.0.0.0")
    assert not proxy.is_loopback_host("10.0.0.5")

    monkeypatch.setenv("PCAI_LLM_API_KEY", "dummy-key")
    assert proxy.main(["--host", "0.0.0.0", "--port", "0", "--cache-dir", str(tmp_path / "cache")]) == 2


# ===========================================================================
# P3-10 -- filter.py: one shared client + a visible memory-recall failure
# ===========================================================================


class _FakeAsyncClient:
    """Stands in for httpx2.AsyncClient (never touches the network)."""

    is_closed = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.raise_on_request = False

    async def get(self, url: str, **kwargs: Any) -> Any:
        return self._respond(url, kwargs)

    async def post(self, url: str, **kwargs: Any) -> Any:
        return self._respond(url, kwargs)

    def _respond(self, url: str, kwargs: dict[str, Any]) -> Any:
        self.requests.append((url, kwargs))
        if self.raise_on_request:
            raise RuntimeError("RAG API unreachable")
        return object()

    async def aclose(self) -> None:
        self.is_closed = True


def _fresh_filter(monkeypatch: pytest.MonkeyPatch, client: Any) -> Any:
    module = load_module("openwebui_filter_under_test", "openwebui_extension/filter.py")
    monkeypatch.setattr(module, "_HTTP_CLIENT", None)
    monkeypatch.setattr(module, "_HTTP_CLIENT_LOOP", None)
    monkeypatch.setattr(module, "_MEMORY_RECALL_UNAVAILABLE_SIGNALLED", False)
    monkeypatch.setattr(module.httpx2, "AsyncClient", lambda *args, **kwargs: client)
    return module


def test_filter_reuses_one_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-10: a fresh AsyncClient per call paid a new TCP(+TLS) handshake each turn."""
    client = _FakeAsyncClient()
    module = _fresh_filter(monkeypatch, client)

    first = module._shared_http_client()
    second = module._shared_http_client()

    assert first is second, "the filter does not reuse one client"
    assert first is client


def test_filter_rebuilds_the_client_when_the_event_loop_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """W2 review: a pool bound to loop A must be rebuilt (not raise) on loop B."""
    module = load_module("openwebui_filter_loop_under_test", "openwebui_extension/filter.py")
    monkeypatch.setattr(module, "_HTTP_CLIENT", None)
    monkeypatch.setattr(module, "_HTTP_CLIENT_LOOP", None)
    created: list[Any] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.is_closed = False
            created.append(self)

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr(module.httpx2, "AsyncClient", FakeClient)

    async def grab() -> Any:
        return module._shared_http_client()

    first_loop = asyncio.new_event_loop()
    second_loop = asyncio.new_event_loop()
    try:
        client_a = first_loop.run_until_complete(grab())
        client_b = second_loop.run_until_complete(grab())
        client_c = second_loop.run_until_complete(grab())
    finally:
        first_loop.close()
        second_loop.close()

    assert client_a is not client_b, "the client was not rebuilt for a second event loop"
    assert client_b is client_c, "the client was needlessly rebuilt on the same loop"
    assert len(created) == 2


def test_filter_shared_client_applies_each_calls_own_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-10: per-call timeout/follow_redirects must survive the shared pool."""
    client = _FakeAsyncClient()
    module = _fresh_filter(monkeypatch, client)

    async def run() -> None:
        async with module._shared_client(timeout=15.0) as handle:
            await handle.get("http://rag/api/datasets")
        async with module._shared_client(timeout=120.0, follow_redirects=True) as handle:
            await handle.post("http://rag/api/staging", files={})

    asyncio.run(run())

    assert client.requests[0][1]["timeout"] == 15.0
    assert "follow_redirects" not in client.requests[0][1]
    assert client.requests[1][1]["timeout"] == 120.0
    assert client.requests[1][1]["follow_redirects"] is True


def test_filter_signals_memory_recall_failure_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-10: a dead memory store was indistinguishable from an empty one."""
    client = _FakeAsyncClient()
    client.raise_on_request = True
    module = _fresh_filter(monkeypatch, client)
    events: list[dict[str, Any]] = []

    async def emitter(event_type: str | None = None, data: dict[str, Any] | None = None) -> None:
        events.append({"event_type": event_type, "data": data or {}})

    chat_filter = module.Filter()
    chat_filter.valves.MEMORY_ENABLED = True
    chat_filter.valves.MEMORY_DATASET_PREFIX = "mem-"

    async def run() -> None:
        for text in ("first question", "second question"):
            body = {"messages": [{"role": "user", "content": text}]}
            await chat_filter.inlet(body, __user__={"id": "u1"}, __event_emitter__=emitter)

    asyncio.run(run())

    signals = [
        event for event in events if "memory recall unavailable" in str(event["data"].get("description", "")).lower()
    ]
    assert len(signals) == 1, "the memory-recall failure was not signalled exactly once"
    assert signals[0]["data"]["status"] == "error"


# ===========================================================================
# P3-11 -- deployment_flow.dot must describe the current MCP surface
# ===========================================================================


def test_deployment_flow_dot_lists_the_current_mcp_surface() -> None:
    """P3-11: the diagram advertised "9 tools" and two retired tool names."""
    text = (REPO_ROOT / "documentation" / "deployment_flow.dot").read_text(encoding="utf-8")
    mcp_lines = [line for line in text.splitlines() if "MCP [label=" in line]
    assert len(mcp_lines) == 1
    label = mcp_lines[0]

    assert "25 tools" in label, f"stale tool count in the diagram: {label}"
    assert "inspect_media" in label
    assert "9 tools" not in label
    assert "describe_media" not in label and "transcribe_audio" not in label


# ===========================================================================
# P3-12 -- migrate-my-memory: poll to a terminal status with an overall cap
# ===========================================================================


class _FakeClock:
    """Stand-in for the time module inside poll_import_status."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds


def _load_poll_helper(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Exec only the poll helpers: importing the script would prompt for secrets."""
    source = (REPO_ROOT / "scripts" / "migrate-my-memory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_functions = {"poll_import_status", "_env_float"}
    wanted_assignments = {"TERMINAL_OK", "TERMINAL_BAD"}
    body: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted_functions)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") in wanted_assignments for target in node.targets)
        )
    ]
    assert len(body) == 4, f"expected the 2 helpers + 2 terminal tuples, found {len(body)}"
    clock = _FakeClock()
    namespace: dict[str, Any] = {"time": clock, "os": os}
    # The script cannot be imported (it prompts for secrets at module level), so only
    # the poll helpers are compiled and executed here: no third-party code, no I/O.
    exec(compile(ast.Module(body=body, type_ignores=[]), "<migrate-my-memory>", "exec"), namespace)  # noqa: S102
    namespace["_clock"] = clock
    return namespace


def test_migrate_polls_past_the_old_ten_minute_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-12: a fixed 60x10s poll declared failure while the import kept running."""
    monkeypatch.delenv("RAG_IMPORT_MAX_WAIT", raising=False)
    monkeypatch.delenv("RAG_IMPORT_POLL_INTERVAL", raising=False)
    namespace = _load_poll_helper(monkeypatch)
    calls = {"count": 0}

    def fetch_status() -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] < 66:
            return {"status": "running", "processed_files": calls["count"], "total_files": 100}
        return {"status": "completed", "result": {"documents": 100}}

    final = namespace["poll_import_status"]("job-1", fetch_status, total_files=100)

    assert final is not None, "the poll gave up while the import was still running"
    assert final["status"] == "completed"
    assert calls["count"] == 66, "the poll stopped at the old fixed window"


def test_migrate_poll_stops_at_the_overall_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-12: polling must still terminate (an overall cap, not forever)."""
    monkeypatch.setenv("RAG_IMPORT_MAX_WAIT", "60")
    monkeypatch.setenv("RAG_IMPORT_POLL_INTERVAL", "10")
    namespace = _load_poll_helper(monkeypatch)

    final = namespace["poll_import_status"]("job-2", lambda: {"status": "running"}, total_files=0)

    assert final is None, "an unterminated job must end at the cap"
    assert 0 < namespace["_clock"].sleeps <= 10


def test_migrate_poll_returns_a_terminal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-12: a failed job is terminal — return it instead of polling on."""
    monkeypatch.delenv("RAG_IMPORT_MAX_WAIT", raising=False)
    namespace = _load_poll_helper(monkeypatch)
    calls = {"count": 0}

    def fetch_status() -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "running"}
        return {"status": "failed", "error": "embedder 503"}

    final = namespace["poll_import_status"]("job-3", fetch_status, total_files=1)

    assert final is not None and final["status"] == "failed"
    assert final["error"] == "embedder 503"
    assert calls["count"] == 2


def test_migrate_poll_survives_a_transient_poll_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3-12: a transient poll failure must not end the wait."""
    monkeypatch.delenv("RAG_IMPORT_MAX_WAIT", raising=False)
    namespace = _load_poll_helper(monkeypatch)
    calls = {"count": 0}

    def fetch_status() -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("socket timeout")
        return {"status": "completed"}

    final = namespace["poll_import_status"]("job-4", fetch_status, total_files=1)

    assert final is not None and final["status"] == "completed"
