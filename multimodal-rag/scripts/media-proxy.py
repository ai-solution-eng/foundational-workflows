#!/usr/bin/env python3
"""rag-media-proxy -- serve RAG dataset media to a browser that cannot reach it.

WHY THIS EXISTS
---------------
MEDIA_BASE_URL points at the PCAI application host, and that host sits behind
the platform's oauth2-proxy SSO gate (a CUSTOM AuthorizationPolicy on the
istio-ingressgateway).  EVERY request -- including the HMAC-signed
/api/datasets/{ds}/files/{f}?token=... URLs the MCP server mints precisely so
<img> tags can fetch protected media -- is answered with a 302 to Keycloak
before FastAPI ever sees it.  An <img src> therefore renders as a broken
image, and the app-level media token is never even evaluated.

A live SSO session does not fix it either: the gateway cookie is
SameSite=Lax, so a cross-site subresource load (an image fetched by a page on
a different origin, e.g. a local GUI on 127.0.0.1:3080) is sent WITHOUT the
cookie and is 302'd again.

This proxy closes the gap along the path the app's own design already
supports: it fetches bytes through the MCP gateway, which authenticates with
a BEARER KEY rather than a cookie, and re-serves them from a localhost origin
the browser can load without any SSO involvement.

USAGE
-----
    python3 scripts/media-proxy.py --port 8788

Then swap the origin of any RAG media URL:

    https://rag-mcp-server.../api/datasets/andrew-personal-dataset/files/DSC01378_preprocessed.JPG
 -> http://127.0.0.1:8788/api/datasets/andrew-personal-dataset/files/DSC01378_preprocessed.JPG

Both the uuid-prefixed stored name and the short human name resolve, and a
?token= suffix is accepted and ignored (the proxy re-authorises through the
gateway key, so a stale or absent token is irrelevant).

SECURITY POSTURE
----------------
Binds 127.0.0.1 only.  It reads exactly what the configured gateway key may
already read -- it adds no authority of its own -- and it exposes no
filesystem path: every byte is fetched by (dataset, filename) through the
gateway.  It is a convenience shim for a human browser, not a service to
deploy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

DEFAULT_GATEWAY = "https://pcai-llm-gateway.pcai-se-ai-application.hst.rdlabs.hpecorp.net/mcp"
CA_BUNDLES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt")
UUID_PREFIX = re.compile(r"^[0-9a-f]{32}_")


def _load_key() -> str:
    """The gateway bearer key: environment first, then the DSH env file."""
    key = os.environ.get("PCAI_LLM_API_KEY", "").strip()
    if key:
        return key
    try:
        text = (Path.home() / ".dsh" / ".env").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("PCAI_LLM_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


class GatewayMCP:
    """Minimal MCP JSON-RPC client for the PCAI gateway (streamable HTTP).

    Deliberately stdlib-only: the deployment certs are signed by an internal
    CA that only the system bundle carries, and a third-party HTTP stack
    would need that trust wired in separately.
    """

    def __init__(self, url: str, key: str, timeout: float = 180.0) -> None:
        self.url = url
        self.key = key
        self.timeout = timeout
        self.session = None
        self._lock = threading.Lock()
        self._seq = 0
        ctx = ssl.create_default_context()
        for bundle in CA_BUNDLES:
            if Path(bundle).is_file():
                ctx = ssl.create_default_context(cafile=bundle)
                break
        self._ctx = ctx
        self._ready = False

    def _post(self, payload: dict) -> str:
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), method="POST"
        )
        request.add_header("Authorization", "Bearer " + self.key)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json, text/event-stream")
        if self.session:
            request.add_header("mcp-session-id", self.session)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self._ctx) as resp:
                session = resp.headers.get("mcp-session-id")
                if session:
                    self.session = session
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"gateway HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _decode(body: str) -> dict:
        """Return the JSON-RPC object from a plain-JSON or SSE response body."""
        found = None
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line:
                continue
            try:
                found = json.loads(line)
            except json.JSONDecodeError:
                continue
        if found is None:
            try:
                found = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"unparseable gateway body: {body[:200]}") from exc
        return found

    def _rpc(self, method: str, params: dict) -> dict:
        with self._lock:
            self._seq += 1
            seq = self._seq
        body = self._post({"jsonrpc": "2.0", "id": seq, "method": method, "params": params})
        obj = self._decode(body)
        if "error" in obj:
            raise RuntimeError(f"gateway error: {obj['error']}")
        return obj.get("result", {})

    def start(self) -> None:
        if self._ready:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "rag-media-proxy", "version": "0.1"},
            },
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._ready = True

    def call(self, tool: str, arguments: dict) -> dict:
        self.start()
        result = self._rpc("tools/call", {"name": tool, "arguments": arguments})
        blocks = result.get("content") or []
        text = blocks[0].get("text", "") if blocks else ""
        try:
            outer = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{tool}: non-JSON result: {text[:200]}") from exc
        if isinstance(outer, dict) and "content" in outer:
            inner = outer["content"]
            if inner and isinstance(inner[0], dict) and "text" in inner[0]:
                try:
                    return json.loads(inner[0]["text"])
                except json.JSONDecodeError:
                    pass
            return outer
        return outer if isinstance(outer, dict) else {"value": outer}


class MediaTooLarge(Exception):
    """The file exceeds the gateway's MCP_FILE_BYTES_MAX tool cap.

    Raised instead of FileNotFoundError so the HTTP layer can answer 413 and
    the operator learns the real remedy (raise MCP_FILE_BYTES_MAX, or serve
    bulk media over REST) rather than hunting for a file that does exist.
    """


class MediaStore:
    """Resolve (dataset, filename) to bytes, with a disk cache in front."""

    def __init__(self, gateway: GatewayMCP, cache_dir: Path) -> None:
        self.gateway = gateway
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._index_lock = threading.Lock()
        self._memory = {}

    def _dataset_index(self, dataset: str) -> dict:
        """Map every known name for a dataset files to the stored name."""
        with self._index_lock:
            cached = self._index.get(dataset)
        if cached is not None:
            return cached
        listing = self.gateway.call("mm-rag__get_dataset_files", {"dataset_name": dataset})
        index = {}
        for stored in self._iter_files(listing):
            index[stored] = stored
            index.setdefault(UUID_PREFIX.sub("", stored), stored)
        with self._index_lock:
            self._index[dataset] = index
        return index

    @staticmethod
    def _iter_files(listing: dict):
        """Yield stored filenames from a get_dataset_files listing payload."""
        candidates = listing.get("files") or listing.get("documents") or []
        if isinstance(candidates, dict):
            candidates = list(candidates)
        for item in candidates:
            if isinstance(item, str):
                yield item.rsplit("/", 1)[-1]
            elif isinstance(item, dict):
                name = item.get("file") or item.get("name") or item.get("path")
                if isinstance(name, str):
                    yield name.rsplit("/", 1)[-1]

    def resolve(self, dataset: str, filename: str) -> str:
        index = self._dataset_index(dataset)
        if filename in index:
            return index[filename]
        with self._index_lock:
            self._index.pop(dataset, None)
        return self._dataset_index(dataset).get(filename, filename)

    def fetch(self, dataset: str, filename: str):
        stored = self.resolve(dataset, filename)
        key = dataset + "/" + stored
        hit = self._memory.get(key)
        if hit is not None:
            return hit
        cached = self._cache_path(dataset, stored)
        if cached.is_file():
            meta = cached.with_suffix(cached.suffix + ".meta")
            mime = "application/octet-stream"
            if meta.is_file():
                try:
                    mime = json.loads(meta.read_text(encoding="utf-8")).get("mime") or mime
                except (OSError, json.JSONDecodeError):
                    pass
            payload = (cached.read_bytes(), mime)
            self._memory[key] = payload
            return payload

        result = self.gateway.call(
            "mm-rag__get_dataset_file_bytes",
            {"dataset_name": dataset, "file_path": stored},
        )
        if result.get("ok") is False or "data" not in result:
            error = result.get("error") or (key + " unavailable")
            # get_dataset_file_bytes refuses anything over MCP_FILE_BYTES_MAX
            # (256 KiB by default).  That is a transport limit, not a
            # permission problem, and the browser deserves a status that says
            # so: 413 rather than a misleading 404.
            if "tool cap" in error or "MCP_FILE_BYTES_MAX" in error:
                raise MediaTooLarge(error)
            raise FileNotFoundError(error)
        raw = base64.b64decode(result["data"])
        mime = (
            result.get("mime_type")
            or mimetypes.guess_type(stored)[0]
            or "application/octet-stream"
        )
        try:
            cached.write_bytes(raw)
            cached.with_suffix(cached.suffix + ".meta").write_text(
                json.dumps({"mime": mime, "size": len(raw)}), encoding="utf-8"
            )
        except OSError:
            pass
        self._memory[key] = (raw, mime)
        return raw, mime

    def _cache_path(self, dataset: str, stored: str) -> Path:
        digest = hashlib.sha256((dataset + "/" + stored).encode()).hexdigest()[:32]
        return self.cache_dir / (digest + (Path(stored).suffix or ".bin"))


def make_handler(store: MediaStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rag-media-proxy/0.1"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

        def _target(self):
            """Split the request path into (dataset, filename)."""
            path = unquote(urlsplit(self.path).path)
            parts = [p for p in path.split("/") if p]
            if parts and parts[0] == "api" and len(parts) >= 5 and parts[1] == "datasets":
                if parts[3] == "files":
                    return parts[2], "/".join(parts[4:])
                return None
            if len(parts) >= 2:
                return parts[0], "/".join(parts[1:])
            return None

        def _respond(self, status, body, mime, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/healthz", "/health"):
                self._respond(200, b'{"ok":true}', "application/json")
                return
            target = self._target()
            if target is None:
                self._respond(
                    404,
                    b"usage: /api/datasets/<dataset>/files/<file>  or  /<dataset>/<file>",
                    "text/plain; charset=utf-8",
                )
                return
            dataset, filename = target
            try:
                body, mime = store.fetch(dataset, filename)
            except MediaTooLarge as exc:
                self._respond(413, str(exc).encode(), "text/plain; charset=utf-8")
                return
            except FileNotFoundError as exc:
                self._respond(404, str(exc).encode(), "text/plain; charset=utf-8")
                return
            except Exception as exc:
                msg = ("upstream: " + str(exc)).encode()
                self._respond(502, msg, "text/plain; charset=utf-8")
                return
            self._respond(200, body, mime, {"X-Content-Type-Options": "nosniff"})

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve RAG dataset media over localhost.")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--gateway", default=os.environ.get("RAG_MEDIA_PROXY_GATEWAY", DEFAULT_GATEWAY)
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "RAG_MEDIA_PROXY_CACHE", str(Path.home() / ".dsh" / "tmp-rag-media-proxy")
        ),
    )
    args = parser.parse_args(argv)

    key = _load_key()
    if not key:
        print(
            "error: no gateway key (set PCAI_LLM_API_KEY or add it to ~/.dsh/.env)",
            file=sys.stderr,
        )
        return 2

    cache_dir = Path(args.cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".writable"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        fallback = Path(__file__).resolve().parent.parent / ".scratch" / "rag-media-proxy-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        cache_dir = fallback

    store = MediaStore(GatewayMCP(args.gateway, key), cache_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    base = "http://" + args.host + ":" + str(args.port)
    print("rag-media-proxy listening on " + base)
    print("  gateway: " + args.gateway)
    print("  cache:   " + str(cache_dir))
    print("  example: " + base + "/api/datasets/<dataset>/files/<file>")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())