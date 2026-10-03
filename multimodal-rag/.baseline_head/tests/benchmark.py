#!/usr/bin/env python3
"""Benchmark the Multimodal RAG platform with N concurrent users.

Sends random search queries in parallel from N simulated users and
measures response rate (req/s), latency percentiles, and error rates.

Two modes are supported:

* **rest** (default) — calls the REST API ``GET /api/datasets/{name}/search``
  on the API server (port 8000).
* **mcp** — calls the ``search_dataset`` MCP tool over the streamable-http
  transport (``/mcp`` endpoint, port 9090).  This is the path that LLM
  clients like opencode use.

Usage examples
--------------
    # 50 users, 30s, REST mode, no reranker
    python tests/benchmark.py --url http://localhost:8000 --dataset my-ds --N 50

    # 200 users, 60s, MCP mode, with reranker
    python tests/benchmark.py --mode mcp --url http://localhost:9090/mcp \\
        --api-url http://localhost:8000 --dataset my-ds \\
        --N 200 --duration 60 --use-reranker --reranker-top-k 5

    # Auto-discover datasets, use custom queries
    python tests/benchmark.py --url http://localhost:8000 --N 100 \\
        --queries-file my_queries.txt

Reranker handling
-----------------
The reranker is enabled by passing ``use_reranker=true`` to the API.
When enabled, the server reranks the top_k embedding results with the
configured cross-encoder and returns only ``reranker_top_k`` of them.
When disabled (default), raw embedding-similarity results are returned
and ``reranker_top_k`` is ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx
import httpx2

# ---------------------------------------------------------------------------
# Default query pool — diverse, sensible queries that work across datasets
# ---------------------------------------------------------------------------

# Query pool for the simulated users, split by how the SERVER treats them:
#
#   * generic queries are answered from stored content — when a doc carries
#     an ingest-time VLM caption ("[Image description]: …"), the server
#     REUSES it and never calls the VLM at query time;
#   * VLM-specific queries (spatial / count / color / comparison wording)
#     trip the server's _query_needs_vlm() heuristic: for a TEXT-ONLY caller
#     (the default) they re-run the VLM once per image hit — realistic, but
#     far slower; vision-capable callers (--base-llm-modalities text,image)
#     skip the VLM entirely.
#
# DEFAULT_QUERIES (mixed) is kept for backward compatibility.
DEFAULT_QUERIES_GENERIC: list[str] = [
    # General knowledge
    "What is machine learning?",
    "Explain how neural networks work",
    "What are the benefits of cloud computing?",
    "What is a transformer architecture?",
    "How does retrieval-augmented generation work?",
    "What is vector similarity search?",
    "Explain the concept of embeddings",
    "What is transfer learning?",
    "Describe the attention mechanism",
    # Technical / code
    "How do I handle errors in Python?",
    "What is a REST API?",
    "Show me examples of data structures",
    "How to optimize database queries",
    "What is containerization?",
    "Explain microservices architecture",
    "How does authentication work?",
    "What are design patterns?",
    # Document / business
    "What are the project requirements?",
    "Summarize the main findings",
    "What are the key metrics?",
    "Describe the deployment process",
    "What are the security policies?",
    "Find information about configuration",
    "What are the best practices?",
    # Short keyword
    "introduction",
    "overview",
    "summary",
    "architecture",
    "performance",
    "security",
    "configuration",
    "installation",
    "troubleshooting",
    "examples",
    # Image / media (for multimodal datasets)
    "The aurora borealis over a snowy mountain",
    "Black and white image of a lake reflecting the trees by its side",
    "A man crouching staring down at the tops of clouds from a mountain",
]
DEFAULT_QUERIES_VLM: list[str] = [
    # From the historical default pool — spatial / comparison wording
    "Describe the difference between supervised and unsupervised learning",
    "A skyscraper high above the other buildings in a city on a cloudy day",
    "The top of a tower with an antenna on an overcast day",
    # Canonical visual-detail questions (count / color / text-in-image)
    "How many products are shown in the image?",
    "What color is the box in the middle of the picture?",
    "What does the label on the packaging say?",
]
DEFAULT_QUERIES: list[str] = DEFAULT_QUERIES_GENERIC + DEFAULT_QUERIES_VLM


# ---------------------------------------------------------------------------
# Data classes for metrics
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkStats:
    """Collected metrics across all users."""

    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    result_counts: list[int] = field(default_factory=list)
    status_codes: dict[int, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def response_rate(self) -> float:
        return self.total / self.duration if self.duration > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return (self.success / self.total * 100) if self.total > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        return statistics.quantiles(self.latencies, n=100, method="inclusive")[int(p) - 1]

    def summary(self) -> dict:
        lat = self.latencies
        return {
            "total_requests": self.total,
            "successful": self.success,
            "failed": self.failed,
            "success_rate_pct": round(self.success_rate, 2),
            "duration_s": round(self.duration, 2),
            "response_rate_rps": round(self.response_rate, 2),
            "latency_ms": {
                "min": round(min(lat) * 1000, 2) if lat else 0,
                "mean": round(statistics.mean(lat) * 1000, 2) if lat else 0,
                "median": round(statistics.median(lat) * 1000, 2) if lat else 0,
                "p95": round(self.percentile(95) * 1000, 2) if lat else 0,
                "p99": round(self.percentile(99) * 1000, 2) if lat else 0,
                "max": round(max(lat) * 1000, 2) if lat else 0,
            },
            "avg_results_per_query": round(statistics.mean(self.result_counts), 2) if self.result_counts else 0,
            "status_codes": dict(sorted(self.status_codes.items())),
            "errors": dict(sorted(self.errors.items(), key=lambda x: -x[1])),
        }


# ---------------------------------------------------------------------------
# Single user session
# ---------------------------------------------------------------------------


async def run_user(
    user_id: int,
    client: httpx.AsyncClient,
    base_url: str,
    dataset: str,
    queries: list[str],
    top_k: int,
    use_reranker: bool,
    reranker_top_k: int,
    password: str | None,
    duration: float,
    ramp_delay: float,
    stats: BenchmarkStats,
    api_key: str | None = None,
) -> None:
    """Simulate one user sending queries for *duration* seconds."""
    rng = random.Random(user_id)
    headers: dict[str, str] = {}
    if password:
        headers["X-Dataset-Password"] = password
    if api_key:
        headers["X-RAG-Api-Key"] = api_key

    # Ramp-up: stagger start times
    if ramp_delay > 0:
        await asyncio.sleep(ramp_delay)

    end_time = time.monotonic() + duration
    search_url = f"{base_url}/api/datasets/{dataset}/search"
    params_base = {
        "top_k": top_k,
        "use_reranker": "true" if use_reranker else "false",
        "reranker_top_k": reranker_top_k,
    }

    while time.monotonic() < end_time:
        query = rng.choice(queries)
        params = {**params_base, "q": query}
        t0 = time.monotonic()
        try:
            resp = await client.get(search_url, params=params, headers=headers, timeout=120.0)
            elapsed = time.monotonic() - t0
            stats.total += 1
            stats.latencies.append(elapsed)
            stats.status_codes[resp.status_code] = stats.status_codes.get(resp.status_code, 0) + 1
            if resp.status_code == 200:
                stats.success += 1
                data = resp.json()
                stats.result_counts.append(len(data.get("results", [])))
            else:
                stats.failed += 1
                err_key = f"HTTP {resp.status_code}"
                stats.errors[err_key] = stats.errors.get(err_key, 0) + 1
        except Exception as exc:
            elapsed = time.monotonic() - t0
            stats.total += 1
            stats.failed += 1
            stats.latencies.append(elapsed)
            err_key = type(exc).__name__
            stats.errors[err_key] = stats.errors.get(err_key, 0) + 1


# ---------------------------------------------------------------------------
# MCP user session
# ---------------------------------------------------------------------------


async def run_user_mcp(
    user_id: int,
    mcp_url: str,
    dataset: str,
    queries: list[str],
    top_k: int,
    use_reranker: bool,
    reranker_top_k: int,
    password: str | None,
    duration: float,
    ramp_delay: float,
    stats: BenchmarkStats,
    base_llm_modalities: list[str] | None = None,
    insecure: bool = False,
    call_timeout: float = 120.0,
) -> None:
    """Simulate one user sending MCP tool calls for *duration* seconds.

    Each user creates its own MCP session (via the streamable-http client),
    initialises it, then calls ``search_dataset`` in a loop.  The session
    is kept alive for the entire duration — mirroring how a real LLM client
    maintains a persistent connection.

    *call_timeout* caps how long a single ``call_tool`` may block.  Without
    this, a queued request can hang for up to the MCP client's SSE read
    timeout (300 s), which makes high-concurrency runs run far past
    *duration* while ``asyncio.gather`` waits for stragglers.  Timed-out
    calls are recorded as failures.
    """
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    rng = random.Random(user_id)

    if ramp_delay > 0:
        await asyncio.sleep(ramp_delay)

    end_time = time.monotonic() + duration
    tool_args_base: dict = {
        "dataset_name": dataset,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "reranker_top_k": reranker_top_k,
    }
    if password:
        tool_args_base["password"] = password
    if base_llm_modalities:
        # Declare the simulated client's LLM modalities — "text" only (the
        # default server-side) makes every image hit convert to text via the
        # VLM; adding "image" passes media through untouched.
        tool_args_base["base_llm_modalities"] = list(base_llm_modalities)

    # When --insecure is set, build an httpx2 client that disables TLS
    # verification while preserving MCP defaults (redirects, timeouts).
    # MCP Python SDK v2 requires an httpx2 client for the HTTP transport.
    client_kwargs: dict = {"url": mcp_url, "terminate_on_close": False}
    if insecure:
        client_kwargs["http_client"] = httpx2.AsyncClient(
            follow_redirects=True,
            verify=False,
            timeout=httpx2.Timeout(30.0, read=300.0),
        )

    # One persistent session per user
    try:
        async with Client(streamable_http_client(**client_kwargs)) as client:
            while time.monotonic() < end_time:
                query = rng.choice(queries)
                tool_args = {**tool_args_base, "query": query}
                t0 = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        client.call_tool("search_dataset", tool_args),
                        timeout=call_timeout,
                    )
                    elapsed = time.monotonic() - t0
                    stats.total += 1
                    stats.latencies.append(elapsed)
                    if result.is_error:
                        stats.failed += 1
                        err_text = ""
                        for block in result.content:
                            if hasattr(block, "text"):
                                err_text = block.text[:120]
                                break
                        err_key = f"MCP error: {err_text}" if err_text else "MCP error"
                        stats.errors[err_key] = stats.errors.get(err_key, 0) + 1
                    else:
                        stats.success += 1
                        stats.result_counts.append(1)
                except TimeoutError:
                    elapsed = time.monotonic() - t0
                    stats.total += 1
                    stats.failed += 1
                    stats.latencies.append(elapsed)
                    err_key = f"Timeout (>{call_timeout:g}s)"
                    stats.errors[err_key] = stats.errors.get(err_key, 0) + 1
                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    stats.total += 1
                    stats.failed += 1
                    stats.latencies.append(elapsed)
                    err_key = type(exc).__name__
                    stats.errors[err_key] = stats.errors.get(err_key, 0) + 1
                    # Brief pause to avoid tight error loop
                    await asyncio.sleep(0.5)
    except Exception as exc:
        # Session creation / initialisation failed
        stats.total += 1
        stats.failed += 1
        stats.latencies.append(0.0)
        err_key = f"MCP connect: {type(exc).__name__}"
        stats.errors[err_key] = stats.errors.get(err_key, 0) + 1


# ---------------------------------------------------------------------------
# Server discovery helpers
# ---------------------------------------------------------------------------


async def check_health(base_url: str, insecure: bool = False) -> bool:
    """Return True if the server health check passes."""
    try:
        async with httpx.AsyncClient(timeout=10, verify=not insecure) as c:
            resp = await c.get(f"{base_url}/healthz")
            return resp.status_code == 200
    except Exception:
        return False


async def discover_datasets(base_url: str, password: str | None = None, insecure: bool = False, api_key: str | None = None) -> list[str]:
    """Return a list of dataset names from the server."""
    headers: dict[str, str] = {}
    if password:
        headers["X-Dataset-Password"] = password
    if api_key:
        headers["X-RAG-Api-Key"] = api_key
    async with httpx.AsyncClient(timeout=10, verify=not insecure) as c:
        resp = await c.get(f"{base_url}/api/datasets", headers=headers)
        resp.raise_for_status()
        return [ds["name"] for ds in resp.json().get("datasets", [])]


def load_queries(queries_file: str | None) -> list[str]:
    """Load queries from a file (one per line) or return defaults."""
    if queries_file is None:
        return DEFAULT_QUERIES
    with open(queries_file, encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not queries:
        print("Error: queries file is empty", file=sys.stderr)
        sys.exit(1)
    return queries


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def _run_with_progress(tasks: list, stats: BenchmarkStats) -> None:
    """Gather *tasks* while printing progress every 5 seconds as a table.

    The column header is reprinted every 20 data rows so long runs stay
    readable when scrolled.
    """
    fmt = "  {:>8}  {:>9}  {:>9}  {:>9}  {:>12}"
    header = fmt.format("elapsed", "requests", "success", "failed", "rate (req/s)")
    sep = fmt.format("-" * 8, "-" * 9, "-" * 9, "-" * 9, "-" * 12)
    rows_since_header = 0

    async def report_progress() -> None:
        nonlocal rows_since_header
        while True:
            await asyncio.sleep(5)
            elapsed = time.monotonic() - stats.start_time
            if rows_since_header % 20 == 0:
                print(header)
                print(sep)
            rate = f"{stats.total / elapsed:.1f}" if elapsed > 0 else "starting"
            print(fmt.format(f"{elapsed:.0f}s", stats.total, stats.success, stats.failed, rate))
            rows_since_header += 1

    progress_task = asyncio.create_task(report_progress())
    await asyncio.gather(*tasks)
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass


async def run_benchmark(args: argparse.Namespace) -> None:
    api_url = args.url.rstrip("/")
    is_mcp = args.mode == "mcp"

    # In MCP mode, --url is the MCP endpoint; --api-url (or derived) is REST.
    # In REST mode, --url is the REST API base.
    if is_mcp:
        mcp_url = args.url.rstrip("/")
        api_url = (args.api_url or "").rstrip("/")
        if not api_url:
            from urllib.parse import urlparse

            parsed = urlparse(mcp_url)
            api_url = f"{parsed.scheme}://{parsed.hostname}:8000"
            print(f"  (derived REST API URL: {api_url})")
    else:
        mcp_url = ""

    # 1. Health check (always via REST API)
    if api_url:
        print(f"Checking server health at {api_url} ...", end=" ", flush=True)
        if not await check_health(api_url, insecure=args.insecure):
            print("FAILED")
            print("Error: server is not responding. Check --url / --api-url.", file=sys.stderr)
            sys.exit(1)
        print("OK")
    else:
        print("Skipping health check (no REST API URL available)")

    # 2. Resolve dataset
    dataset = args.dataset
    if not dataset:
        if not api_url:
            print("Error: --dataset is required when no REST API URL is available for discovery.", file=sys.stderr)
            sys.exit(1)
        print("No --dataset specified; discovering ...", end=" ", flush=True)
        try:
            datasets = await discover_datasets(api_url, args.password, insecure=args.insecure, api_key=args.api_key)
            if not datasets:
                print("none found")
                print("Error: no datasets available. Create one first.", file=sys.stderr)
                sys.exit(1)
            dataset = datasets[0]
            print(f"using '{dataset}'")
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

    # 3. Load queries
    if args.queries_file:
        queries = load_queries(args.queries_file)
        query_set_label = "custom (--queries-file)"
    else:
        queries = {
            "generic": DEFAULT_QUERIES_GENERIC,
            "vlm": DEFAULT_QUERIES_VLM,
            "mixed": DEFAULT_QUERIES,
        }[args.query_set]
        query_set_label = args.query_set
    print(f"Query pool: {len(queries)} queries (set: {query_set_label})")

    # 4. Print config
    reranker_status = f"ON (reranker_top_k={args.reranker_top_k})" if args.use_reranker else "OFF"
    print()
    print("=" * 60)
    print("  BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"  Mode:            {args.mode.upper()}")
    if is_mcp:
        print(f"  MCP endpoint:    {mcp_url}")
    print(f"  REST API:        {api_url or '(not used)'}")
    print(f"  Dataset:         {dataset}")
    print(f"  Users (N):       {args.N}")
    print(f"  Duration:        {args.duration}s")
    print(f"  Ramp-up:         {args.ramp_up}s")
    print(f"  Top K:           {args.top_k}")
    print(f"  Reranker:        {reranker_status}")
    print(f"  Queries:         {len(queries)}")
    print(f"  Password:        {'set' if args.password else 'none'}")
    print(f"  TLS verify:      {'OFF (--insecure)' if args.insecure else 'ON'}")
    if is_mcp:
        print(f"  Call timeout:    {args.call_timeout:g}s")
    if not is_mcp:
        print(f"  HTTP/2:          {args.http2}")
    print("=" * 60)
    print()

    # 5. Build and run tasks for N users
    stats = BenchmarkStats()
    ramp_step = args.ramp_up / args.N if args.N > 0 else 0

    if is_mcp:
        tasks = [
            run_user_mcp(
                user_id=i,
                mcp_url=mcp_url,
                dataset=dataset,
                queries=queries,
                top_k=args.top_k,
                use_reranker=args.use_reranker,
                reranker_top_k=args.reranker_top_k,
                password=args.password,
                duration=args.duration,
                ramp_delay=i * ramp_step,
                stats=stats,
                base_llm_modalities=args.base_llm_modalities,
                insecure=args.insecure,
                call_timeout=args.call_timeout,
            )
            for i in range(args.N)
        ]
        stats.start_time = time.monotonic()
        print(f"Launching {args.N} concurrent users ({args.mode} mode) ...", flush=True)
        await _run_with_progress(tasks, stats)
        stats.end_time = time.monotonic()
    else:
        limits = httpx.Limits(
            max_connections=args.N * 2 + 10,
            max_keepalive_connections=args.N,
        )
        async with httpx.AsyncClient(limits=limits, http2=args.http2, verify=not args.insecure) as client:
            tasks = [
                run_user(
                    user_id=i,
                    client=client,
                    base_url=api_url,
                    dataset=dataset,
                    queries=queries,
                    top_k=args.top_k,
                    use_reranker=args.use_reranker,
                    reranker_top_k=args.reranker_top_k,
                    password=args.password,
                    duration=args.duration,
                    ramp_delay=i * ramp_step,
                    stats=stats,
                    api_key=args.api_key,
                )
                for i in range(args.N)
            ]
            stats.start_time = time.monotonic()
            print(f"Launching {args.N} concurrent users ({args.mode} mode) ...", flush=True)
            await _run_with_progress(tasks, stats)
            stats.end_time = time.monotonic()

    # 6. Print results
    summary = stats.summary()
    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Duration:            {summary['duration_s']}s")
    print(f"  Total requests:      {summary['total_requests']}")
    print(f"  Successful:          {summary['successful']}")
    print(f"  Failed:              {summary['failed']}")
    print(f"  Success rate:        {summary['success_rate_pct']}%")
    print(f"  Response rate:       {summary['response_rate_rps']} req/s")
    print()
    print("  Latency (ms):")
    lat = summary["latency_ms"]
    print(f"    min:     {lat['min']}")
    print(f"    mean:    {lat['mean']}")
    print(f"    median:  {lat['median']}")
    print(f"    p95:     {lat['p95']}")
    print(f"    p99:     {lat['p99']}")
    print(f"    max:     {lat['max']}")
    print()
    print(f"  Avg results/query:   {summary['avg_results_per_query']}")
    if summary["status_codes"]:
        print(f"  Status codes:        {summary['status_codes']}")
    if summary["errors"]:
        print(f"  Errors:              {summary['errors']}")
    print("=" * 60)

    # 7. Save JSON if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "mode": args.mode,
                        "url": api_url,
                        "mcp_url": mcp_url if is_mcp else None,
                        "dataset": dataset,
                        "N": args.N,
                        "duration": args.duration,
                        "ramp_up": args.ramp_up,
                        "top_k": args.top_k,
                        "use_reranker": args.use_reranker,
                        "reranker_top_k": args.reranker_top_k,
                        "num_queries": len(queries),
                        "insecure": args.insecure,
                        "call_timeout": args.call_timeout if is_mcp else None,
                    },
                    "results": summary,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Multimodal RAG platform with N concurrent users.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # 50 users, 30s, REST mode, no reranker
  python tests/benchmark.py --url http://localhost:8000 --dataset my-ds --N 50

  # 200 users, 60s, REST mode, with reranker
  python tests/benchmark.py --url http://localhost:8000 --dataset my-ds \\
      --N 200 --duration 60 --use-reranker --reranker-top-k 5

  # 100 users, MCP mode (port-forwarded: MCP on 9090, REST on 8000)
  python tests/benchmark.py --mode mcp --url http://localhost:9090/mcp \\
      --api-url http://localhost:8000 --dataset my-ds --N 100

  # MCP mode via external virtualservice (same host, /mcp path)
  python tests/benchmark.py --mode mcp --url https://rag.example.com/mcp \\
      --api-url https://rag.example.com --dataset my-ds --N 100

  # Auto-discover first dataset, save results
  python tests/benchmark.py --url http://localhost:8000 --N 100 --output results.json
""",
    )
    parser.add_argument(
        "--mode",
        choices=["rest", "mcp"],
        default="rest",
        help="API to benchmark: 'rest' (GET /api/datasets/{name}/search) or "
        "'mcp' (search_dataset tool via streamable-http /mcp endpoint). Default: rest.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="In REST mode: base URL of the RAG API server. "
        "In MCP mode: full MCP endpoint URL (e.g. http://localhost:9090/mcp). "
        "Default: http://localhost:8000",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="REST API URL for health check and dataset discovery in MCP mode. "
        "If omitted in MCP mode, derived from --url (same host, port 8000).",
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset name to query. If omitted, auto-discovers the first available dataset."
    )
    parser.add_argument(
        "-N", "--N", dest="N", type=int, default=10, help="Number of concurrent simulated users (default: 10)"
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds (default: 30)")
    parser.add_argument(
        "--ramp-up", type=float, default=5.0, help="Ramp-up time in seconds to stagger user starts (default: 5)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve per query (default: 10)")
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking. When set, the server reranks top_k "
        "results and returns reranker_top_k of them.",
    )
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=3,
        help="Number of results to keep after reranking (only used when --use-reranker is set, default: 3)",
    )
    parser.add_argument("--password", default=None, help="Dataset password (for password-protected datasets)")
    parser.add_argument(
        "--api-key",
        default=None,
        help="RAG API key sent as X-RAG-Api-Key. Required for REST mode when the "
        "server enforces security.apiKey; dataset discovery uses it too. "
        "(The MCP sidecar does not enforce the key.)",
    )
    parser.add_argument(
        "--base-llm-modalities",
        default=None,
        help="Comma-separated modalities of the simulated LLM for MCP mode "
        "(e.g. 'text,image'). Default: unset -> the server assumes text-only and "
        "converts every image hit to text via the VLM, which dominates latency "
        "for image-heavy datasets.",
    )
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Path to a file with one query per line. If omitted, uses the built-in pool selected by --query-set.",
    )
    parser.add_argument(
        "--query-set",
        choices=["generic", "vlm", "mixed"],
        default="generic",
        help="Built-in query pool when --queries-file is not given (default: generic). "
        "'generic' exercises the caption-reuse path (no VLM at query time for "
        "pre-captioned docs); 'vlm' trips the server's _query_needs_vlm() heuristic "
        "— for text-only callers this re-runs the VLM per image hit, which is "
        "realistic but much slower; 'mixed' is the historical combined pool.",
    )
    parser.add_argument("--output", default=None, help="Path to save results as JSON")
    parser.add_argument("--http2", action="store_true", help="Use HTTP/2 (requires h2 package)")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (for internal/dev servers with "
        "private or self-signed CAs). Applies to REST health check, dataset discovery, "
        "and the MCP streamable-http client.",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=120.0,
        help="Per-call timeout in seconds for a single search (MCP mode only). "
        "Caps how long call_tool may block so high-concurrency runs terminate near "
        "--duration instead of waiting for queued stragglers (MCP read timeout is 300s). "
        "Timed-out calls are recorded as failures. Default: 120.",
    )
    args = parser.parse_args()

    if args.N < 1:
        parser.error("-N must be at least 1")
    if args.duration < 1:
        parser.error("--duration must be at least 1 second")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.reranker_top_k < 1:
        parser.error("--reranker-top-k must be at least 1")
    if args.call_timeout < 1:
        parser.error("--call-timeout must be at least 1 second")

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
