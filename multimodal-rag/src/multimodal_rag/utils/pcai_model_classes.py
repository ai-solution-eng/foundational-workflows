import asyncio
import base64
import io
import os
import threading
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI, OpenAI

try:
    from agents import Agent, set_tracing_disabled
    from agents.mcp import MCPServerStreamableHttp
    from agents.models.openai_responses import OpenAIResponsesModel
except ImportError:
    Agent = None  # type: ignore[assignment, misc]
    set_tracing_disabled = None  # type: ignore[assignment]
    MCPServerStreamableHttp = None  # type: ignore[assignment, misc]
    OpenAIResponsesModel = None  # type: ignore[assignment, misc]

from .general_tools import sync_wrapper_safe
from .logging_utils import logging
from .model_adapters import MultiModalEmbeddings, MultiModalReranker
from .token_text_splitter import TokenTextSplitter

logger = logging.getLogger(__name__)

_MLIS_PAGE = "https://mlis.pcai-se-ai-application.hst.rdlabs.hpecorp.net/ui/deployments"
_NOT_DEPLOYED = ""


def _pool_limits_from_env(default_max_connections: int | None = 30) -> httpx.Limits:
    """Build httpx connection-pool limits from env vars.

    Async clients default to ``max_connections=30`` (matching the original
    single-replica chart).  The sync ``_SHARED_REMOTE_HTTP_CLIENT`` calls
    this with ``default_max_connections=None`` to preserve its original
    unlimited behaviour.  The scale chart raises both via
    ``MODEL_POOL_MAX_CONNECTIONS`` so the embedder — on every search's
    critical path — is not the bottleneck at high concurrency.
    """
    raw = os.environ.get("MODEL_POOL_MAX_CONNECTIONS")
    max_conn: int | None
    if raw:
        try:
            max_conn = int(raw)
        except (TypeError, ValueError):
            max_conn = default_max_connections
    else:
        max_conn = default_max_connections
    try:
        max_keepalive = int(os.environ.get("MODEL_POOL_MAX_KEEPALIVE_CONNECTIONS", "10"))
    except (TypeError, ValueError):
        max_keepalive = 10
    return httpx.Limits(
        max_connections=max_conn if max_conn and max_conn > 0 else None,
        max_keepalive_connections=max(1, max_keepalive),
        keepalive_expiry=120.0,
    )


# Sync clients are thread-safe and can be shared globally.
# The sync remote client preserves the original unlimited max_connections
# (only keepalive is capped) so it never becomes a bottleneck for one-off
# model-discovery calls.
_SHARED_HTTP_CLIENT = httpx.Client()
_SHARED_REMOTE_HTTP_CLIENT = httpx.Client(
    verify=False,
    limits=_pool_limits_from_env(default_max_connections=None),
)

# Async clients must be bound to a single event loop.
# We cache one per (running loop, endpoint base_url) so that each model
# endpoint (embedder, reranker, VLM, ...) gets its own connection pool
# — a backlog of slow reranker calls can't starve the fast embedder path.
# The outer WeakKeyDictionary is keyed by loop (auto-evicts on loop GC);
# the inner dict is keyed by base_url.
_async_client_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_async_remote_client_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_async_client_lock = threading.Lock()

# Cache of discovered model names keyed by the ``{base_url}/models`` URL.
# Only successful lookups are stored so a temporarily-down endpoint is
# retried on the next construction. Sessions rebuild ASR/LLM/TTS frequently,
# so this avoids redundant /v1/models round-trips per session.
_discovered_model_names: dict[str, str] = {}
_discovered_model_lock = threading.Lock()


def discover_model_name(base_url: str, api_key: str = "", remote: bool = True) -> str:
    """Best-effort lookup of the served model id via ``GET {base_url}/models``.

    ``base_url`` is the OpenAI base URL (root + ``"/v1"``).  PCAI/vLLM endpoints
    serve a single model, so the first ``data[].id`` is returned.  Returns
    ``""`` on any failure (network, non-200, empty list) -- keeping this
    best-effort so a transient blip never hard-fails the caller.  Successful
    lookups are cached per URL; failures are not, so a recovered endpoint is
    retried on the next call.
    """
    url = f"{base_url}/models"
    cached = _discovered_model_names.get(url)
    if cached:
        return cached
    client = _SHARED_REMOTE_HTTP_CLIENT if remote else _SHARED_HTTP_CLIENT
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = client.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        ids = [str(m["id"]) for m in models if isinstance(m, dict) and m.get("id")]
        if ids:
            logger.info(
                "Auto-discovered model name '%s' from %s (available: %s)",
                ids[0],
                url,
                ids,
            )
            with _discovered_model_lock:
                _discovered_model_names[url] = ids[0]
            return ids[0]
        logger.warning("No models listed at %s; leaving model_name empty.", url)
    except Exception as e:
        logger.warning(
            "Could not auto-discover model name from %s: %s. "
            "Set model_name explicitly if the endpoint lacks /v1/models.",
            url,
            e,
        )
    return ""


def _get_async_client(remote: bool = False, base_url: str = "") -> httpx.AsyncClient:
    """Return a cached ``httpx.AsyncClient`` for *base_url* on the running loop.

    Each (event loop, endpoint) pair gets its own connection pool so a
    slow endpoint (e.g. the reranker) cannot exhaust connections needed
    by a fast one (e.g. the embedder).  When *base_url* is empty a
    fallback shared pool is used (back-compat for callers that don't
    specify an endpoint).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        # No running loop — create a short-lived client.  Caller is
        # responsible for closing it.
        return httpx.AsyncClient(
            verify=not remote,
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=_pool_limits_from_env(),
        )
    cache = _async_remote_client_cache if remote else _async_client_cache
    with _async_client_lock:
        loop_clients: dict[str, httpx.AsyncClient] | None = cache.get(loop)
        if loop_clients is None:
            loop_clients = {}
            cache[loop] = loop_clients
        client = loop_clients.get(base_url)
        if client is None:
            client = httpx.AsyncClient(
                verify=not remote,
                timeout=httpx.Timeout(300.0, connect=30.0),
                limits=_pool_limits_from_env(),
            )
            loop_clients[base_url] = client
        return client


__all__ = [
    "BaseModel",
    "ChatModel",
    "EmbeddingModel",
    "RerankerModel",
    "ToolDefinition",
    "VoiceModel",
    "discover_model_name",
]

input_modalities = Literal["text", "audio", "image", "video"]
messages_dtype = str | dict[str, Any] | list[dict[str, Any]]


@dataclass
class ToolDefinition:
    """Describes a single MCP tool exposed by a model endpoint.

    The ``handler`` is an async callable taking a single ``dict[str, Any]``
    of arguments (matching ``input_schema``) and returning a JSON-serializable
    dict.  The orchestration layer wraps handlers with depth/budget
    instrumentation before they reach the MCP server.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]


async def _get_mcp_servers(mcp_servers: dict[str, dict[str, Any]]) -> list[MCPServerStreamableHttp]:
    """Build and connect one MCPServerStreamableHttp per entry in the
    {name: {url, headers, transport}} config dict. Each server is returned
    already connected; the caller is responsible for calling cleanup() on
    each when the owning session ends.
    """
    servers: list[MCPServerStreamableHttp] = []
    for name, cfg in mcp_servers.items():
        try:
            params: dict[str, Any] = {
                "url": cfg["url"],
                # TLS-bypass httpx factory (PCAI ingress serves self-signed
                # certs); defined below.
                "httpx_client_factory": _streamable_http_factory,
                # Skip the session-terminate DELETE on cleanup - the PCAI
                # istio ingress doesn't support DELETE on /mcp and the call
                # hangs until asyncio tears the loop down. MCP servers will
                # reap idle sessions by TTL on their side.
                "terminate_on_close": False,
            }
            if cfg.get("headers"):
                params["headers"] = cfg["headers"]
            server = MCPServerStreamableHttp(params=params, name=name)  # type: ignore[arg-type]
            await server.connect()
            servers.append(server)
        except Exception as e:
            logger.warning("Failed to load MCP server %s: %s", name, e)
    return servers


def _streamable_http_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """httpx client factory for MCPServerStreamableHttp that disables TLS
    verification - required for the PCAI ingress which serves self-signed
    certs. Mirrors the default factory signature so it can be dropped in
    via params['httpx_client_factory']."""
    kwargs: dict = {"follow_redirects": False, "verify": False}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


class _NamedBytesIO(io.BytesIO):
    """BytesIO with a ``name`` attribute, required by the OpenAI
    transcription API to infer the audio format."""

    def __init__(self, content: bytes, name: str = "audio"):
        super().__init__(content)
        self.name = name


@dataclass
class BaseModel:
    # Model args. ``model_name`` is optional: when left empty (and the model
    # is deployed) it is auto-discovered at construction time from the
    # OpenAI-compatible ``GET {base_url}/models`` endpoint -- the same trick
    # Open WebUI uses. Pass an explicit name to pin/override it.
    model_name: str = ""
    url_remote: str = ""

    # Model args w/ defaults
    description: str = ""
    model_instantiation_class: Callable | None = None
    model_instantiation_kwargs: dict[str, Any] = field(default_factory=dict)

    # OpenAI clients
    model_client_class: Callable = OpenAI
    model_async_client_class: Callable = AsyncOpenAI

    model_usage: Literal["local", "remote"] = "local"
    api_key: str = ""

    _cached_properties: tuple[str, ...] = field(
        default=(
            "client",
            "async_client",
            "base_url",
            "http_client",
            "http_async_client",
            "model",
        ),
        init=False,
        repr=False,
    )
    _cached_functions: tuple[str, ...] = field(default=(), init=False, repr=False)

    currently_deployed: bool = True

    allowable_modalities: tuple[input_modalities, ...] = ("text",)

    @classmethod
    def from_config(cls, config: dict[str, Any], api_key: str = "") -> "BaseModel":
        """Construct from a DB config dict + resolved api_key.

        Only fields that exist on the dataclass are mapped.  Callable
        fields (``preprocessor``, ``model_instantiation_class``) are
        skipped when the JSON value is a string — they can't be
        deserialized.  Lists are converted to tuples/sets where the
        dataclass expects them.
        """
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(cls)}
        _CALLABLE_FIELDS = frozenset(
            {
                "preprocessor",
                "model_instantiation_class",
                "model_client_class",
                "model_async_client_class",
            }
        )
        kwargs: dict[str, Any] = {}
        for k, v in config.items():
            if k not in field_names:
                continue
            if k in _CALLABLE_FIELDS and not callable(v):
                continue
            if k == "allowable_modalities" and isinstance(v, list):
                v = tuple(v)
            if k == "tts_supported_voices" and isinstance(v, list):
                v = set(v)
            kwargs[k] = v
        kwargs["api_key"] = api_key
        return cls(**kwargs)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        # Auto-discover the model name from the serving endpoint when none was
        # supplied.  This mirrors Open WebUI, which lists models via the
        # OpenAI ``/v1/models`` API rather than requiring a hardcoded name.
        # Disabled models are skipped so the catalog can hold placeholders.
        if not self.model_name and self.currently_deployed:
            self.model_name = self._discover_model_name()

    def _discover_model_name(self) -> str:
        """Look up the served model id via ``GET {base_url}/models``.

        Thin instance wrapper around :func:`discover_model_name` that passes
        this model's resolved base URL, API key, and transport.
        """
        return discover_model_name(
            self.base_url,
            self.api_key,
            remote=self.model_usage == "remote",
        )

    def _clear_cached_class_elements(self) -> None:
        for property in self._cached_properties + self._cached_functions:
            if property in self.__dict__:
                self.__dict__.pop(property)
                logger.info(f"Removed property {property}")

    @staticmethod
    def _convert_remote_url_to_local(path: str) -> str:
        new_path = path.replace("https", "http")
        return new_path[: new_path.find(".serving.")] + ".svc.cluster.local"

    @cached_property
    def client(self) -> OpenAI:
        assert self.currently_deployed, (
            f"Model {self.model_name} is not currently deployed. "
            f"See {_MLIS_PAGE} and change flag `currently_deployed` to enable."
        )
        return self.model_client_class(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_client,
        )

    @cached_property
    def async_client(self) -> AsyncOpenAI:
        assert self.currently_deployed, (
            f"Model {self.model_name} is not currently deployed. "
            f"See {_MLIS_PAGE} and change flag `currently_deployed` to enable."
        )
        return self.model_async_client_class(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_async_client,
        )

    @cached_property
    def url_local(self) -> str:
        return self._convert_remote_url_to_local(self.url_remote)

    @cached_property
    def base_url(self) -> str:
        return (self.url_local if self.model_usage == "local" else self.url_remote) + "/v1"

    @cached_property
    def http_client(self):
        return _SHARED_REMOTE_HTTP_CLIENT if self.model_usage == "remote" else _SHARED_HTTP_CLIENT

    @cached_property
    def http_async_client(self):
        return _get_async_client(remote=self.model_usage == "remote", base_url=self.base_url)

    @cached_property
    def model(self):
        return self.build_model()

    def build_model(self, **kwargs):
        if self.model_instantiation_class is None:
            raise ValueError(
                "model_instantiation_class is not set. The old langchain-based "
                "(ChatOpenAI / OpenAIEmbeddings) were removed; pass an explicit "
                "callable (e.g. MultiModalEmbeddings) or use `.client` / "
                "`.async_client` for direct OpenAI API access."
            )
        return self.model_instantiation_class(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_client,
            http_async_client=self.http_async_client,
            **kwargs,
            **self.model_instantiation_kwargs,
        )

    def remote(self) -> None:
        self.model_usage = "remote"
        self._clear_cached_class_elements()

    def local(self) -> None:
        self.model_usage = "local"
        self._clear_cached_class_elements()

    _REPR_SKIP = frozenset(
        {
            "_cached_properties",
            "_cached_functions",
            "model_client_class",
            "model_async_client_class",
            "model_instantiation_class",
            "model_instantiation_kwargs",
        }
    )

    def __repr__(self) -> str:
        fields = []
        from dataclasses import fields as dc_fields

        for f in dc_fields(self):
            if f.name in self._REPR_SKIP or not f.repr:
                continue
            val = getattr(self, f.name)
            # Mask long API keys
            if f.name == "api_key" and isinstance(val, str) and len(val) > 20:
                val = val[:12] + "..." + val[-4:]
            # Truncate long URLs
            if f.name == "url_remote" and isinstance(val, str) and len(val) > 80:
                val = val[:60] + "..." + val[-17:]
            # Pretty-print modality tuples
            if f.name == "allowable_modalities" and isinstance(val, tuple):
                val = "(" + ", ".join(val) + ")"
            fields.append(f"  {f.name}={val}")
        return self.__class__.__name__ + "(\n" + "\n".join(fields) + "\n)"


@dataclass(repr=False)
class ChatModel(BaseModel):
    _cached_functions = (
        "llm_chat_function",
        "llm_async_chat_function",
        "llm_response_function",
        "llm_async_response_function",
    )

    @cached_property
    def model(self):
        m = super().model
        return m

    @staticmethod
    def _fix_chat_inputs(messages: messages_dtype) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        return messages

    @cached_property
    def llm_chat_function(self) -> Callable:
        return partial(self.client.chat.completions.create, model=self.model_name)

    def llm_chat_function_call(self, messages: messages_dtype, **chat_kwargs):

        return self.llm_chat_function(messages=self._fix_chat_inputs(messages), **chat_kwargs)

    @cached_property
    def llm_async_chat_function(self) -> Callable:
        return partial(self.async_client.chat.completions.create, model=self.model_name)

    def llm_async_chat_function_call(self, messages: str | dict[str, Any] | list[dict[str, Any]], **chat_kwargs):
        return self.llm_async_chat_function(messages=self._fix_chat_inputs(messages), **chat_kwargs)

    @cached_property
    def llm_response_function(self) -> Callable:
        return partial(self.client.responses.create, model=self.model_name)

    def llm_response_function_call(self, input: str | dict[str, Any] | list[dict[str, Any]], **chat_kwargs):
        return self.llm_response_function(input=input, **chat_kwargs)

    @cached_property
    def llm_async_response_function(self) -> Callable:
        return partial(self.async_client.responses.create, model=self.model_name)

    def llm_async_response_function_call(self, input: str | dict[str, Any] | list[dict[str, Any]], **chat_kwargs):
        return self.llm_async_response_function(input=input, **chat_kwargs)

    def agent(self, tool_json: dict[str, dict[str, Any]] | None = None):
        if Agent is None:
            raise ImportError("The `agents` SDK is not installed. Install with: pip install openai-agents")
        return sync_wrapper_safe(self.aagent, {"tool_json": tool_json})

    async def aagent(self, tool_json: dict[str, dict[str, Any]] | None = None) -> Agent:
        if Agent is None:
            raise ImportError("The `agents` SDK is not installed. Install with: pip install openai-agents")
        # Tracing off: the SDK phones home to api.openai.com by default,
        # which fails behind the PCAI firewall and adds noise to logs.
        # set_tracing_disabled is process-global; safe to call repeatedly.
        set_tracing_disabled(True)
        model_obj = OpenAIResponsesModel(model=self.model_name, openai_client=self.async_client)
        if not tool_json:
            return Agent(name=self.model_name, model=model_obj)
        servers = await _get_mcp_servers(tool_json)
        return Agent(name=self.model_name, model=model_obj, mcp_servers=servers)  # type: ignore[arg-type]

    def to_mcp_tools(self) -> list[ToolDefinition]:
        """Expose this chat model as a single ``respond`` tool using the
        OpenAI Responses API (``responses.create``).

        The handler always does a single non-looping call — tool-calling
        agent loops are handled by the orchestration layer, not here.
        """

        async def _respond(arguments: dict[str, Any]) -> dict[str, Any]:
            params: dict[str, Any] = {
                "model": self.model_name,
                "input": arguments["input"],
            }
            if "instructions" in arguments:
                params["instructions"] = arguments["instructions"]
            for k in ("temperature", "max_output_tokens", "top_p"):
                if k in arguments:
                    params[k] = arguments[k]

            response = await self.async_client.responses.create(**params)
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                }
            return {
                "output": response.output_text,
                "model": self.model_name,
                "usage": usage,
            }

        modalities_str = ", ".join(self.allowable_modalities)
        return [
            ToolDefinition(
                name="respond",
                description=self.description
                or f"Generate a response using {self.model_name}. Supports {modalities_str} inputs.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": ["string", "array"],
                            "description": (
                                "The input text, or a structured message array for multi-turn / multimodal input."
                            ),
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Optional system-level instructions to guide the model's behavior.",
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0-2). Higher = more random.",
                        },
                        "max_output_tokens": {
                            "type": "integer",
                            "description": "Maximum number of tokens to generate.",
                        },
                    },
                    "required": ["input"],
                },
                handler=_respond,
            )
        ]


@dataclass(repr=False)
class VoiceModel(BaseModel):
    model_instantiation_class: Callable = OpenAI

    model_type: Literal["ASR", "TTS", "JOINT"] = "ASR"

    tts_supported_voices: set[str] = field(default_factory=set)
    tts_voice: str = "alys"
    tts_skip_verify: bool = False

    _cached_properties = (
        "client",
        "async_client",
        "base_url",
        "http_client",
        "http_async_client",
        "model",
        "tts_supported_voices",
    )
    _cached_functions = (
        "tts_function",
        "tts_async_function",
        "asr_function",
        "asr_async_function",
    )

    allowable_modalities = (
        "text",
        "audio",
    )

    def __post_init__(self) -> None:
        # Resolve model_name first (BaseModel auto-discovery via /v1/models),
        # then seed the lazy voices cache.
        super().__post_init__()
        # Save the init-provided voices as a seed and remove from __dict__
        # so the cached_property descriptor takes over on first access.
        # This lets callers do:  tts = VoiceModel(...); tts.remote();
        # tts.tts_supported_voices  → fetches via remote transport.
        seed = self.__dict__.pop("tts_supported_voices", set())
        self._tts_voices_seed: set[str] = seed

    @cached_property  # type: ignore[no-redef]
    def tts_supported_voices(self) -> set[str]:  # noqa: F811
        """Available voices, resolved lazily on first access.

        If voices were provided at init time they are used directly.
        Otherwise voices are fetched from ``{base_url}/audio/voices``.
        Because this is a cached_property listed in ``_cached_properties``,
        calling ``.remote()`` or ``.local()`` clears the cache so the next
        access re-fetches with the updated endpoint / transport.
        """
        if self.tts_skip_verify:
            return set()
        if self._tts_voices_seed:
            return self._tts_voices_seed
        voices: set[str] = set()
        try:
            url = f"{self.base_url}/audio/voices"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = self.http_client.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            for v in data.get("voices", []):
                voices.add(v if isinstance(v, str) else v.get("name", ""))
            for v in data.get("uploaded_voices", []):
                voices.add(v.get("name", "") if isinstance(v, dict) else str(v))
            voices.discard("")
            logger.info(f"Dynamically fetched {len(voices)} voices from {url}: {sorted(voices)}")
        except Exception as e:
            logger.warning(f"Could not fetch voices from {url}: {e}. Voice set will be empty.")
        return voices

    @cached_property
    def model(self) -> OpenAI:
        raise ValueError(
            "Use `client` instead of `model`, or make calls directly with `tts_function` or `asr_function`."
        )

    @cached_property
    def tts_function(self) -> Callable:
        assert self.model_type != "ASR", "No `tts_function` is available for ASR model_type."
        return partial(self.client.audio.speech.create, model=self.model_name, voice=self.tts_voice)

    def tts_function_call(self, input: str, **chat_kwargs):
        return self.tts_function(input=input, **chat_kwargs)

    @cached_property
    def tts_async_function(self) -> Callable:
        assert self.model_type != "ASR", "No `tts_function` is available for ASR model_type."
        return partial(
            self.async_client.audio.speech.create,
            model=self.model_name,
            voice=self.tts_voice,
        )

    def tts_async_function_call(self, input: str, **chat_kwargs):
        return self.tts_async_function(input=input, **chat_kwargs)

    @cached_property
    def asr_function(self) -> Callable:
        assert self.model_type != "TTS", "No `asr_function` is available for TTS model_type."
        return partial(self.client.audio.transcriptions.create, model=self.model_name)

    def asr_function_call(self, file, **chat_kwargs):
        return self.asr_function(file=file, **chat_kwargs)

    @cached_property
    def asr_async_function(self) -> Callable:
        assert self.model_type != "TTS", "No `asr_function` is available for TTS model_type."
        return partial(self.async_client.audio.transcriptions.create, model=self.model_name)

    def asr_async_function_call(self, file, **chat_kwargs):
        return self.asr_async_function(file=file, **chat_kwargs)

    def to_mcp_tools(self) -> list[ToolDefinition]:
        """Expose TTS as ``synthesize`` and/or ASR as ``transcribe``,
        depending on ``model_type``."""

        tools: list[ToolDefinition] = []

        if self.model_type != "ASR":

            async def _synthesize(arguments: dict[str, Any]) -> dict[str, Any]:
                text = arguments["text"]
                voice = arguments.get("voice", self.tts_voice)
                response = await self.tts_async_function_call(input=text, voice=voice)
                return {
                    "audio_base64": base64.b64encode(response.content).decode(),
                    "model": self.model_name,
                    "voice": voice,
                }

            voices_str = ", ".join(sorted(self.tts_supported_voices)) or "default"
            tools.append(
                ToolDefinition(
                    name="synthesize",
                    description=self.description or f"Synthesize speech from text using {self.model_name}.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to synthesize.",
                            },
                            "voice": {
                                "type": "string",
                                "description": f"Voice to use. Available: {voices_str}.",
                            },
                        },
                        "required": ["text"],
                    },
                    handler=_synthesize,
                )
            )

        if self.model_type != "TTS":

            async def _transcribe(arguments: dict[str, Any]) -> dict[str, Any]:
                audio_b64 = arguments.get("audio_base64")
                audio_url = arguments.get("audio_url")

                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                elif audio_url:
                    resp = await self.http_async_client.get(audio_url, follow_redirects=True)
                    resp.raise_for_status()
                    audio_bytes = resp.content
                else:
                    return {"error": "Either audio_base64 or audio_url must be provided."}

                buf = _NamedBytesIO(audio_bytes)
                result = await self.asr_async_function_call(file=buf)
                return {
                    "text": result.text,
                    "model": self.model_name,
                }

            tools.append(
                ToolDefinition(
                    name="transcribe",
                    description=self.description or f"Transcribe audio to text using {self.model_name}.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "audio_base64": {
                                "type": "string",
                                "description": "Base64-encoded audio data.",
                            },
                            "audio_url": {
                                "type": "string",
                                "description": "URL of the audio file to transcribe.",
                            },
                        },
                    },
                    handler=_transcribe,
                )
            )

        return tools


@dataclass(repr=False)
class EmbeddingModel(BaseModel):
    # Model args w/ defaults
    model_instantiation_class: Callable | None = None

    # Optional RAG args
    embedding_dim: int = 4096
    chunk_size: int = 2048
    chunk_overlap: int = 256
    code_chunk_size: int = 8192
    code_chunk_overlap: int = 512

    # For enabling splitting by token
    tokenizer_name: str | None = None
    tokenizer_type: Literal["HuggingFace", "TikToken"] | None = None

    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)

    # If input should be preprocessed
    preprocessor: Callable | None = None

    allowable_modalities = ("text", "audio", "image", "video")

    @cached_property
    def text_splitter(self) -> TokenTextSplitter | None:
        """Return a token-count-aware text splitter, or ``None`` if the
        tokenizer file is not available (fall back to character-based)."""
        if self.tokenizer_type != "HuggingFace":
            logger.info(
                "Text chunking: character-based (tokenizer_type=%r; set "
                "tokenizer_type=HuggingFace to enable token counts)",
                self.tokenizer_type,
            )
            return None
        splitter = TokenTextSplitter.from_bundled(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if splitter is not None:
            logger.info(
                "Text chunking: token-based (chunk_size=%d, chunk_overlap=%d)",
                self.chunk_size,
                self.chunk_overlap,
            )
        else:
            logger.warning(
                "Text chunking: character-based fallback (tokenizer file not found despite tokenizer_type=HuggingFace)",
            )
        return splitter

    @cached_property
    def code_text_splitter(self) -> TokenTextSplitter | None:
        """Return a token-count-aware text splitter for code/structured data,
        using ``code_chunk_size`` / ``code_chunk_overlap``."""
        if self.tokenizer_type != "HuggingFace":
            logger.info(
                "Code chunking: character-based (tokenizer_type=%r; set "
                "tokenizer_type=HuggingFace to enable token counts)",
                self.tokenizer_type,
            )
            return None
        splitter = TokenTextSplitter.from_bundled(
            chunk_size=self.code_chunk_size,
            chunk_overlap=self.code_chunk_overlap,
        )
        if splitter is not None:
            logger.info(
                "Code chunking: token-based (chunk_size=%d, chunk_overlap=%d)",
                self.code_chunk_size,
                self.code_chunk_overlap,
            )
        else:
            logger.warning(
                "Code chunking: character-based fallback (tokenizer file not found despite tokenizer_type=HuggingFace)",
            )
        return splitter

    @cached_property
    def model(self):
        if self.model_instantiation_class is None or self.model_instantiation_class is MultiModalEmbeddings:
            return MultiModalEmbeddings(self)
        return super().model

    def to_mcp_tools(self) -> list[ToolDefinition]:
        """Expose this embedder as a single ``embed`` tool."""

        async def _embed(arguments: dict[str, Any]) -> dict[str, Any]:
            texts = arguments["texts"]
            embeddings = await self.model.aembed_documents(texts)
            return {
                "embeddings": embeddings,
                "model": self.model_name,
                "dim": self.embedding_dim,
                "count": len(embeddings),
            }

        modalities_str = ", ".join(self.allowable_modalities)
        return [
            ToolDefinition(
                name="embed",
                description=self.description
                or f"Generate embeddings using {self.model_name}. Supports {modalities_str} inputs.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": ["string", "object"]},
                            "description": (
                                "List of texts (strings) or multimodal dicts ({text, image, video, audio}) to embed."
                            ),
                        },
                    },
                    "required": ["texts"],
                },
                handler=_embed,
            )
        ]


@dataclass(repr=False)
class RerankerModel(BaseModel):
    # Model args w/ defaults
    model_instantiation_class: Callable = MultiModalReranker

    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)

    # If input should be preprocessed
    preprocessor: Callable | None = None

    allowable_modalities = ("text", "audio", "image", "video")

    @cached_property
    def model(self) -> MultiModalReranker:
        return self.model_instantiation_class(self)  # type: ignore[return-value]

    def to_mcp_tools(self) -> list[ToolDefinition]:
        """Expose this reranker as a single ``rerank`` tool."""

        async def _rerank(arguments: dict[str, Any]) -> dict[str, Any]:
            query = arguments["query"]
            documents = arguments["documents"]
            results = await self.model.arerank(query, documents)
            # arerank returns list[list[dict]] (one inner list per query).
            # For a single query (the common case), unwrap.
            if isinstance(query, (str, dict)):
                return {
                    "results": results[0] if results else [],
                    "model": self.model_name,
                }
            return {
                "results": results,
                "model": self.model_name,
            }

        return [
            ToolDefinition(
                name="rerank",
                description=self.description or f"Rerank documents by relevance to a query using {self.model_name}.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": ["string", "object"],
                            "description": "The search query (string or multimodal dict).",
                        },
                        "documents": {
                            "type": "array",
                            "items": {"type": ["string", "object"]},
                            "description": "List of documents to rank.",
                        },
                    },
                    "required": ["query", "documents"],
                },
                handler=_rerank,
            )
        ]
