from dataclasses import dataclass, field
from functools import cached_property, partial
import asyncio
import httpx
import threading
import weakref
from typing import Optional, Literal, Callable, Any

from openai import OpenAI, AsyncOpenAI

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain.agents import create_agent as lc_ca

from .logging_utils import logging
from .langchain_overrides import MultiModalEmbeddings, MultiModalReranker
from .general_tools import sync_wrapper_safe
from .token_text_splitter import TokenTextSplitter

logger = logging.getLogger(__name__)

_MLIS_PAGE = "https://mlis.pcai-se-ai-application.hst.rdlabs.hpecorp.net/ui/deployments"
_NOT_DEPLOYED = ""

# Sync clients are thread-safe and can be shared globally.
_SHARED_HTTP_CLIENT = httpx.Client()
_SHARED_REMOTE_HTTP_CLIENT = httpx.Client(
    verify=False,
    limits=httpx.Limits(
        max_keepalive_connections=10,
        keepalive_expiry=120.0,
    ),
)

# Async clients must be bound to a single event loop.
# We cache one per running loop (keyed by the loop object itself, not
# id(loop), to avoid id() recycling after GC hands a stale client to a
# fresh loop).  WeakKeyDictionary auto-evicts when the loop is GC'd.
_async_client_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_async_remote_client_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_async_client_lock = threading.Lock()


def _get_async_client(remote: bool = False) -> httpx.AsyncClient:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        # No running loop — create a short-lived client.  Caller is
        # responsible for closing it.
        limits = httpx.Limits(
            max_connections=30,
            max_keepalive_connections=10,
            keepalive_expiry=120.0,
        )
        return httpx.AsyncClient(
            verify=False if remote else True,
            timeout=httpx.Timeout(300.0, connect=30.0),
            limits=limits,
        )
    cache = _async_remote_client_cache if remote else _async_client_cache
    with _async_client_lock:
        client = cache.get(loop)
        if client is None:
            limits = httpx.Limits(
                max_connections=30,
                max_keepalive_connections=10,
                keepalive_expiry=120.0,
            )
            client = httpx.AsyncClient(
                verify=False if remote else True,
                timeout=httpx.Timeout(300.0, connect=30.0),
                limits=limits,
            )
            cache[loop] = client
        return client


__all__ = [
    "ChatModel",
    "EmbeddingModel",
    "RerankerModel",
    "VoiceModel",
]

input_modalities = Literal["text", "audio", "image", "video"]
messages_dtype = str | dict[str, Any] | list[dict[str, Any]]


async def _get_mcp_tools(mcp_servers) -> list[BaseTool]:
    try:
        client = MultiServerMCPClient(mcp_servers)
        tools = await client.get_tools()
        return tools
    except Exception as e:
        logger.warning("Failed to load MCP tools: %s", e)
        return []


@dataclass
class BaseModel:
    # Model args
    model_name: str
    url_remote: str

    # Model args w/ defaults
    model_instantiation_class: Callable = ChatOpenAI
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
    _cached_functions: tuple[str, ...] = field(default=tuple(), init=False, repr=False)

    currently_deployed: bool = True

    allowable_modalities: tuple[input_modalities, ...] = ("text",)

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
        return _get_async_client(remote=self.model_usage == "remote")

    @cached_property
    def model(self):
        return self.build_model()

    def build_model(self, **kwargs):
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
    def model(self) -> BaseChatModel:
        m = super().model

        assert isinstance(m, BaseChatModel)
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

    def agent(self, tool_json: Optional[dict[str, dict[str, Connection]]] = None):
        return sync_wrapper_safe(self.aagent, dict(tool_json=tool_json))

    async def aagent(self, tool_json: Optional[dict[str, dict[str, Connection]]] = None):
        if not tool_json:
            return lc_ca(self.model, None)
        tools = await _get_mcp_tools(tool_json)
        return lc_ca(self.model, tools=tools)


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
            logger.warning(f"Could not fetch voices from {url}: {e}. " f"Voice set will be empty.")
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


@dataclass(repr=False)
class EmbeddingModel(BaseModel):
    # Model args w/ defaults
    model_instantiation_class: Callable = OpenAIEmbeddings

    # Optional RAG args
    embedding_dim: int = 4096
    chunk_size: int = 2048
    chunk_overlap: int = 256
    code_chunk_size: int = 8192
    code_chunk_overlap: int = 512

    # For enabling splitting by token
    tokenizer_name: Optional[str] = None
    tokenizer_type: Optional[Literal["HuggingFace", "TikToken"]] = None

    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)

    # If input should be preprocessed
    preprocessor: Optional[Callable] = None

    allowable_modalities = ("text", "audio", "image", "video")

    @cached_property
    def text_splitter(self) -> Optional[TokenTextSplitter]:
        """Return a token-count-aware text splitter, or ``None`` if the
        tokenizer file is not available (fall back to character-based)."""
        if self.tokenizer_type != "HuggingFace":
            return None
        return TokenTextSplitter.from_bundled(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @cached_property
    def code_text_splitter(self) -> Optional[TokenTextSplitter]:
        """Return a token-count-aware text splitter for code/structured data,
        using ``code_chunk_size`` / ``code_chunk_overlap``."""
        if self.tokenizer_type != "HuggingFace":
            return None
        return TokenTextSplitter.from_bundled(
            chunk_size=self.code_chunk_size,
            chunk_overlap=self.code_chunk_overlap,
        )

    @cached_property
    def model(self) -> Embeddings:
        if self.model_instantiation_class is MultiModalEmbeddings:
            return self.model_instantiation_class(self)
        m = super().model
        assert isinstance(m, Embeddings)
        return m


@dataclass(repr=False)
class RerankerModel(BaseModel):
    # Model args w/ defaults
    model_instantiation_class: Callable = MultiModalReranker

    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)

    # If input should be preprocessed
    preprocessor: Optional[Callable] = None

    allowable_modalities = ("text", "audio", "image", "video")

    @cached_property
    def model(self) -> MultiModalReranker:
        return self.model_instantiation_class(self)  # type: ignore[return-value]
