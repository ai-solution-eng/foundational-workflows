import importlib
from pathlib import Path
from typing import Optional

from .logging_utils import logging

logger = logging.getLogger(__name__)

__all__ = ["TokenTextSplitter"]


class TokenTextSplitter:
    """Token-count-aware text splitter using the standalone HuggingFace
    ``tokenizers`` library (Rust-based, CPU-only — no PyTorch needed).

    Designed to be embedded in the Docker image so no runtime download is
    required.  When the tokenizer file is not found, ``from_bundled()``
    returns ``None`` and callers fall back to character-based chunking.

    Parameters
    ----------
    tokenizer_path:
        Path to a ``tokenizer.json`` file on disk.
    chunk_size:
        Target number of tokens per chunk.
    chunk_overlap:
        Number of overlap tokens between consecutive chunks.
    """

    def __init__(self, tokenizer_path: str, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tok = _load_tokenizer(tokenizer_path)
        self._tokenizer_path = tokenizer_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._tok.encode(text).ids)

    def split_text(self, text: str) -> list[str]:
        """Split *text* into chunks of at most *chunk_size* tokens.

        Overlap is applied at token boundaries.  If the final chunk would
        contain fewer than 5 % *chunk_size* **net-new** tokens (tokens not
        already present in the previous chunk via overlap), those few tokens
        are appended to the previous chunk instead, avoiding a tiny tail.
        """
        if not text:
            return []
        ids = self._tok.encode(text).ids
        if len(ids) <= self.chunk_size:
            return [text]

        min_new = max(self.chunk_size // 20, 1)

        chunks: list[str] = []
        start = 0
        prev_start = 0
        prev_end = 0
        while start < len(ids):
            end = min(start + self.chunk_size, len(ids))

            if chunks and end >= len(ids):
                new_content = len(ids) - prev_end
                if new_content < min_new:
                    chunks[-1] = self._tok.decode(ids[prev_start:end])
                    break

            chunks.append(self._tok.decode(ids[start:end]))
            if end >= len(ids):
                break
            prev_start = start
            prev_end = end
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks

    def merge_until_budget(self, texts: list[str]) -> list[list[str]]:
        """Merge text fragments into groups that fit within *chunk_size* tokens.

        This is the core method that processors call instead of the old
        character-count merge pattern::

            if len(current) + len(next) > chunk_size:   # old
            if splitter.count_tokens(current) + splitter.count_tokens(next) > chunk_size:  # new

        Returns a list of groups (each group is a list of text fragments)
        so that callers can attach per-fragment metadata (images, sources,
        page numbers) to the merged result.
        """
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_tokens = 0

        for text in texts:
            n = self.count_tokens(text)

            if current_tokens + n > self.chunk_size and current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0

                # Carry overlap from the last fragment of the previous group
                if self.chunk_overlap > 0 and groups:
                    prev_text = groups[-1][-1]
                    carry_ids = self._tok.encode(prev_text).ids[-self.chunk_overlap :]
                    carry = self._tok.decode(carry_ids)
                    if carry.strip():
                        current_group.append(carry)
                        current_tokens = self.count_tokens(carry)

            current_group.append(text)
            current_tokens += n

        if current_group:
            groups.append(current_group)

        return groups

    def overlap_text(self, text: str) -> str:
        """Return the last *chunk_overlap* tokens of *text* decoded back to text.

        Call this when a single oversized fragment needs to carry overlap
        into the next chunk.
        """
        if self.chunk_overlap <= 0 or not text:
            return ""
        ids = self._tok.encode(text).ids
        if len(ids) <= self.chunk_overlap:
            return text
        carry_ids = ids[-self.chunk_overlap :]
        return self._tok.decode(carry_ids).strip()

    # ------------------------------------------------------------------
    # Classmethod helper for bundled tokenizer
    # ------------------------------------------------------------------

    @classmethod
    def from_bundled(
        cls,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer_rel: str = "tokenizer.json",
    ) -> Optional["TokenTextSplitter"]:
        """Load the tokenizer bundled with the application.

        Searches upward from the ``utils/`` directory for *tokenizer_rel*.
        Returns ``None`` if the file is not found (callers fall back to
        character-based chunking).
        """
        path = _find_bundled_tokenizer(tokenizer_rel)
        if path is None:
            return None
        return cls(str(path), chunk_size, chunk_overlap)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _load_tokenizer(path: str):
    """Import ``tokenizers`` and load the tokenizer file."""
    try:
        mod = importlib.import_module("tokenizers")
        return mod.Tokenizer.from_file(path)
    except Exception as exc:
        logger.warning("Failed to load tokenizer from %s: %s", path, exc)
        raise


_TOKENIZER_CACHE: dict[str, Path | None] = {}


def _find_bundled_tokenizer(rel: str) -> Path | None:
    """Search upward from this file's directory for *rel*."""
    if rel in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[rel]

    # Walk up from the utils directory
    start = Path(__file__).resolve().parent  # multimodal_rag/utils/
    for parent in [start, start.parent, start.parent.parent]:
        candidate = parent / rel
        if candidate.exists():
            _TOKENIZER_CACHE[rel] = candidate
            return candidate

    _TOKENIZER_CACHE[rel] = None
    return None
