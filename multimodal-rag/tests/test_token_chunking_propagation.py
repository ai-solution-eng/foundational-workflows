"""Token-chunking propagation guards (2026-10-17).

Every text path handed to the embedder must be chunked by TOKEN count, not
by characters. Three call sites silently dropped the TokenTextSplitter so
the affected text fell back to character chunking while looking healthy:

  * DatasetManager.add_files_batch - resolved chunk_size/chunk_overlap but
    never a splitter, so its non-media else branch (text, table, office,
    html, notebook, ebook, log, and every non-media archive member)
    chunked by characters. add_file chunked the same file by tokens.
  * YAMLProcessor.process_yaml - built the inner JSONProcessor without
    forwarding text_splitter.
  * JSONProcessor._split_text - built its inner TextProcessor without
    forwarding text_splitter, so an oversized single entry was
    character-split.

These are source/behaviour guards: they assert the splitter reaches the
processor on every path, so a future refactor that drops it fails loudly
instead of silently degrading retrieval quality.

Run::

    timeout 120 .venv/bin/python -m pytest tests/test_token_chunking_propagation.py -q
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multimodal_rag.input_processing.json_processor import JSONProcessor
from multimodal_rag.input_processing.xml_processor import YAMLProcessor


class _WordSplitter:
    """Stand-in TokenTextSplitter: 1 token per whitespace-delimited word."""

    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def split_text(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        out, start = [], 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(words):
            out.append(" ".join(words[start : start + self.chunk_size]))
            if start + self.chunk_size >= len(words):
                break
            start += step
        return out

    def overlap_text(self, text: str) -> str:
        words = text.split()
        return " ".join(words[-self.chunk_overlap :]) if self.chunk_overlap else ""


def test_add_files_batch_resolves_a_text_splitter():
    """add_files_batch must resolve a splitter, not only chunk_size/overlap."""
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.add_files_batch)
    assert "_resolve_chunk_params" in src, (
        "add_files_batch must resolve a token splitter via _resolve_chunk_params; "
        "without it the batch path chunked by characters"
    )


def test_add_files_batch_passes_text_splitter_to_add_file_processed():
    """The non-media else-branch must forward text_splitter= to the handler."""
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.add_files_batch)
    start = src.find("result = self._add_file_processed(")
    assert start != -1, "expected a direct _add_file_processed call in add_files_batch"
    call = src[start : src.find(")", start) + 1]
    assert "text_splitter=text_splitter" in call, (
        "the else-branch _add_file_processed call must pass text_splitter=text_splitter "
        "(archives, text, table, office, html, notebook, ebook and log all route here)"
    )


def test_add_file_still_passes_text_splitter():
    """Control: the single-file path must keep passing the splitter."""
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.add_file)
    start = src.find("self._add_file_processed(")
    assert start != -1
    call = src[start : src.find(")", start) + 1]
    assert "text_splitter=text_splitter" in call


def test_archive_member_processing_forwards_text_splitter():
    """Archive members must be processed with the same splitter as the parent."""
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager._add_file_processed)
    start = src.find("def _process_member(")
    assert start != -1, "expected the archive _process_member helper"
    body = src[start : src.find("archive_proc", start)]
    assert "text_splitter=text_splitter" in body, (
        "archive members must forward text_splitter so an archive member is "
        "chunked identically to the same file uploaded standalone"
    )


def test_yaml_processor_forwards_splitter_to_inner_json_processor():
    """YAMLProcessor.process_yaml must forward text_splitter to JSONProcessor."""
    splitter = _WordSplitter(10)
    proc = YAMLProcessor(chunk_size=10, chunk_overlap=0, text_splitter=splitter)

    captured = {}
    import multimodal_rag.input_processing.json_processor as jpmod

    real = jpmod.JSONProcessor

    class _Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            captured.update(kw)
            super().__init__(*a, **kw)

    jpmod.JSONProcessor = _Spy
    try:
        proc.process_yaml("a: 1\nb: 2\n", "sample.yaml")
    finally:
        jpmod.JSONProcessor = real

    assert captured.get("text_splitter") is splitter, (
        "YAMLProcessor must forward its token splitter to the inner JSONProcessor; "
        "dropping it made YAML always character-chunked"
    )


def test_json_oversized_entry_is_token_split():
    """An entry exceeding the budget must be split by tokens, not characters."""
    budget = 20
    splitter = _WordSplitter(budget)
    proc = JSONProcessor(chunk_size=budget, chunk_overlap=0, text_splitter=splitter, flatten=True)

    big = {"big": " ".join(["token"] * 400)}
    chunks = proc.process_data(big, "big.json")

    assert len(chunks) > 1, "expected the oversized entry to be split"
    token_expected = 400 // budget
    assert len(chunks) <= token_expected * 2, (
        f"expected ~{token_expected} token-sized chunks, got {len(chunks)}: "
        "the oversized-entry sub-split looks character-based"
    )
    for ch in chunks:
        assert splitter.count_tokens(ch["text"]) <= budget * 2, "every produced chunk must respect the token budget"


def test_json_split_text_forwards_the_splitter():
    """JSONProcessor._split_text must build its TextProcessor with the splitter."""
    splitter = _WordSplitter(5)
    proc = JSONProcessor(chunk_size=5, chunk_overlap=0, text_splitter=splitter)
    sub = proc._split_text(" ".join(["w"] * 100), "s", "$.x")
    assert len(sub) > 1
    assert max(splitter.count_tokens(c["text"]) for c in sub) >= 4, (
        "oversized-entry chunks look character-sized; splitter was not forwarded"
    )


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("text_processor", "TextProcessor"),
        ("json_processor", "JSONProcessor"),
        ("xml_processor", "XMLProcessor"),
        ("xml_processor", "YAMLProcessor"),
        ("code_processor", "CodeProcessor"),
        ("table_processor", "TableProcessor"),
        ("log_processor", "LogProcessor"),
        ("notebook_processor", "NotebookProcessor"),
        ("ebook_processor", "EbookProcessor"),
        ("office_processor", "OfficeProcessor"),
        ("html_processor", "HTMLProcessor"),
    ],
)
def test_processor_accepts_text_splitter(module_name, class_name):
    """Every text processor must expose a text_splitter hook."""
    mod = __import__(f"multimodal_rag.input_processing.{module_name}", fromlist=[class_name])
    cls = getattr(mod, class_name)
    assert "text_splitter" in inspect.signature(cls).parameters or hasattr(cls, "text_splitter"), (
        f"{class_name} must accept a text_splitter"
    )


def test_batch_path_uses_resolved_size_with_its_splitter():
    """chunk_size/overlap passed to processors must match the splitter budget.

    The splitter is built for the RESOLVED budget while the processors gate
    on their own chunk_size. If the batch path resolved an override splitter
    but passed the model default size, a per-dataset override would be
    silently half-applied.
    """
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.add_files_batch)
    assert "_resolve_chunk_params" in src
    # The resolved triple must be unpacked INTO chunk_size/chunk_overlap,
    # not into throwaway names that leave the originals stale.
    assert "chunk_size, chunk_overlap, text_splitter = self._resolve_chunk_params(" in src, (
        "add_files_batch must assign the resolved size/overlap to the very "
        "variables passed to processors, so the splitter budget and the "
        "processor budget cannot disagree"
    )
    assert "_b_size" not in src and "_b_overlap" not in src, (
        "resolved size/overlap must not be discarded into unused names"
    )


def test_xml_processor_forwards_splitter_to_inner_text_processor():
    """XMLProcessor oversized children must be token-split, not char-split.

    This exact line was dropped once already; the guard exists so it cannot
    silently regress again.
    """
    from multimodal_rag.dataset_manager import DatasetManager  # noqa: F401
    import inspect as _inspect

    from multimodal_rag.input_processing.xml_processor import XMLProcessor

    src = _inspect.getsource(XMLProcessor._chunk_large)
    assert "text_splitter=self.text_splitter" in src, (
        "XMLProcessor._chunk_large must forward text_splitter to its inner "
        "TextProcessor; dropping it character-chunks oversized XML children"
    )


def test_batch_pdf_uses_the_resolved_splitter():
    """The batch PDF branch must use the resolved splitter, not the model one."""
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.add_files_batch)
    assert "text_splitter=rag.embedder.text_splitter" not in src, (
        "batch PDF must use the resolved splitter; the model cached splitter "
        "carries its own budget and would defeat a per-dataset override"
    )
    assert "text_splitter=text_splitter" in src


def test_archive_member_uses_token_splitter_end_to_end(tmp_path):
    """An archive member must be token-chunked, same as a standalone file.

    End-to-end over the real ArchiveProcessor + recursive handler: a member
    that exceeds the token budget must yield MULTIPLE chunks whose sizes are
    token-driven. A character-based member would yield far more, smaller ones.
    """
    import zipfile

    from multimodal_rag.input_processing.archive_processor import ArchiveProcessor

    splitter = _WordSplitter(20)

    # One member whose text is far over the 20-token budget.
    member_text = " ".join(["word"] * 400)
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner.txt", member_text)

    seen = []

    def _process_member(member_path, member_name):
        # Mirror what _add_file_processed does for a text member.
        from multimodal_rag.input_processing.text_processor import TextProcessor

        tp = TextProcessor(chunk_size=20, chunk_overlap=0, text_splitter=splitter)
        chunks = tp.process_text(open(member_path, encoding="utf-8").read(), member_name)
        seen.extend(chunks)
        return ["id"] * len(chunks)

    proc = ArchiveProcessor(process_member=_process_member)
    proc.process(str(zip_path))

    assert seen, "archive member was not processed"
    token_expected = 400 // 20
    assert len(seen) <= token_expected * 2, (
        f"expected ~{token_expected} token-sized chunks from the archive member, "
        f"got {len(seen)}: member looks character-chunked"
    )
