import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multimodal_rag.input_processing.image_processor import ImageProcessor
from multimodal_rag.input_processing.json_processor import JSONProcessor
from multimodal_rag.input_processing.pdf_processor import PDFProcessor
from multimodal_rag.input_processing.table_processor import TableProcessor
from multimodal_rag.input_processing.text_processor import TextProcessor
from multimodal_rag.input_processing.video_processor import VideoProcessor
from multimodal_rag.utils.logging_utils import logging

logger = logging.getLogger(__name__)

_SUPPORTED_ARCHIVE_EXTS = frozenset({".zip", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz2", ".txz", ".rar"})
_MAX_DEPTH = 3

# ---------------------------------------------------------------------------
# Archive-bomb guards: total uncompressed budget, per-member cap, entry cap.
# The declared (uncompressed) member sizes are inspected BEFORE extracting, so
# a crafted archive cannot fill the disk / memory in the extraction step.
# ---------------------------------------------------------------------------
_ARCHIVE_MAX_TOTAL_BYTES = max(0, int(os.environ.get("ARCHIVE_MAX_TOTAL_BYTES", str(2 * 1024 * 1024 * 1024))))
_ARCHIVE_MAX_MEMBER_BYTES = max(0, int(os.environ.get("ARCHIVE_MAX_MEMBER_BYTES", str(1024 * 1024 * 1024))))
_ARCHIVE_MAX_ENTRIES = max(0, int(os.environ.get("ARCHIVE_MAX_ENTRIES", "10000")))


def _is_archive(path: str) -> bool:
    return Path(path).suffix.lower() in _SUPPORTED_ARCHIVE_EXTS


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"})
_VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"})
_AUDIO_EXTS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma"})
_TABLE_EXTS = frozenset({".csv", ".tsv", ".xlsx", ".xls", ".ods"})
_TEXT_EXTS = frozenset({".txt", ".md"})
_CODE_EXTS = frozenset(
    {
        ".py",
        ".pyw",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".java",
        ".cpp",
        ".cxx",
        ".cc",
        ".c",
        ".h",
        ".hpp",
        ".hxx",
        ".cs",
        ".rb",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".php",
        ".r",
        ".R",
        ".sh",
        ".bash",
        ".zsh",
    }
)
_OFFICE_EXTS = frozenset({".docx", ".pptx", ".odt", ".odp"})
_HTML_EXTS = frozenset({".html", ".htm"})
_XML_EXTS = frozenset({".xml"})
_YAML_EXTS = frozenset({".yaml", ".yml"})


def _classify_ext(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _TEXT_EXTS:
        return "text"
    if ext == ".json":
        return "json"
    if ext in _TABLE_EXTS:
        return "table"
    if ext in _CODE_EXTS:
        return "code"
    if ext in _OFFICE_EXTS:
        return "office"
    if ext in _HTML_EXTS:
        return "html"
    if ext in _XML_EXTS:
        return "xml"
    if ext in _YAML_EXTS:
        return "yaml"
    return "unknown"


def _tar_extract(path: str, mode: str, dest_dir: str) -> None:
    import tarfile

    with tarfile.open(path, mode) as tf:  # type: ignore[call-overload]
        tf.extractall(dest_dir, filter="data")


def _is_safe_member(dest_dir: str, member_path: str) -> bool:
    """Return True if *member_path* stays within *dest_dir* after extraction.

    Prevents Zip Slip / path traversal via crafted archive member names
    (e.g. ``../../etc/cron.d/exfil``).  Mirrors the lexical check performed
    by tarfile's ``filter="data"``.
    """
    dest = os.path.abspath(dest_dir)
    target = os.path.abspath(os.path.join(dest_dir, member_path))
    try:
        return os.path.commonpath([dest, target]) == dest
    except ValueError:
        # Different drives (Windows only) — treat as unsafe.
        return False


@dataclass
class ArchiveProcessor:
    """Extract archives and recursively process contained files.

    Supports ``.zip``, ``.tar``, ``.tar.gz`` / ``.tgz``, ``.tar.bz2`` /
    ``.tbz2``, ``.tar.xz`` / ``.txz``, and ``.rar`` archives.

    Each extracted file is dispatched to the appropriate processor
    (PDF, image, video, audio, text, JSON, table, etc.).  Nested
    archives are extracted recursively up to a configurable depth.
    """

    chunk_size: int = 8192
    chunk_overlap: int = 512
    text_splitter: Any | None = None
    max_depth: int = _MAX_DEPTH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, archive_path: str, source: str = "") -> list[dict[str, Any]]:
        """Extract *archive_path* and return all document chunks.

        Returns
        -------
        list[dict]
            Document dicts from all contained files, with ``source``
            pointing to the original archive path.
        """
        self._check_bounds(archive_path)
        archive_source = source or str(archive_path)
        extract_dir = tempfile.mkdtemp(prefix="mmrag_archive_")
        try:
            self._extract(archive_path, extract_dir)
            docs = self._process_dir(extract_dir, archive_source, depth=0)
            return docs
        except Exception as e:
            logger.error("Failed to process archive %s: %s", archive_path, e)
            return []
        finally:
            import shutil

            shutil.rmtree(extract_dir, ignore_errors=True)

    def _process_nested_archive(self, path: str, archive_source: str, depth: int) -> list[dict[str, Any]]:
        # Bounds check applies to nested archives too — otherwise a crafted
        # nested bomb could bypass the top-level guard.
        self._check_bounds(path)
        nested_dir = tempfile.mkdtemp(prefix="mmrag_nested_")
        try:
            self._extract(path, nested_dir)
            return self._process_dir(nested_dir, archive_source, depth + 1)
        except Exception as e:
            logger.warning("Failed to extract nested archive %s: %s", path, e)
            return []
        finally:
            import shutil

            shutil.rmtree(nested_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Bounds checking (archive-bomb guard)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_bounds(path: str) -> None:
        """Raise ``ValueError`` if the archive's declared sizes exceed the caps.

        The uncompressed sizes in the member headers are audited BEFORE any
        extraction so a crafted zip/tar/rar cannot fill the PVC.  Bounds are
        configured via ``ARCHIVE_MAX_TOTAL_BYTES`` / ``ARCHIVE_MAX_MEMBER_BYTES``
        / ``ARCHIVE_MAX_ENTRIES`` (0 disables a check).
        """
        if _ARCHIVE_MAX_TOTAL_BYTES <= 0 and _ARCHIVE_MAX_MEMBER_BYTES <= 0 and _ARCHIVE_MAX_ENTRIES <= 0:
            return
        ext = Path(path).suffix.lower()
        total = 0
        count = 0

        def _account(size: int) -> None:
            nonlocal total, count
            count += 1
            if _ARCHIVE_MAX_ENTRIES > 0 and count > _ARCHIVE_MAX_ENTRIES:
                raise ValueError(
                    f"Archive {path} contains more than ARCHIVE_MAX_ENTRIES ({_ARCHIVE_MAX_ENTRIES}) entries"
                )
            if _ARCHIVE_MAX_MEMBER_BYTES > 0 and size > _ARCHIVE_MAX_MEMBER_BYTES:
                raise ValueError(f"Archive member exceeds ARCHIVE_MAX_MEMBER_BYTES ({size} bytes)")
            total += size
            if _ARCHIVE_MAX_TOTAL_BYTES > 0 and total > _ARCHIVE_MAX_TOTAL_BYTES:
                raise ValueError(
                    f"Archive total exceeds ARCHIVE_MAX_TOTAL_BYTES ({_ARCHIVE_MAX_TOTAL_BYTES} bytes)"
                )

        if ext == ".zip":
            import zipfile

            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    _account(info.file_size)
        elif ext == ".rar":
            # Pre-scan only when the pure-python reader is available; the
            # unrar-CLI fallback rejects ".." itself but has no size guard.
            try:
                import rarfile

                with rarfile.RarFile(path) as rf:
                    for member in rf.infolist():
                        _account(getattr(member, "file_size", 0) or 0)
            except Exception:
                logger.warning("rar bounds check unavailable (%s) — extracting without size guard", path)
        else:
            # tar / tar.gz / tar.bz2 / tar.xz (+ .tgz/.tbz2/.txz)
            import tarfile

            mode: Any = {"gz": "r:gz", "tgz": "r:gz", "bz2": "r:bz2", "tbz2": "r:bz2", "xz": "r:xz", "txz": "r:xz"}.get(
                ext.lstrip("."), "r:"
            )
            with tarfile.open(path, mode) as tf:
                for member in tf.getmembers():
                    _account(member.size)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(path: str, dest_dir: str) -> None:
        ext = Path(path).suffix.lower()
        stem = Path(path).stem.lower()

        if ext == ".zip" or Path(path).suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(path, "r") as zf:
                for member in zf.infolist():
                    if not _is_safe_member(dest_dir, member.filename):
                        logger.warning("Skipping unsafe zip member: %s", member.filename)
                        continue
                    zf.extract(member, dest_dir)
        elif ext == ".rar":
            try:
                import rarfile

                with rarfile.RarFile(path) as rf:
                    for member in rf.infolist():
                        fname = getattr(member, "filename", "") or ""
                        if not _is_safe_member(dest_dir, fname):
                            logger.warning("Skipping unsafe rar member: %s", fname)
                            continue
                        rf.extract(member, dest_dir)
            except Exception:
                logger.warning("rarfile extraction failed for %s, trying unrar", path)
                import subprocess as sp

                # unrar ≥5.x refuses paths containing ".." on its own.
                sp.run(
                    ["unrar", "x", "-y", path, dest_dir + "/"],
                    capture_output=True,
                    timeout=120,
                )
        elif ext in (".gz", ".bz2", ".xz") or stem.endswith(".tar") or ext in (".tgz", ".tbz2", ".txz"):
            if ext in (".gz", ".tgz"):
                _tar_extract(path, "r:gz", dest_dir)
            elif ext in (".bz2", ".tbz2"):
                _tar_extract(path, "r:bz2", dest_dir)
            elif ext in (".xz", ".txz"):
                _tar_extract(path, "r:xz", dest_dir)
            else:
                _tar_extract(path, "r:", dest_dir)
        elif ext == ".tar":
            _tar_extract(path, "r:", dest_dir)
        else:
            raise ValueError(f"Unsupported archive format: {ext}")

    # ------------------------------------------------------------------
    # Recursive processing
    # ------------------------------------------------------------------

    def _process_dir(self, directory: str, archive_source: str, depth: int) -> list[dict[str, Any]]:
        if depth > self.max_depth:
            logger.warning("Max archive depth %s reached, skipping nested content", self.max_depth)
            return []

        docs: list[dict[str, Any]] = []
        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                docs.extend(self._process_dir(full_path, archive_source, depth))
            elif os.path.isfile(full_path):
                if _is_archive(full_path):
                    nested = self._process_nested_archive(full_path, archive_source, depth)
                    docs.extend(nested)
                else:
                    file_docs = self._process_single_file(full_path, archive_source)
                    docs.extend(file_docs)
        return docs

    def _process_single_file(self, path: str, archive_source: str) -> list[dict[str, Any]]:
        file_type = _classify_ext(path)
        try:
            if file_type == "pdf":
                pdf_proc = PDFProcessor()
                chunks = pdf_proc.extract_chunks(path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                for c in chunks:
                    c["source"] = c.get("source", archive_source)
                return chunks
            elif file_type == "image":
                img_proc = ImageProcessor()
                doc = img_proc.process(path)
                doc["source"] = archive_source
                return [doc]
            elif file_type == "video":
                vid_proc = VideoProcessor()
                docs = vid_proc.process(path)
                for d in docs:
                    d["source"] = archive_source
                return docs
            elif file_type == "audio":
                with open(path, "rb") as f:
                    raw = f.read()
                import base64
                import mimetypes

                mime = mimetypes.guess_type(path)[0] or "audio/mpeg"
                b64 = base64.b64encode(raw).decode("utf-8")
                return [
                    {
                        "text": f"[Audio: {Path(path).name}]",
                        "audio": f"data:{mime};base64,{b64}",
                        "source": archive_source,
                    }
                ]
            elif file_type == "json":
                json_proc = JSONProcessor(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                chunks = json_proc.process(path)
                for c in chunks:
                    c["source"] = archive_source
                return chunks
            elif file_type == "table":
                table_proc = TableProcessor(chunk_size=self.chunk_size, text_splitter=self.text_splitter)
                chunks = table_proc.process(path)
                for c in chunks:
                    c["source"] = archive_source
                return chunks
            elif file_type == "text":
                text_proc = TextProcessor(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                chunks = text_proc.process(path)
                for c in chunks:
                    c["source"] = archive_source
                return chunks
            else:
                logger.debug("Skipping unknown file type in archive: %s", path)
                return []
        except Exception as e:
            logger.warning("Failed to process %s in archive: %s", Path(path).name, e)
            return []
