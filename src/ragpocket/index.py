"""Folder discovery and atomic SQLite index construction."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .chunking import CHUNKER_VERSION, chunk_text
from .metadata import MetadataError, load_metadata_file, parse_front_matter
from .plugins import EmbeddingProvider
from .tokenize import TOKENIZER_VERSION, tokenize


SCHEMA_VERSION = "1"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".yaml", ".yml", ".toml",
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h",
    ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".go", ".rs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".sql", ".xml", ".html", ".htm", ".css",
    ".scss", ".ini", ".cfg", ".conf", ".csv", ".log", ".lock",
}
LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".c": "c",
    ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hh": "cpp",
    ".hpp": "cpp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".sql": "sql",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".markdown": "markdown", ".rst": "restructuredtext",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
}


class IndexError_(ValueError):
    """An expected user-facing indexing error."""


def _schema(connection: sqlite3.Connection, with_embeddings: bool = False) -> None:
    connection.executescript(
        """
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            ext TEXT NOT NULL,
            language TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL,
            normalized_chars INTEGER NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            UNIQUE(document_id, sequence)
        );
        CREATE TABLE terms (term_id INTEGER PRIMARY KEY, term TEXT UNIQUE NOT NULL);
        CREATE TABLE postings (
            term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            tf INTEGER NOT NULL,
            positions_json TEXT,
            PRIMARY KEY(term_id, chunk_id)
        );
        CREATE INDEX postings_chunk_idx ON postings(chunk_id);
        CREATE INDEX chunks_document_idx ON chunks(document_id);
        """
    )
    if with_embeddings:
        connection.execute(
            """CREATE TABLE embeddings (
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY(chunk_id, provider, model)
            )"""
        )


def connect_index(path: str | os.PathLike[str], read_only: bool = False) -> sqlite3.Connection:
    """Open and validate an index rather than silently guessing its format."""

    db_path = Path(path)
    if not db_path.is_file():
        raise IndexError_(f"index does not exist: {db_path}")
    if read_only:
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        values = dict(connection.execute("SELECT key, value FROM index_meta").fetchall())
    except sqlite3.Error as exc:
        connection.close()
        raise IndexError_(f"not a RAGPocket index: {db_path}") from exc
    if values.get("schema_version") != SCHEMA_VERSION or values.get("chunker_version") != CHUNKER_VERSION or values.get("tokenizer_version") != TOKENIZER_VERSION:
        connection.close()
        raise IndexError_(f"unsupported RAGPocket index schema: {db_path}")
    return connection


def _path_matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatchcase(normalized, pattern) or PurePosixPath(normalized).match(pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(normalized, pattern[3:]):
            return True
    return False


def _is_hidden(relative: str) -> bool:
    return any(part.startswith(".") for part in PurePosixPath(relative).parts)


def _default_excluded(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if any(part in {".git", ".ragpocket", "__pycache__", ".venv", "venv", "env", "virtualenv"} for part in parts[:-1]):
        return True
    return False


def discover_files(root: Path, includes: list[str], excludes: list[str], max_bytes: int, follow_symlinks: bool) -> tuple[list[Path], list[str]]:
    """Return sorted candidates and non-fatal diagnostics."""

    diagnostics: list[str] = []
    candidates: list[Path] = []
    root_real = root.resolve()
    for current, dirs, files in os.walk(root, topdown=True, followlinks=follow_symlinks):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in dirs:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if _default_excluded(relative) or (_is_hidden(relative) and not _path_matches(relative, includes)) or _path_matches(relative, excludes):
                continue
            if candidate.is_symlink() and follow_symlinks:
                try:
                    candidate.resolve().relative_to(root_real)
                except ValueError:
                    diagnostics.append(f"skipped symlink outside folder: {relative}")
                    continue
            elif candidate.is_symlink():
                diagnostics.append(f"skipped symlink: {relative}")
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in files:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if _default_excluded(relative) or (_is_hidden(relative) and not _path_matches(relative, includes)) or _path_matches(relative, excludes):
                continue
            if candidate.is_symlink():
                if not follow_symlinks:
                    diagnostics.append(f"skipped symlink: {relative}")
                    continue
                try:
                    candidate.resolve().relative_to(root_real)
                except ValueError:
                    diagnostics.append(f"skipped symlink outside folder: {relative}")
                    continue
            if not candidate.is_file():
                diagnostics.append(f"skipped special file: {relative}")
                continue
            selected = Path(name).suffix.lower() in TEXT_EXTENSIONS or _path_matches(relative, includes)
            if not selected:
                continue
            try:
                size = candidate.stat().st_size
            except OSError as exc:
                diagnostics.append(f"failed {relative}: {exc}")
                continue
            if size > max_bytes:
                diagnostics.append(f"skipped oversized file: {relative} ({size} bytes)")
                continue
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.relative_to(root).as_posix())
    return candidates, diagnostics


def _read_source(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IndexError_(f"invalid UTF-8: {path}") from exc
    return raw, decoded.replace("\r\n", "\n").replace("\r", "\n")


def _validate_user_metadata(metadata: dict[str, Any], path: str) -> None:
    for key in metadata:
        if key in {"path", "name", "ext", "bytes", "sha256", "mtime_ns", "language"}:
            raise IndexError_(f"metadata key collides with built-in field for {path}: {key}")


def _provider_instance(spec: str) -> EmbeddingProvider:
    import importlib

    if ":" not in spec:
        raise IndexError_("embedding must be MODULE:OBJECT")
    module_name, object_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        provider = module
        for part in object_name.split("."):
            provider = getattr(provider, part)
        provider = provider() if callable(provider) else provider
    except Exception as exc:  # plugin exceptions are user-facing command errors
        raise IndexError_(f"cannot load embedding provider {spec}: {exc}") from exc
    for field in ("name", "model", "dimensions", "embed_documents", "embed_query"):
        if not hasattr(provider, field):
            raise IndexError_(f"embedding provider is missing {field}")
    if not isinstance(provider.dimensions, int) or provider.dimensions <= 0:
        raise IndexError_("embedding provider dimensions must be a positive integer")
    return provider


def _checked_vectors(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    import math

    try:
        vectors = provider.embed_documents(texts)
    except Exception as exc:
        raise IndexError_(f"embedding provider failed: {exc}") from exc
    try:
        vector_count = len(vectors)
    except TypeError as exc:
        raise IndexError_("embedding provider returned a non-sized vector collection") from exc
    if vector_count != len(texts):
        raise IndexError_("embedding provider returned the wrong number of vectors")
    checked: list[list[float]] = []
    for vector in vectors:
        try:
            valid = len(vector) == provider.dimensions and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise IndexError_("embedding provider returned an invalid vector")
        checked.append([float(value) for value in vector])
    return checked


def _checked_query_vector(provider: EmbeddingProvider, text: str) -> list[float]:
    import math

    try:
        vector = provider.embed_query(text)
        valid = len(vector) == provider.dimensions and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector)
    except (Exception,) as exc:
        raise IndexError_(f"embedding provider failed: {exc}") from exc
    if not valid:
        raise IndexError_("embedding provider returned an invalid query vector")
    return [float(value) for value in vector]


def _config(chunk_chars: int, overlap_chars: int, max_bytes: int, follow_symlinks: bool, includes: list[str], excludes: list[str], embedding: str | None) -> str:
    return json.dumps({"chunk_chars": chunk_chars, "overlap_chars": overlap_chars, "max_bytes": max_bytes, "follow_symlinks": follow_symlinks, "include": includes, "exclude": excludes, "embedding": embedding}, sort_keys=True, separators=(",", ":"))


def build_index(
    folder: str | os.PathLike[str],
    db: str | os.PathLike[str] | None = None,
    *,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    follow_symlinks: bool = False,
    embedding: str | None = None,
    metadata_file: str | os.PathLike[str] | None = None,
) -> dict[str, int]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise IndexError_(f"folder is not a directory: {folder}")
    if max_bytes <= 0:
        raise IndexError_("max_bytes must be greater than zero")
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise IndexError_("chunk_chars must be > 0 and 0 <= overlap_chars < chunk_chars")
    includes = list(includes or [])
    excludes = list(excludes or [])
    db_path = Path(db) if db is not None else root / ".ragpocket" / "index.sqlite3"
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    source_metadata: dict[str, dict[str, Any]] = {}
    if metadata_file:
        source_metadata = load_metadata_file(str(metadata_file))
    candidates, diagnostics = discover_files(root, includes, excludes, max_bytes, follow_symlinks)
    candidate_paths = {path.relative_to(root).as_posix() for path in candidates}
    for path in sorted(set(source_metadata) - candidate_paths):
        diagnostics.append(f"metadata path not indexed: {path}")

    old_docs: dict[str, sqlite3.Row] = {}
    old_config = None
    if db_path.is_file():
        try:
            old_connection = connect_index(db_path, read_only=True)
            old_config = old_connection.execute("SELECT value FROM index_meta WHERE key='config'").fetchone()[0]
            old_docs = {row["path"]: row for row in old_connection.execute("SELECT path, sha256 FROM documents")}
            old_connection.close()
        except IndexError_:
            old_docs = {}

    config = _config(chunk_chars, overlap_chars, max_bytes, follow_symlinks, includes, excludes, embedding)
    provider = _provider_instance(embedding) if embedding else None
    counts = {"discovered": len(candidates), "indexed": 0, "reused": 0, "skipped": 0, "failed": 0}
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{db_path.name}.", suffix=".tmp", dir=str(db_path.parent))
    os.close(temp_fd)
    temporary = Path(temp_name)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(temporary))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        _schema(connection, with_embeddings=provider is not None)
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        meta_rows = {
            "schema_version": SCHEMA_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "root_path": str(root),
            "config": config,
            "created_at": now,
        }
        connection.executemany("INSERT INTO index_meta(key, value) VALUES (?, ?)", meta_rows.items())
        all_embedding_texts: list[str] = []
        all_embedding_ids: list[str] = []
        for candidate in candidates:
            relative = candidate.relative_to(root).as_posix()
            try:
                raw, text = _read_source(candidate)
                source_hash = hashlib.sha256(raw).hexdigest()
                user_meta, indexed_text = parse_front_matter(text)
                explicit = source_metadata.get(relative, {})
                user_meta.update(explicit)
                _validate_user_metadata(user_meta, relative)
                stat = candidate.stat()
                ext = candidate.suffix.lower()
                builtins = {"path": relative, "name": candidate.name, "ext": ext, "bytes": len(raw), "sha256": source_hash, "mtime_ns": stat.st_mtime_ns, "language": LANGUAGES.get(ext, "unknown")}
                document_metadata = {**builtins, **user_meta}
                cursor = connection.execute(
                    "INSERT INTO documents(path,name,ext,language,bytes,sha256,mtime_ns,normalized_chars,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
                    (relative, candidate.name, ext, builtins["language"], len(raw), source_hash, stat.st_mtime_ns, len(indexed_text), json.dumps(document_metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":"))),
                )
                document_id = cursor.lastrowid
                chunks = chunk_text(indexed_text, source_hash, chunk_chars, overlap_chars)
                for item in chunks:
                    tokens = tokenize(item.text)
                    connection.execute(
                        "INSERT INTO chunks(chunk_id,document_id,sequence,start_offset,end_offset,line_start,line_end,text,token_count) VALUES(?,?,?,?,?,?,?,?,?)",
                        (item.chunk_id, document_id, item.sequence, item.start, item.end, item.line_start, item.line_end, item.text, len(tokens)),
                    )
                    positions: dict[str, list[int]] = {}
                    for position, token in enumerate(tokens):
                        positions.setdefault(token, []).append(position)
                    for term, locations in positions.items():
                        connection.execute("INSERT OR IGNORE INTO terms(term) VALUES(?)", (term,))
                        term_id = connection.execute("SELECT term_id FROM terms WHERE term=?", (term,)).fetchone()[0]
                        connection.execute("INSERT INTO postings(term_id,chunk_id,tf,positions_json) VALUES(?,?,?,?)", (term_id, item.chunk_id, len(locations), json.dumps(locations, separators=(",", ":"))))
                    if provider:
                        all_embedding_ids.append(item.chunk_id)
                        all_embedding_texts.append(item.text)
                counts["reused" if old_config == config and relative in old_docs and old_docs[relative]["sha256"] == source_hash else "indexed"] += 1
            except (OSError, UnicodeError, IndexError_, MetadataError, ValueError) as exc:
                counts["failed"] += 1
                diagnostics.append(f"failed {relative}: {exc}")
        if provider:
            vectors = _checked_vectors(provider, all_embedding_texts)
            for chunk_id, vector in zip(all_embedding_ids, vectors):
                connection.execute("INSERT INTO embeddings(chunk_id,provider,model,dimensions,vector_blob) VALUES(?,?,?,?,?)", (chunk_id, str(provider.name), str(provider.model), provider.dimensions, json.dumps(vector, separators=(",", ":")).encode("utf-8")))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.close()
        connection = None
        os.replace(temporary, db_path)
    except Exception:
        if connection is not None:
            connection.rollback()
            connection.close()
        temporary.unlink(missing_ok=True)
        raise
    for diagnostic in diagnostics:
        print(diagnostic, file=__import__("sys").stderr)
    counts["skipped"] = sum(1 for item in diagnostics if item.startswith("skipped") or item.startswith("metadata path"))
    return counts
