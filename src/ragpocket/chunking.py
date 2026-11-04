"""Deterministic, line-aware chunking and citation helpers."""

import hashlib
from dataclasses import dataclass


CHUNKER_VERSION = "line-aware-v1"


@dataclass(frozen=True)
class Chunk:
    sequence: int
    start: int
    end: int
    line_start: int
    line_end: int
    text: str
    chunk_id: str


def _line_start(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _line_end(text: str, end: int) -> int:
    if end <= 0:
        return 1
    newlines = text.count("\n", 0, end)
    if end > 0 and text[end - 1 : end] != "\n":
        return newlines + 1
    return max(1, newlines)


def canonical_chunk_id(document_sha256: str, start: int, end: int) -> str:
    preimage = f"chunk-v1\0{document_sha256}\0{start}\0{end}".encode("utf-8")
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def chunk_text(text: str, document_sha256: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    """Split text using the specification's code-point and line rules."""

    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < chunk_chars")
    if not text:
        return []

    result: list[Chunk] = []
    start = 0
    sequence = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + chunk_chars)
        end = hard_end
        # A newline in the final 20% is preferred.  The newline belongs to
        # the preceding line, so include it in this chunk.
        preferred_from = start + ((hard_end - start) * 4 // 5)
        newline = text.rfind("\n", preferred_from, hard_end)
        if newline >= preferred_from:
            end = newline + 1

        chunk_id = canonical_chunk_id(document_sha256, start, end)
        result.append(
            Chunk(
                sequence=sequence,
                start=start,
                end=end,
                line_start=_line_start(text, start),
                line_end=_line_end(text, end),
                text=text[start:end],
                chunk_id=chunk_id,
            )
        )
        sequence += 1
        if end >= length:
            break

        next_start = max(start + 1, end - overlap_chars)
        # If the overlap starts in the middle of a line, use the next line
        # start when it is still inside the overlap.  This makes starts stable
        # and avoids repeatedly beginning in the same line.
        if next_start < end and text[next_start - 1 : next_start] != "\n":
            boundary = text.find("\n", next_start, end)
            if boundary != -1:
                next_start = boundary + 1
        if next_start <= start:
            next_start = end
        start = next_start
    return result


def display_path(path: str) -> str:
    """Escape path characters that have meaning in a citation."""

    return path.replace("\\", "\\\\").replace("]", "\\]").replace("|", "\\|")


def citation(path: str, line_start: int, line_end: int, sequence: int) -> str:
    return f"[[{display_path(path)}#L{line_start}-L{line_end}|c{sequence:04d}]]"
