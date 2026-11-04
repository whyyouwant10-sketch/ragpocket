"""Safe scalar metadata parsing and controlled search filters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable


KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
FILTER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)(\^=|~=|!=|>=|<=|=|>|<)(.+)$")
BUILTIN_FIELDS = {"path", "name", "ext", "bytes", "sha256", "mtime_ns", "language"}


class MetadataError(ValueError):
    pass


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        return parsed
    if (len(value) >= 2 and value[0] == value[-1] and value[0] in "'\""):
        return value[1:-1]
    return value


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Extract simple leading Markdown front matter while preserving lines."""

    if not (text == "---" or text.startswith("---\n")):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "---":
        return {}, text
    closing = None
    for index in range(1, len(lines)):
        if lines[index].rstrip("\n") in {"---", "..."}:
            closing = index
            break
    if closing is None:
        return {}, text

    values: dict[str, Any] = {}
    remove: set[int] = {0, closing}
    for index in range(1, closing):
        line = lines[index].rstrip("\n")
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*):[ \t]*(.*)$", line)
        if not match:
            continue
        key, raw = match.groups()
        if not raw.strip():
            # A mapping header is not a scalar and stays searchable.
            continue
        values[key] = _scalar(raw)
        remove.add(index)
    preserved = "".join("\n" if index in remove and line.endswith("\n") else "" if index in remove else line for index, line in enumerate(lines))
    return values, preserved


def validate_metadata_object(value: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise MetadataError(f"{source} must contain a JSON object")
    result: dict[str, dict[str, Any]] = {}
    for path, metadata in value.items():
        if not isinstance(path, str) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path:
            raise MetadataError(f"invalid metadata path: {path!r}")
        if not isinstance(metadata, dict):
            raise MetadataError(f"metadata for {path!r} must be an object")
        clean: dict[str, Any] = {}
        for key, item in metadata.items():
            if not isinstance(key, str) or not KEY_RE.fullmatch(key):
                raise MetadataError(f"invalid metadata key for {path!r}: {key!r}")
            if not (isinstance(item, (str, int, float, bool)) or item is None) or isinstance(item, (list, dict)):
                raise MetadataError(f"metadata value for {path!r}.{key} must be scalar")
            clean[key] = item
        result[path] = clean
    return result


def load_metadata_file(path: str) -> dict[str, dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read metadata file {path}: {exc}") from exc
    return validate_metadata_object(data, path)


@dataclass(frozen=True)
class Filter:
    field: str
    op: str
    value: str

    def as_json(self) -> dict[str, str]:
        return {"field": self.field, "op": self.op, "value": self.value}


def parse_filter(expression: str) -> Filter:
    match = FILTER_RE.fullmatch(expression)
    if not match:
        raise MetadataError(f"invalid filter expression: {expression!r}")
    field, op, value = match.groups()
    if field not in BUILTIN_FIELDS and not field.startswith("meta."):
        raise MetadataError(f"unknown filter field: {field}")
    if field == "meta.":
        raise MetadataError(f"invalid filter field: {field}")
    return Filter(field, op, value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _filter_value(raw: str, actual: Any) -> Any:
    if _is_number(actual):
        try:
            return float(raw) if "." in raw or "e" in raw.lower() else int(raw)
        except ValueError as exc:
            raise MetadataError(f"numeric filter requires a numeric value: {raw!r}") from exc
    if isinstance(actual, bool):
        if raw.lower() not in {"true", "false"}:
            raise MetadataError(f"boolean filter requires true or false: {raw!r}")
        return raw.lower() == "true"
    if actual is None:
        if raw.lower() in {"null", "none"}:
            return None
        return raw
    return raw


def matches_filters(metadata: dict[str, Any], filters: Iterable[Filter]) -> bool:
    for item in filters:
        key = item.field[5:] if item.field.startswith("meta.") else item.field
        if key not in metadata:
            return False
        actual = metadata[key]
        comparison = _filter_value(item.value, actual)
        try:
            if item.op == "=":
                matched = actual == comparison
            elif item.op == "!=":
                matched = actual != comparison
            elif item.op == "^=":
                if not isinstance(actual, str):
                    raise MetadataError(f"prefix filter requires a string field: {item.field}")
                matched = actual.startswith(item.value)
            elif item.op == "~=":
                if not isinstance(actual, str):
                    raise MetadataError(f"substring filter requires a string field: {item.field}")
                matched = item.value.casefold() in actual.casefold()
            else:
                if not _is_number(actual) or not _is_number(comparison):
                    raise MetadataError(f"numeric comparison requires a numeric field: {item.field}")
                matched = {">": actual > comparison, ">=": actual >= comparison, "<": actual < comparison, "<=": actual <= comparison}[item.op]
        except TypeError as exc:
            raise MetadataError(f"invalid comparison for {item.field}") from exc
        if not matched:
            return False
    return True
