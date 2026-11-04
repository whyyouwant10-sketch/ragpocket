"""Versioned tokenization used by both indexing and querying."""

import re
from typing import List


TOKENIZER_VERSION = "unicode-word-v1"
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Return case-folded maximal Unicode word runs, excluding a lone '_'."""

    return [token for token in (m.group(0).casefold() for m in _WORD_RE.finditer(text)) if token != "_"]
