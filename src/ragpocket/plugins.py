"""The deliberately small optional embedding-provider boundary."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """A provider supplied explicitly by the caller, never auto-discovered."""

    name: str
    model: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...

    def embed_query(self, text: str) -> Sequence[float]:
        ...
