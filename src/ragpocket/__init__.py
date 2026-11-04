"""RAGPocket: deterministic local retrieval over a folder of text files."""

__version__ = "0.1.0"

from .plugins import EmbeddingProvider

__all__ = ["EmbeddingProvider", "__version__"]
