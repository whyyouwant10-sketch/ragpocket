"""BM25 and optional hybrid search over a RAGPocket SQLite index."""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

from .chunking import citation
from .index import IndexError_, _checked_query_vector, _provider_instance, connect_index
from .metadata import Filter, matches_filters
from .tokenize import tokenize


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _load_vectors(connection: sqlite3.Connection, provider: Any) -> dict[str, list[float]]:
    rows = connection.execute("SELECT chunk_id, provider, model, dimensions, vector_blob FROM embeddings WHERE provider=? AND model=? AND dimensions=?", (str(provider.name), str(provider.model), provider.dimensions)).fetchall()
    vectors: dict[str, list[float]] = {}
    for row in rows:
        try:
            vector = json.loads(bytes(row["vector_blob"]).decode("utf-8"))
        except (ValueError, UnicodeError, TypeError) as exc:
            raise IndexError_("index contains an invalid embedding vector") from exc
        if len(vector) != provider.dimensions or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in vector):
            raise IndexError_("index contains an invalid embedding vector")
        vectors[row["chunk_id"]] = [float(value) for value in vector]
    return vectors


def search_index(db: str, query: str, filters: list[Filter] | None = None, top_k: int = 8, rank: str = "bm25", snippet_chars: int = 600, embedding: str | None = None, explain: bool = False) -> dict[str, Any]:
    if top_k <= 0:
        raise IndexError_("top-k must be greater than zero")
    if snippet_chars <= 0:
        raise IndexError_("snippet-chars must be greater than zero")
    terms = tokenize(query)
    if not terms:
        raise IndexError_("query contains no searchable terms")
    unique_terms = list(dict.fromkeys(terms))
    connection = connect_index(db, read_only=True)
    try:
        rows = connection.execute("SELECT c.*, d.path, d.metadata_json FROM chunks c JOIN documents d ON d.document_id=c.document_id ORDER BY d.path, c.sequence, c.chunk_id").fetchall()
        candidates = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if matches_filters(metadata, filters or []):
                candidates.append((row, metadata))
        n = len(candidates)
        if not n:
            return {"query": query, "ranker": rank, "filters": [item.as_json() for item in (filters or [])], "results": []}
        candidate_ids = {row["chunk_id"] for row, _ in candidates}
        frequencies: dict[str, dict[str, int]] = {term: {} for term in unique_terms}
        term_ids = {}
        for term in unique_terms:
            found = connection.execute("SELECT term_id FROM terms WHERE term=?", (term,)).fetchone()
            if found:
                term_ids[term] = found[0]
        if term_ids:
            placeholders = ",".join("?" for _ in term_ids)
            posting_rows = connection.execute(f"SELECT term_id, chunk_id, tf FROM postings WHERE term_id IN ({placeholders})", tuple(term_ids.values())).fetchall()
            reverse = {value: key for key, value in term_ids.items()}
            for item in posting_rows:
                if item["chunk_id"] in candidate_ids:
                    frequencies[reverse[item["term_id"]]][item["chunk_id"]] = item["tf"]
        avgdl = sum(row["token_count"] for row, _ in candidates) / n
        bm25_scores: dict[str, float] = {}
        contributions: dict[str, dict[str, float]] = {}
        for row, _ in candidates:
            score = 0.0
            details: dict[str, float] = {}
            for term in unique_terms:
                df = len(frequencies[term])
                tf = frequencies[term].get(row["chunk_id"], 0)
                if not tf:
                    continue
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                contribution = idf * (tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * row["token_count"] / avgdl))
                score += contribution
                details[term] = contribution
            bm25_scores[row["chunk_id"]] = score
            contributions[row["chunk_id"]] = details

        final_scores = bm25_scores.copy()
        if rank not in {"bm25", "hybrid"}:
            raise IndexError_(f"unsupported ranker: {rank}")
        cosine_scores: dict[str, float] = {}
        if rank == "hybrid":
            if not embedding:
                raise IndexError_("hybrid ranking requires --embedding")
            provider = _provider_instance(embedding)
            query_vector = _checked_query_vector(provider, query)
            vectors = _load_vectors(connection, provider)
            if any(chunk_id not in vectors for chunk_id in candidate_ids):
                raise IndexError_("embedding vectors are missing for the selected provider")
            cosine_scores = {chunk_id: _cosine(vectors[chunk_id], query_vector) for chunk_id in candidate_ids}
            bm_min, bm_max = min(bm25_scores.values()), max(bm25_scores.values())
            cos_min, cos_max = min(cosine_scores.values()), max(cosine_scores.values())
            for chunk_id in candidate_ids:
                bm_norm = (bm25_scores[chunk_id] - bm_min) / (bm_max - bm_min) if bm_max != bm_min else 0.0
                cos_norm = (cosine_scores[chunk_id] - cos_min) / (cos_max - cos_min) if cos_max != cos_min else 0.0
                final_scores[chunk_id] = 0.7 * bm_norm + 0.3 * cos_norm

        # A lexical result must contain at least one matching term.  Keeping
        # zero-score rows would make an absent query return arbitrary context
        # merely because the filter had candidates.
        candidates = [item for item in candidates if final_scores[item[0]["chunk_id"]] > 0]
        if not candidates:
            return {"query": query, "ranker": rank, "filters": [item.as_json() for item in (filters or [])], "results": []}
        ordered = sorted(candidates, key=lambda item: (-final_scores[item[0]["chunk_id"]], item[0]["path"], item[0]["sequence"], item[0]["chunk_id"]))[:top_k]
        output_results = []
        for rank_number, (row, metadata) in enumerate(ordered, 1):
            snippet = row["text"].strip()
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars].rstrip()
            item = {"rank": rank_number, "score": round(final_scores[row["chunk_id"]], 6), "citation": citation(row["path"], row["line_start"], row["line_end"], row["sequence"]), "chunk_id": row["chunk_id"], "path": row["path"], "line_start": row["line_start"], "line_end": row["line_end"], "snippet": snippet, "metadata": metadata}
            if explain:
                item["explain"] = {"terms": unique_terms, "contributions": {key: round(value, 6) for key, value in contributions[row["chunk_id"]].items()}}
            output_results.append(item)
        return {"query": query, "ranker": rank, "filters": [item.as_json() for item in (filters or [])], "results": output_results}
    finally:
        connection.close()
