"""The installed ``ragpocket`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .index import IndexError_, build_index
from .metadata import MetadataError, parse_filter
from .search import search_index


def _flatten(values: list[list[str]] | None) -> list[str]:
    return [item for group in (values or []) for item in group]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragpocket", description="Deterministic local folder retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="build a SQLite index")
    index.add_argument("folder")
    index.add_argument("--db")
    index.add_argument("--include", action="append", nargs="+", default=[])
    index.add_argument("--exclude", action="append", nargs="+", default=[])
    index.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    index.add_argument("--chunk-chars", type=int, default=1200)
    index.add_argument("--overlap-chars", type=int, default=200)
    index.add_argument("--follow-symlinks", action="store_true")
    index.add_argument("--embedding")
    index.add_argument("--metadata-file")

    search = commands.add_parser("search", help="search an SQLite index")
    search.add_argument("query")
    search.add_argument("--db", default=str(Path.cwd() / ".ragpocket" / "index.sqlite3"))
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--where", action="append", default=[])
    search.add_argument("--format", choices=("text", "json"), default="text")
    search.add_argument("--rank", choices=("bm25", "hybrid"), default="bm25")
    search.add_argument("--embedding")
    search.add_argument("--snippet-chars", type=int, default=600)
    search.add_argument("--explain", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "index":
            counts = build_index(args.folder, args.db, includes=_flatten(args.include), excludes=_flatten(args.exclude), max_bytes=args.max_bytes, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars, follow_symlinks=args.follow_symlinks, embedding=args.embedding, metadata_file=args.metadata_file)
            print(json.dumps(counts, sort_keys=True))
            return 1 if counts["skipped"] or counts["failed"] else 0
        filters = [parse_filter(expression) for expression in args.where]
        result = search_index(args.db, args.query, filters=filters, top_k=args.top_k, rank=args.rank, snippet_chars=args.snippet_chars, embedding=args.embedding, explain=args.explain)
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, sort_keys=False))
        else:
            for item in result["results"]:
                print(f"{item['rank']}. {item['score']:.6f} {item['citation']} {item['path']}")
                print(f"   {item['snippet']}")
        return 0
    except (IndexError_, MetadataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
