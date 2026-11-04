# RAGPocket

RAGPocket is a local-first Python command-line tool for indexing a folder of
text documents and retrieving deterministic BM25-ranked chunks with citations.
It is designed to make a small, inspectable SQLite index that can provide
relevant context to another tool or answer generator. RAGPocket retrieves
context; it does not generate answers.

## Install

RAGPocket requires Python 3.10 or newer and has no runtime dependencies outside
the Python standard library and SQLite.

```console
$ python -m pip install ragpocket
```

For a checkout, use an editable install:

```console
$ python -m pip install --editable .
```

The installation provides the `ragpocket` command. To verify it:

```console
$ ragpocket --help
```

## Quick start

Index a local folder. The default database is created at
`FOLDER/.ragpocket/index.sqlite3`.

```console
$ ragpocket index ./handbook
{"discovered": 12, "failed": 0, "indexed": 12, "reused": 0, "skipped": 0}
```

Search the resulting index:

```console
$ ragpocket search "retention policy" \
    --db ./handbook/.ragpocket/index.sqlite3
1. 2.432314 [[docs/retention.md#L1-L3|c0000]] docs/retention.md
   # Retention policy

   Retention is reviewed quarterly and retained for seven years.
```

The output above is representative sample output. Scores and snippets depend
on the documents being indexed. Index diagnostics are written to stderr; the
summary and search results are written to stdout.

## Indexing

The index command accepts UTF-8 text, Markdown, JSON, YAML, TOML,
reStructuredText, and common source-code files by default:

```console
$ ragpocket index ./handbook \
    --db ./var/handbook.sqlite3 \
    --exclude 'archive/**' \
    --chunk-chars 1200 \
    --overlap-chars 200
```

Useful options include:

- `--include GLOB` to add explicit relative-path patterns.
- `--exclude GLOB` to exclude paths; exclusions take precedence.
- `--max-bytes N` to skip files larger than the per-file limit.
- `--metadata-file PATH` to attach scalar metadata from a JSON mapping.
- `--follow-symlinks` to opt into symlink traversal within the indexed folder.

Indexing is deterministic for the same input bytes and configuration. A
changed file is rebuilt, unchanged files can be reused, and a failed build
does not replace the previous usable database. Hidden paths, `.git`, virtual
environments, `.ragpocket`, and common generated directories are excluded by
default.

## Searching and filtering

Search defaults to the BM25 ranker and returns up to eight results. Use JSON
when another program needs structured results:

```console
$ ragpocket search "deployment" \
    --db ./handbook/.ragpocket/index.sqlite3 \
    --format json \
    --top-k 5
```

Apply one or more filters before ranking. Repeated `--where` options are
combined with AND:

```console
$ ragpocket search retention \
    --db ./handbook/.ragpocket/index.sqlite3 \
    --where meta.project=alpha \
    --where ext=.md
```

Supported operators are `=`, `!=`, `^=` (prefix), `~=` (case-insensitive
substring), `>`, `>=`, `<`, and `<=`. Built-in fields include `path`, `name`,
`ext`, `bytes`, `sha256`, `mtime_ns`, and `language`; user fields are addressed
as `meta.<key>`. Filter expressions are parsed and compared by RAGPocket, not
evaluated as Python or interpolated into SQL.

The JSON result shape includes the query, ranker, filters, score, path, line
range, snippet, metadata, and stable chunk identifier:

```json
{
  "query": "retention",
  "ranker": "bm25",
  "filters": [{"field": "meta.project", "op": "=", "value": "alpha"}],
  "results": [
    {
      "rank": 1,
      "score": 1.126849,
      "citation": "[[docs/guide.md#L1-L3|c0000]]",
      "chunk_id": "sha256:...",
      "path": "docs/guide.md",
      "line_start": 1,
      "line_end": 3,
      "snippet": "# Alpha deployment\ndeployment retention alpha",
      "metadata": {"ext": ".md", "project": "alpha"}
    }
  ]
}
```

## Citations

Every result has a human-readable citation in this form:

```text
[[relative/path#L<start>-L<end>|c<sequence>]]
```

For example, `[[docs/guide.md#L18-L29|c0003]]` identifies the source file,
the one-based inclusive line range, and the chunk sequence. The accompanying
`chunk_id` is the machine-stable SHA-256 identity. The path is relative to the
indexed folder, and the CLI does not require a web server to use citations.

## Offline and privacy behavior

Core indexing and BM25 search are local operations. By default RAGPocket:

- opens files from the local filesystem and stores the index in SQLite;
- makes no network requests, telemetry calls, URL fetches, model downloads, or
  hosted-service calls;
- does not execute, import, or render indexed documents;
- does not include file contents, snippets, or metadata in diagnostics; and
- skips symlinks, credential-like hidden files, and oversized or invalid files
  according to the documented defaults.

The index is an inspectable artifact, but it contains source-derived text and
metadata. Protect the database with the same care as the source folder.

## Optional embedding providers

Embeddings are an explicit extension, never an installation or default-search
dependency. Pass a `MODULE:OBJECT` reference only when selecting
`--embedding`; the object can be an instance or a zero-argument factory that
implements the small `EmbeddingProvider` protocol:

```python
from collections.abc import Sequence

class Provider:
    name = "example"
    model = "example-v1"
    dimensions = 3

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...

    def embed_query(self, text: str) -> Sequence[float]:
        ...
```

Use `--rank hybrid` with a provider to combine BM25 and cosine scores. Provider
code is trusted executable code and is loaded only because it was explicitly
selected; it is never discovered from the indexed folder. The provider owns
its model and runtime dependencies and may send chunk or query text outside
the process. The default RAGPocket path never does so.

## Current limitations

RAGPocket is early development software. In particular:

- it retrieves snippets but does not generate answers, chat history, or a
  hosted API;
- the default ranker is lexical BM25 without stemming, stop-word lists,
  synonyms, or semantic matching;
- core indexing handles text formats, not OCR, audio/video transcription,
  binary extraction, or URL crawling;
- the command-line interface and SQLite schema may evolve before a stable 1.0
  release; and
- embedding providers are optional integrations and are not bundled or tested
  as part of the default installation.

See [`docs/SPEC.md`](docs/SPEC.md) for the complete behavioral contract and
determinism requirements.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the local development and test
workflow. Contributions should preserve deterministic behavior and the
offline-by-default boundary.

## License

RAGPocket is released under the MIT License. See [`LICENSE`](LICENSE).
