# RAGPocket specification

Status: proposed

RAGPocket is a local-first Python command-line tool for asking questions over a
folder of text files. It builds a small, inspectable index and returns source
snippets with deterministic citations. The default retrieval method is BM25;
an embedding provider may be selected explicitly, but is never required for
installation, indexing, searching, or tests.

## 1. Scope and goals

### Goals

- Index a directory without uploading files or contacting a service.
- Make the same input bytes, configuration, and query produce the same chunks,
  citations, ordering, and scores across runs.
- Return useful context for a downstream answer generator without pretending
  that retrieval is an answer.
- Support repeatable metadata filtering before ranking.
- Keep the default path dependency-light: Python standard library plus SQLite.
- Keep the index inspectable and replaceable rather than hiding state in a
  model-specific cache.

### Non-goals

- Generating answers, chat history, or a hosted API.
- Crawling URLs, calling an LLM, or downloading models.
- OCR, audio/video transcription, or binary document extraction in the core
  implementation.
- Stemming, language-specific stop-word lists, or semantic search in the
  default ranker.

The initial implementation MUST accept UTF-8 text, Markdown, JSON, YAML, TOML,
reStructuredText, and common source-code files. Other extensions MAY be
accepted with `--include`; binary files and files that exceed the configured
size limit are skipped with a diagnostic.

## 2. CLI contract

The installed console script is `ragpocket`. All paths supplied by the user are
local filesystem paths. Unless stated otherwise, diagnostics go to stderr and
machine-readable output goes to stdout.

### 2.1 Index a folder

```text
ragpocket index FOLDER [--db PATH]
  [--include GLOB ...] [--exclude GLOB ...]
  [--max-bytes N] [--chunk-chars N] [--overlap-chars N]
  [--follow-symlinks] [--embedding MODULE:OBJECT]
  [--metadata-file PATH]
```

Defaults:

- `--db .ragpocket/index.sqlite3`, created relative to `FOLDER`.
- `--max-bytes 5242880` (5 MiB per source file).
- `--chunk-chars 1200` and `--overlap-chars 200`.
- Exclude `.git/**`, `.ragpocket/**`, `__pycache__/**`, virtual environments,
  and hidden files/directories unless explicitly included.
- Do not follow symlinks.
- Incrementally reuse a document only when its content hash, effective
  chunking configuration, and tokenizer version are unchanged.

`--include` adds glob patterns relative to `FOLDER`; `--exclude` takes
precedence. At least one include pattern may be supplied. A source is selected
using its POSIX-style relative path, even on Windows. The command MUST write
through a temporary database and atomically replace the target database only
after a successful complete build. A failed build MUST leave the previous
usable index intact.

`--metadata-file` is an optional UTF-8 JSON object mapping relative POSIX paths
to scalar metadata, for example:

```json
{
  "docs/guide.md": {"project": "alpha", "visibility": "public"},
  "notes/todo.txt": {"project": "alpha", "visibility": "private"}
}
```

Keys MUST match `^[A-Za-z_][A-Za-z0-9_.-]*$`; values MUST be strings, numbers,
booleans, or null. Unknown paths are reported and ignored. A source's
metadata is the merge of built-in metadata, Markdown scalar front matter (if
present), and this file, with the explicit metadata file taking precedence.
Nested objects and arrays are not required in v1.

The index command reports counts for discovered, indexed, reused, skipped, and
failed files. Exit status is zero when all selected files are indexed or
reused, one when files were skipped or failed, and two for invalid arguments or
an unusable folder.

### 2.2 Search the index

```text
ragpocket search QUERY [--db PATH] [--top-k N]
  [--where EXPR ...] [--format text|json]
  [--rank bm25|hybrid] [--embedding MODULE:OBJECT]
  [--snippet-chars N] [--explain]
```

Defaults are `--top-k 8`, `--format text`, `--rank bm25`, and
`--snippet-chars 600`. Repeating `--where` combines clauses with AND. A filter
is applied before scoring, so a document outside the filter cannot consume a
top-k slot.

Each `--where EXPR` is written without spaces as `FIELD OP VALUE`, for example
`meta.project=alpha`, `path^=docs/`, or `bytes>=1000`. Supported operators are
`=`, `!=`, `^=` (prefix), `~=` (case-insensitive substring), `>`, `>=`, `<`, and
`<=`. Numeric operators apply only to numeric built-ins or numeric metadata. No
filter text is evaluated as Python or interpolated into SQL.

Examples:

```text
ragpocket index ./handbook
ragpocket search "retention policy" --db ./handbook/.ragpocket/index.sqlite3
ragpocket search "deployment" --where meta.project=alpha --where ext=.md
ragpocket search "authentication" --format json --top-k 5
```

The text format prints a one-based result number, score, citation, relative
path, and a bounded snippet. JSON output is an object with this shape:

```json
{
  "query": "retention policy",
  "ranker": "bm25",
  "filters": [{"field": "meta.project", "op": "=", "value": "alpha"}],
  "results": [
    {
      "rank": 1,
      "score": 4.125,
      "citation": "[[docs/guide.md#L18-L29|c0003]]",
      "chunk_id": "sha256:...",
      "path": "docs/guide.md",
      "line_start": 18,
      "line_end": 29,
      "snippet": "...",
      "metadata": {"ext": ".md", "project": "alpha"}
    }
  ]
}
```

Scores MUST be emitted as JSON numbers rounded to six decimal places. An empty
result is a successful search with `results: []`. A query with no index, no
searchable terms, or invalid filter syntax exits two and explains the error.

`--explain` MAY add token, document-frequency, and per-term contribution data
to JSON. It MUST NOT change result ordering or score calculation.

## 3. File discovery, normalization, and metadata

1. Resolve `FOLDER` once and reject a non-directory.
2. Walk in lexical order of normalized POSIX relative paths.
3. Skip symlinks by default, special files, excluded paths, and files over the
   byte limit. With `--follow-symlinks`, resolve each candidate and reject a
   resolved path outside the requested folder.
4. Read bytes, compute SHA-256 over the exact bytes, and decode UTF-8. A UTF-8
   BOM is removed. Invalid UTF-8 is skipped by default and reported; no lossy
   replacement is allowed.
5. Normalize `CRLF` and `CR` to `LF` for indexing. The normalized text is the
   source of chunk offsets and line numbers. A source hash remains the hash of
   original bytes so changes cannot be hidden by normalization.

Every document stores these built-in fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `path` | string | POSIX path relative to the indexed folder |
| `name` | string | Final path component |
| `ext` | string | Lowercase suffix including the dot, or empty |
| `bytes` | integer | Original byte count |
| `sha256` | string | SHA-256 of original bytes |
| `mtime_ns` | integer | Source modification time observed at indexing |
| `language` | string | Best-effort extension mapping, or `unknown` |

User/front-matter fields are addressed as `meta.<key>`. Built-in names cannot
be overwritten; an attempted collision is reported as an indexing error.
Markdown front matter is an optional leading `---` block. Only simple scalar
`key: value` pairs are required. Recognized front-matter delimiter and scalar
lines are metadata-only and are excluded from indexed text, while their
original line positions remain available for citations. Unsupported YAML
constructs are retained as document text and do not become metadata.

## 4. Deterministic chunking and citations

The chunker version is part of the index configuration and is currently
`line-aware-v1`.

1. Use the normalized Unicode text, with line endings represented by `\n`.
2. Start at code-point offset zero. The hard window is at most
   `chunk_chars`; prefer the last newline in the final 20% of the window so a
   chunk normally ends on a line boundary.
3. If no suitable newline exists, end exactly at the hard window. A single
   overlong line is therefore split deterministically by code-point offset.
4. The next start is `end - overlap_chars`, moved forward to the next line
   start when that line boundary lies within the overlap. If the resulting
   start does not advance, use `end` to guarantee progress.
5. Stop when `start >= len(text)`. Empty files produce zero chunks.

The implementation MUST validate `chunk_chars > 0` and
`0 <= overlap_chars < chunk_chars`. It MUST store normalized code-point start
and end offsets, one-based inclusive line start/end, and a zero-based chunk
sequence. Line numbers are calculated from the normalized source, and a chunk
that contains any part of a line cites that whole line.

The canonical chunk ID is:

```text
sha256("chunk-v1\0" + document_sha256 + "\0" + start + "\0" + end)
```

where the numeric offsets are base-10 ASCII and the complete preimage is UTF-8
encoded before hashing. The displayed citation is:

```text
[[<relative-posix-path>#L<line-start>-L<line-end>|c<sequence>]]
```

`sequence` is four or more zero-padded decimal digits (`c0000`, `c0001`, ...).
The path is escaped for display so `]`, `|`, and backslash cannot alter the
citation grammar. The path and line range are human-readable; `chunk_id` is
the exact machine-stable identity. Every result MUST provide both. A renderer
MAY turn the citation into a link, but the CLI MUST NOT require a web server.

Snippets are taken from the normalized chunk text, trimmed only for display,
and bounded by `snippet_chars`. Retrieval MUST never mutate the stored chunk or
the line range used by a citation.

## 5. Storage format

The default store is SQLite 3 at `.ragpocket/index.sqlite3`. SQLite foreign
keys and WAL mode SHOULD be enabled during a build. The schema is versioned in
`index_meta` and MUST reject an unsupported schema or tokenizer version rather
than silently guessing.

Required logical tables:

- `index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)`: schema version,
  chunker/tokenizer versions, root path, effective configuration, and creation
  metadata.
- `documents(document_id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT,
  ext TEXT, language TEXT, bytes INTEGER, sha256 TEXT, mtime_ns INTEGER,
  normalized_chars INTEGER, metadata_json TEXT NOT NULL)`.
- `chunks(chunk_id TEXT PRIMARY KEY, document_id INTEGER NOT NULL,
  sequence INTEGER NOT NULL, start_offset INTEGER, end_offset INTEGER,
  line_start INTEGER, line_end INTEGER, text TEXT NOT NULL, token_count INTEGER,
  UNIQUE(document_id, sequence))`.
- `terms(term_id INTEGER PRIMARY KEY, term TEXT UNIQUE NOT NULL)`.
- `postings(term_id INTEGER, chunk_id TEXT, tf INTEGER, positions_json TEXT,
  PRIMARY KEY(term_id, chunk_id))`.
- `embeddings(chunk_id TEXT, provider TEXT, model TEXT, dimensions INTEGER,
  vector_blob BLOB, PRIMARY KEY(chunk_id, provider, model))`, created only when
  an embedding provider is explicitly requested.

`positions_json` is optional for v1 ranking but allows later highlighting. If
  present, positions are zero-based token indexes. Metadata filtering MUST use
  decoded values with a controlled comparison implementation; it MUST NOT use
  untrusted SQL fragments.

The index is an artifact, not the source of truth. Re-indexing a changed file
replaces its document, chunks, and postings in one transaction. Unseen files
are removed from the new complete index. A second index of identical input
bytes and configuration MUST produce identical content except for explicitly
documented build timestamps.

## 6. Tokenization and BM25 ranking

The tokenizer is `unicode-word-v1`:

- Apply Unicode `casefold()`.
- Emit maximal runs matching Python's Unicode-aware `\w`, excluding `_` when
  it is the only character; punctuation and whitespace are separators.
- Do not stem, remove stop words, or apply synonyms.
- Apply the same tokenizer to chunk text and the query.

The tokenizer version MUST be stored in `index_meta`; changing it requires a
fresh postings build. Query terms are de-duplicated for scoring while their
first appearance is retained for explain output. A query with no tokens is an
error.

For a filtered candidate set, let `N` be the number of candidate chunks,
`df(t)` the number of candidates containing term `t`, `tf(t,d)` its frequency in
chunk `d`, `dl(d)` the chunk token count, and `avgdl` the mean candidate length.
The default score is:

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
BM25(t,d) = idf(t) * (tf(t,d) * (k1 + 1)) /
            (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
score(d) = sum(BM25(t,d) for t in query_terms)
```

Use `k1 = 1.2` and `b = 0.75`. Terms absent from a chunk contribute zero.
When a filter leaves no candidates, `avgdl` is not evaluated and the result is
empty. Results sort by descending score, then ascending POSIX path, then
ascending chunk sequence, then ascending chunk ID. This tie-break is part of
the public contract and avoids dependence on SQLite row order.

## 7. Optional embedding plugin boundary

Embeddings are an opt-in extension, never a default dependency. The core
package MUST import no embedding SDK at module import time and MUST pass all
default tests when no provider is installed.

The public boundary is a small Python protocol:

```python
from collections.abc import Sequence
from typing import Protocol

class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
    def embed_query(self, text: str) -> Sequence[float]: ...
```

`MODULE:OBJECT` is imported only after `--embedding` is supplied. The object
may be an instance or a zero-argument factory. The provider MUST return finite
vectors of exactly `dimensions`; dimension, model, and provider name become
part of the embedding key. It MUST be called in stable chunk order and MUST
not receive file paths, metadata, credentials, or raw filesystem handles.

`--rank hybrid` retrieves BM25 candidates first, obtains or reads vectors, and
combines min-max-normalized BM25 and cosine scores as
`0.7 * bm25 + 0.3 * cosine`. A zero-range score normalizes to zero. Hybrid
results use the same path/sequence/ID tie-break. An embedding failure is a
clear command error and MUST NOT silently fall back to a different ranker;
plain BM25 remains available by omitting `--embedding` and `--rank hybrid`.

Providers are responsible for their own model/runtime dependencies and any
network behavior. RAGPocket core MUST document that selecting a provider may
send chunk/query text outside the process; the default CLI never does so.
Plugin code is trusted executable code and is not loaded from the indexed
folder automatically.

## 8. Safety and operational behavior

- Local-only by default: no network sockets, URL fetching, telemetry, or model
  downloads are performed by core commands.
- Never execute, import, or render source documents. Treat file contents,
  front matter, and retrieved text as untrusted data.
- Do not follow symlinks unless explicitly requested, and prevent an opted-in
  symlink from escaping the root folder.
- Use parameterized SQLite statements for all paths, terms, and filter values.
- Reject path traversal in metadata-file keys and normalize all displayed paths
  to relative POSIX paths.
- Do not index the database itself, temporary files, or common credential files
  by default. Users can explicitly include a file, but the CLI SHOULD warn for
  names such as `.env`, private keys, and credential stores.
- Enforce the per-file byte limit before decoding and bound snippet output.
- Use atomic replacement and retain the last valid index after a failed build.
- Avoid logging file contents, metadata values, vectors, or provider secrets.
- Plugins are explicit code execution. Core MUST NOT discover plugins by
  scanning documents or importing arbitrary names from metadata.

## 9. Acceptance criteria

An implementation is complete when all of the following hold:

1. `ragpocket index FIXTURE_DIR` builds a SQLite index with no embedding
   package installed.
2. A default search returns BM25-ranked results with a snippet, stable score,
   path, line range, and canonical citation.
3. Running index twice over unchanged fixture bytes produces the same document,
   chunk, posting, citation, and result ordering data.
4. Changing only file mtime does not change chunk IDs, postings, or citations.
5. Changing file content replaces only the affected document's chunks/postings
   and removes deleted files on the next complete build.
6. `--where` supports repeated AND filters over built-in and `meta.*` fields;
   invalid operators and type comparisons fail safely.
7. Overlap, an empty file, a file with CRLF, a Unicode file, and a line longer
   than `chunk_chars` all produce documented deterministic results.
8. Search ordering is stable for equal scores and does not depend on directory
   enumeration or SQLite row order.
9. Invalid UTF-8, unreadable files, oversized files, symlinks, and excluded
   paths are reported without aborting a valid build.
10. No default test imports, downloads, or invokes an embedding provider.
11. A fake in-process provider can be injected in a plugin test and its vector
    dimension/model validation is enforced.
12. A failed build leaves the prior index readable and searchable.

## 10. Required test fixtures

The implementation SHOULD keep small, committed fixtures under
`tests/fixtures/rag/` so tests never depend on the host home directory or
network. The fixture tree and required contents are:

```text
rag/
├── docs/guide.md
├── docs/retention.md
├── src/example.py
├── notes/unicode.txt
├── notes/crlf.txt
├── notes/long-line.txt
├── empty.txt
├── ignored/.env
├── outside-link -> ../outside.txt       # only when symlinks are supported
└── metadata.json
```

`guide.md` contains a heading and the terms `deployment`, `retention`, and
`alpha`; `retention.md` contains `retention` and `beta`; `example.py` contains
`deployment` in a comment and function; `unicode.txt` contains non-ASCII
letters and a case variant; `crlf.txt` uses CRLF endings; `long-line.txt`
contains one line longer than the configured chunk size; and `empty.txt` is
zero bytes. `ignored/.env` contains a token-shaped dummy value and must not be
indexed by default. `metadata.json` assigns `project=alpha` to the guide and
example, and `project=beta` to retention.

Fixture tests MUST assert at least:

- `deployment` returns the expected guide/example citations in deterministic
  tie order.
- `retention --where meta.project=alpha` excludes the beta document.
- `--where ext=.md` excludes Python and text files.
- The CRLF fixture has the same normalized line citations as its LF equivalent.
- The long-line fixture has multiple chunks with advancing offsets and stable
  IDs.
- Re-indexing and changing only mtime preserve IDs and result order.
- Empty/no-token queries and malformed filters return exit status two.
- A test run with the embedding provider module unavailable still passes all
  default indexing and BM25 search tests.
- A fake provider is exercised only in a separately marked plugin test.

These fixtures define observable behavior; they do not require a particular
test framework or embedding service.
