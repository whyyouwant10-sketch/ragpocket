# RAGPocket

RAGPocket is a local-only command-line tool for indexing a folder of UTF-8
documents and retrieving deterministic BM25-ranked chunks with citations.

```console
$ ragpocket index ./handbook
$ ragpocket search "retention policy" --db ./handbook/.ragpocket/index.sqlite3
```

The core installation has no embedding or network dependency. An optional
provider can be supplied as `MODULE:OBJECT` for the `--embedding` and
`--rank hybrid` options; provider code is trusted executable code and may have
its own runtime or network behavior.
