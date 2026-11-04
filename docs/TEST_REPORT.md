# Test report

Baseline: commit `17d8f0d` (`feat: implement local deterministic rag search`)

Environment: Python 3.11.2, pytest 9.2.0, Linux.

## Regression coverage

`tests/test_ragpocket.py` now covers deterministic chunk boundaries and IDs,
empty and malformed UTF-8 files, SQLite schema and replacement/removal
lifecycle, atomic failed builds, BM25 scoring and tie ordering, canonical
citations, repeated metadata filters and safe type handling, JSON/text output,
read-only index access, front matter line citations, and the optional fake
embedding provider/hybrid ranker. The fake provider is in
`tests/fake_provider.py` and provider-dependent cases use the `plugin` marker.

## Checks

All commands were run from the repository root unless noted.

### Fresh install

```console
$ rm -rf /tmp/opencode/ragpocket-fresh && python -m venv /tmp/opencode/ragpocket-fresh && /tmp/opencode/ragpocket-fresh/bin/python -m pip install --no-deps --force-reinstall . && /tmp/opencode/ragpocket-fresh/bin/ragpocket --help >/tmp/opencode/ragpocket-fresh-help.txt && grep -q 'Deterministic local folder retrieval' /tmp/opencode/ragpocket-fresh-help.txt && echo 'fresh install: PASS'
Successfully installed ragpocket-0.1.0
fresh install: PASS
```

### Tests

```console
$ python -m pytest -q
...............                                                          [100%]
15 passed in 1.29s
```

The timing varied slightly between runs; the result was 15 passed, 0 failed.

```console
$ python -m pytest -q -m 'not plugin'
.............                                                            [100%]
13 passed, 2 deselected in 1.60s

$ python -m pytest -q -m plugin
..                                                                       [100%]
2 passed, 13 deselected in 0.22s

$ python -m unittest discover -s tests -q
Ran 15 tests in 1.840s

OK
```

The unittest run also emitted the expected diagnostics for invalid queries,
the malformed UTF-8 fixture, and the intentionally unknown metadata path.

### Compile check

```console
$ python -m compileall -q src tests && echo compileall:PASS
compileall:PASS
```

### Index/search CLI smoke

The fixture includes an outside symlink, so this smoke command explicitly
excludes it and receives the successful index exit status required for a clean
CLI smoke run.

```console
$ rm -rf /tmp/opencode/ragpocket-smoke && cp -a tests/fixtures/rag /tmp/opencode/ragpocket-smoke
$ /tmp/opencode/ragpocket-fresh/bin/ragpocket index /tmp/opencode/ragpocket-smoke --db /tmp/opencode/ragpocket-cli-smoke.sqlite3 --metadata-file /tmp/opencode/ragpocket-smoke/metadata.json --exclude outside-link > /tmp/opencode/cli-index.out
$ /tmp/opencode/ragpocket-fresh/bin/ragpocket search deployment --db /tmp/opencode/ragpocket-cli-smoke.sqlite3 --format json --top-k 2 > /tmp/opencode/cli-search.json
$ /tmp/opencode/ragpocket-fresh/bin/ragpocket search deployment --db /tmp/opencode/ragpocket-cli-smoke.sqlite3 --where meta.project=alpha --format text --top-k 2 > /tmp/opencode/cli-filter.txt
$ python - <<'PY'
> import json
> with open('/tmp/opencode/cli-index.out') as f: counts=json.load(f)
> with open('/tmp/opencode/cli-search.json') as f: payload=json.load(f)
> assert counts == {'discovered': 8, 'failed': 0, 'indexed': 8, 'reused': 0, 'skipped': 0}, counts
> assert payload['ranker'] == 'bm25'
> assert [item['path'] for item in payload['results']] == ['docs/guide.md', 'src/example.py']
> assert all(item['citation'].startswith('[[') for item in payload['results'])
> with open('/tmp/opencode/cli-filter.txt') as f: text=f.read()
> assert 'docs/guide.md' in text and 'src/example.py' in text
> PY
index/search CLI smoke: PASS
```

### Hybrid CLI smoke

```console
$ PYTHONPATH=/home/_a_h/projects_git/ragpocket /tmp/opencode/ragpocket-fresh/bin/ragpocket index /tmp/opencode/ragpocket-smoke --db /tmp/opencode/ragpocket-hybrid-smoke.sqlite3 --metadata-file /tmp/opencode/ragpocket-smoke/metadata.json --exclude outside-link --embedding tests.fake_provider:FakeProvider > /tmp/opencode/hybrid-index.out
$ PYTHONPATH=/home/_a_h/projects_git/ragpocket /tmp/opencode/ragpocket-fresh/bin/ragpocket search deployment --db /tmp/opencode/ragpocket-hybrid-smoke.sqlite3 --rank hybrid --embedding tests.fake_provider:FakeProvider --format json --top-k 2 > /tmp/opencode/hybrid-search.json
$ python - <<'PY'
> import json
> with open('/tmp/opencode/hybrid-index.out') as f: counts=json.load(f)
> with open('/tmp/opencode/hybrid-search.json') as f: payload=json.load(f)
> assert counts == {'discovered': 8, 'failed': 0, 'indexed': 8, 'reused': 0, 'skipped': 0}, counts
> assert payload['ranker'] == 'hybrid'
> assert len(payload['results']) == 2
> assert all(item['score'] >= 0 for item in payload['results'])
> PY
hybrid CLI smoke: PASS
```

No implementation defect was exposed by these checks, so no production source
files were changed.
