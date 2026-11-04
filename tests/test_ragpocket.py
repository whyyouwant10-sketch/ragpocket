import os
import json
import io
import math
import sqlite3
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from ragpocket.chunking import chunk_text
from ragpocket.cli import main
from ragpocket.index import IndexError_, build_index, connect_index
from ragpocket.metadata import MetadataError, parse_filter
from ragpocket.search import search_index


ROOT = Path(__file__).parent / "fixtures" / "rag"


class RAGPocketTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.folder = self.temp / "rag"
        shutil.copytree(ROOT, self.folder)
        self.db = self.temp / "index.sqlite3"

    def tearDown(self):
        shutil.rmtree(self.temp)

    def test_build_search_and_filters(self):
        counts = build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        self.assertEqual(counts["failed"], 0)
        result = search_index(str(self.db), "deployment")
        self.assertEqual([item["path"] for item in result["results"][:2]], ["docs/guide.md", "src/example.py"])
        self.assertTrue(result["results"][0]["citation"].startswith("[[docs/guide.md#L"))
        alpha = search_index(str(self.db), "retention", filters=[parse_filter("meta.project=alpha")])
        self.assertTrue(all(item["metadata"]["project"] == "alpha" for item in alpha["results"]))
        markdown = search_index(str(self.db), "deployment", filters=[parse_filter("ext=.md")])
        self.assertTrue(all(item["path"].endswith(".md") for item in markdown["results"]))

    def test_reindex_and_mtime_preserve_ids(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        with connect_index(self.db, read_only=True) as connection:
            before = connection.execute("SELECT chunk_id, path FROM chunks JOIN documents USING(document_id) ORDER BY path, sequence").fetchall()
        guide = self.folder / "docs" / "guide.md"
        os.utime(guide, None)
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        with connect_index(self.db, read_only=True) as connection:
            after = connection.execute("SELECT chunk_id, path FROM chunks JOIN documents USING(document_id) ORDER BY path, sequence").fetchall()
        self.assertEqual([(row["chunk_id"], row["path"]) for row in before], [(row["chunk_id"], row["path"]) for row in after])

    def test_chunk_boundaries_and_crlf(self):
        chunks = chunk_text("one\ntwo\nthree\nfour\n", "a" * 64, 8, 2)
        self.assertTrue(all(right.start > left.start for left, right in zip(chunks, chunks[1:])))
        self.assertEqual(chunk_text("a\nb\n", "a" * 64, 20, 0)[0].text, "a\nb\n")

    def test_fixture_crlf_long_line_and_json_cli(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json", chunk_chars=80, overlap_chars=10)
        with connect_index(self.db, read_only=True) as connection:
            crlf = connection.execute("SELECT text, line_start, line_end FROM chunks JOIN documents USING(document_id) WHERE path='notes/crlf.txt'").fetchone()
            long_line = connection.execute("SELECT chunk_id, start_offset, end_offset FROM chunks JOIN documents USING(document_id) WHERE path='notes/long-line.txt' ORDER BY sequence").fetchall()
        self.assertEqual(crlf["text"], "first line\nsecond line with retention\nthird line\n")
        self.assertEqual((crlf["line_start"], crlf["line_end"]), (1, 3))
        self.assertGreater(len(long_line), 1)
        self.assertTrue(all(right["start_offset"] > left["start_offset"] for left, right in zip(long_line, long_line[1:])))
        self.assertEqual(len({row["chunk_id"] for row in long_line}), len(long_line))
        self.assertEqual(main(["search", "retention", "--db", str(self.db), "--format", "json"]), 0)

    def test_cli_invalid_query_and_filter(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        self.assertEqual(main(["search", "!!!", "--db", str(self.db)]), 2)
        self.assertEqual(main(["search", "deployment", "--db", str(self.db), "--where", "not-a-field=x"]), 2)

    def test_chunking_is_deterministic_and_validates_configuration(self):
        text = "first line\nsecond line\nthird line\nfourth line\n"
        first = chunk_text(text, "b" * 64, chunk_chars=18, overlap_chars=5)
        second = chunk_text(text, "b" * 64, chunk_chars=18, overlap_chars=5)
        self.assertEqual(first, second)
        self.assertEqual([item.sequence for item in first], list(range(len(first))))
        self.assertTrue(all(item.end - item.start <= 18 for item in first))
        self.assertTrue(all(right.start > left.start for left, right in zip(first, first[1:])))
        self.assertEqual(chunk_text("", "b" * 64), [])
        with self.assertRaises(ValueError):
            chunk_text("text", "b" * 64, chunk_chars=0)
        with self.assertRaises(ValueError):
            chunk_text("text", "b" * 64, chunk_chars=4, overlap_chars=4)

    def test_sqlite_lifecycle_replaces_changed_and_deleted_documents(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        with connect_index(self.db, read_only=True) as connection:
            tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            before_guide = [row["chunk_id"] for row in connection.execute("SELECT c.chunk_id FROM chunks c JOIN documents d USING(document_id) WHERE d.path='docs/guide.md' ORDER BY c.sequence")]
        self.assertTrue({"index_meta", "documents", "chunks", "terms", "postings"} <= tables)
        self.assertNotIn("embeddings", tables)

        (self.folder / "docs" / "guide.md").write_text("changed deployment content\n", encoding="utf-8")
        (self.folder / "docs" / "retention.md").unlink()
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        with connect_index(self.db, read_only=True) as connection:
            paths = [row["path"] for row in connection.execute("SELECT path FROM documents ORDER BY path")]
            after_guide = [row["chunk_id"] for row in connection.execute("SELECT c.chunk_id FROM chunks c JOIN documents d USING(document_id) WHERE d.path='docs/guide.md' ORDER BY c.sequence")]
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM documents WHERE path='docs/retention.md'").fetchone()[0], 0)
        self.assertIn("docs/guide.md", paths)
        self.assertNotIn("docs/retention.md", paths)
        self.assertNotEqual(before_guide, after_guide)

    @pytest.mark.plugin
    def test_failed_build_is_atomic_and_previous_index_remains_searchable(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        before = self.db.read_bytes()
        with self.assertRaises(IndexError_):
            build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json", embedding="tests.fake_provider:InvalidProvider")
        self.assertEqual(before, self.db.read_bytes())
        self.assertEqual(search_index(str(self.db), "deployment")["results"][0]["path"], "docs/guide.md")

    def test_bm25_score_formula_and_deterministic_tie_order(self):
        folder = self.temp / "bm25"
        folder.mkdir()
        (folder / "a.txt").write_text("target target common\n", encoding="utf-8")
        (folder / "b.txt").write_text("target common common common\n", encoding="utf-8")
        db = self.temp / "bm25.sqlite3"
        build_index(folder, db)
        result = search_index(str(db), "target")
        self.assertEqual([item["path"] for item in result["results"]], ["a.txt", "b.txt"])
        n = 2
        df = 2
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        avgdl = (3 + 4) / 2
        expected = idf * (2 * 2.2) / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl))
        self.assertAlmostEqual(result["results"][0]["score"], round(expected, 6))
        self.assertGreater(result["results"][0]["score"], result["results"][1]["score"])

    def test_citations_metadata_and_filter_operators(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        guide = search_index(str(self.db), "deployment", filters=[parse_filter("path^=docs/"), parse_filter("name~=GUIDE"), parse_filter("meta.project=alpha")])
        self.assertEqual(len(guide["results"]), 1)
        item = guide["results"][0]
        self.assertEqual(item["citation"], "[[docs/guide.md#L1-L3|c0000]]")
        self.assertEqual(item["line_start"], 1)
        self.assertEqual(item["line_end"], 3)
        self.assertEqual(item["metadata"]["project"], "alpha")
        self.assertTrue(search_index(str(self.db), "retention", filters=[parse_filter("meta.project!=alpha")])["results"])
        self.assertTrue(all(item["path"].endswith(".md") for item in search_index(str(self.db), "retention", filters=[parse_filter("ext=.md")])["results"]))
        with self.assertRaises(MetadataError):
            search_index(str(self.db), "deployment", filters=[parse_filter("bytes>not-a-number")])

    def test_malformed_and_empty_sources_are_reported_without_losing_valid_files(self):
        malformed = self.folder / "malformed.txt"
        malformed.write_bytes(b"valid prefix\xff\n")
        counts = build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        self.assertEqual(counts["failed"], 1)
        with connect_index(self.db, read_only=True) as connection:
            empty = connection.execute("SELECT normalized_chars FROM documents WHERE path='empty.txt'").fetchone()
            self.assertEqual(empty["normalized_chars"], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM chunks JOIN documents USING(document_id) WHERE path='empty.txt'").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM documents WHERE path='malformed.txt'").fetchone()[0], 0)
        self.assertTrue(search_index(str(self.db), "deployment")["results"])

    def test_json_and_text_output_contracts(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        json_stdout = io.StringIO()
        with redirect_stdout(json_stdout):
            self.assertEqual(main(["search", "deployment", "--db", str(self.db), "--format", "json", "--top-k", "1"]), 0)
        payload = json.loads(json_stdout.getvalue())
        self.assertEqual(payload["ranker"], "bm25")
        self.assertEqual(payload["results"][0]["rank"], 1)
        self.assertIsInstance(payload["results"][0]["score"], float)
        self.assertEqual(len(f"{payload['results'][0]['score']:.6f}".split(".")[1]), 6)

        text_stdout = io.StringIO()
        with redirect_stdout(text_stdout):
            self.assertEqual(main(["search", "deployment", "--db", str(self.db), "--format", "text", "--top-k", "1"]), 0)
        self.assertIn("1. ", text_stdout.getvalue())
        self.assertIn("[[docs/guide.md#L1-L3|c0000]]", text_stdout.getvalue())
        self.assertIn("deployment", text_stdout.getvalue())

    def test_read_only_connection_and_search_do_not_modify_database(self):
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json")
        before = self.db.read_bytes()
        with connect_index(self.db, read_only=True) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE should_not_exist(value TEXT)")
        search_index(str(self.db), "deployment")
        self.assertEqual(before, self.db.read_bytes())
        with connect_index(self.db, read_only=True) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='should_not_exist'").fetchone()[0], 0)

    @pytest.mark.plugin
    def test_fake_embedding_provider_builds_embeddings_and_hybrid_searches(self):
        provider = "tests.fake_provider:FakeProvider"
        build_index(self.folder, self.db, metadata_file=self.folder / "metadata.json", embedding=provider)
        with connect_index(self.db, read_only=True) as connection:
            self.assertIn("embeddings", {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")})
            embedding_count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            self.assertEqual(embedding_count, chunk_count)
            self.assertEqual(connection.execute("SELECT DISTINCT dimensions FROM embeddings").fetchone()[0], 2)
        result = search_index(str(self.db), "deployment", rank="hybrid", embedding=provider)
        self.assertEqual(result["ranker"], "hybrid")
        self.assertTrue(result["results"])
        self.assertTrue(all(item["score"] >= 0 for item in result["results"]))

    def test_front_matter_citation_preserves_original_line_numbers(self):
        front_matter = self.temp / "front-matter"
        front_matter.mkdir()
        (front_matter / "doc.md").write_text("---\nproject: alpha\n---\nbody deployment\n", encoding="utf-8")
        db = self.temp / "front-matter.sqlite3"
        build_index(front_matter, db)
        result = search_index(str(db), "deployment")
        self.assertEqual(result["results"][0]["citation"], "[[doc.md#L1-L4|c0000]]")
        self.assertNotIn("project", result["results"][0]["snippet"])
        self.assertEqual(result["results"][0]["metadata"]["project"], "alpha")


if __name__ == "__main__":
    unittest.main()
