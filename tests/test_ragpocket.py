import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ragpocket.chunking import chunk_text
from ragpocket.cli import main
from ragpocket.index import build_index, connect_index
from ragpocket.metadata import parse_filter
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


if __name__ == "__main__":
    unittest.main()
