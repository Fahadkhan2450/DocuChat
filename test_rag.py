"""Tests for RAG retrieval, citation validation, query expansion, and grounded response generation."""
import unittest
import re
from dataclasses import dataclass
from pathlib import Path

import config
import ingest
import rag
import store


@dataclass
class MockHit:
    chunk_id: int
    doc_id: int
    doc_name: str
    path: str
    kind: str
    page: int
    text: str
    dense: float = 0.8
    score: float = 0.1
    printed_page: int | None = None

    @property
    def label(self):
        if self.kind == "pdf":
            pp = self.printed_page
            if pp is None and self.path:
                pp = rag.get_printed_page(self.path, self.page)
            if pp is not None and pp != self.page:
                return f"PDF p. {self.page} (printed p. {pp})"
            return f"PDF p. {self.page}"
        return f"section {self.page}"

    @property
    def heading(self):
        first = self.text.split("\n", 1)[0]
        return first[2:].strip() if first.startswith("# ") else ""


class TestRagPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store.init()
        cls.retriever = rag.Retriever(store)

    def test_printed_page_extraction_and_citation_labels(self):
        """Test distinguishing PDF page indices from printed page numbers."""
        pdf_path = config.DOCS_DIR / "d5a8945b20_office-ergonomics-handbook.pdf"
        if not pdf_path.exists():
            self.skipTest("Ergonomics PDF not found")

        # Test printed page detection helper
        self.assertEqual(rag.get_printed_page(str(pdf_path), 3), 1)
        self.assertEqual(rag.get_printed_page(str(pdf_path), 5), 3)
        self.assertEqual(rag.get_printed_page(str(pdf_path), 7), 5)
        self.assertEqual(rag.get_printed_page(str(pdf_path), 9), 7)
        self.assertEqual(rag.get_printed_page(str(pdf_path), 11), 9)
        self.assertEqual(rag.get_printed_page(str(pdf_path), 23), 21)

        hit = MockHit(
            chunk_id=1, doc_id=2, doc_name="office-ergonomics-handbook.pdf",
            path=str(pdf_path), kind="pdf", page=5,
            text="# Adjusting Your Chair\nAdjust seat height so thighs are horizontal.",
            printed_page=3
        )
        self.assertEqual(hit.label, "PDF p. 5 (printed p. 3)")
        self.assertEqual(hit.heading, "Adjusting Your Chair")

        # When printed page equals PDF page
        hit_same = MockHit(
            chunk_id=2, doc_id=2, doc_name="sample.pdf",
            path="", kind="pdf", page=5,
            text="# Sample\nSample text.",
            printed_page=5
        )
        self.assertEqual(hit_same.label, "PDF p. 5")

    def test_citation_validation_and_deduplication(self):
        """Test that invalid citations are stripped or remapped and duplicate bullets removed."""
        hits = [
            MockHit(1, 2, "doc.pdf", "", "pdf", 5,
                    "# Adjusting Your Chair\nAdjust seat height so thighs are horizontal and feet flat on the floor.",
                    printed_page=3),
            MockHit(2, 2, "doc.pdf", "", "pdf", 7,
                    "# Positioning Your Keyboard and Mouse\nMaintain a straight wrist. Place home row at elbow height.",
                    printed_page=5),
            MockHit(3, 2, "doc.pdf", "", "pdf", 9,
                    "# Adjusting Your Monitor\nTop line of text at eye level. Viewing distance 18 to 24 inches. Report screen flicker.",
                    printed_page=7),
            MockHit(4, 2, "doc.pdf", "", "pdf", 11,
                    "# Adjusting the Lighting\nUse adjustable task lighting or desk lamps at document sources. Reduce overhead lighting.",
                    printed_page=9),
        ]

        # Valid citations should be preserved
        valid_text = (
            "### Chair\n"
            "- Adjust seat height so thighs are horizontal [1].\n\n"
            "### Lighting\n"
            "- Use adjustable task lighting or desk lamps at document sources [4]."
        )
        cleaned, cited = rag.validate_citations(valid_text, hits)
        self.assertIn("[1]", cleaned)
        self.assertIn("[4]", cleaned)
        self.assertEqual(cited, [1, 4])

        # Wrong citation index re-mapped: statement mentions screen flicker but cited [2] (keyboard)
        mismatched_text = "- Report screen flicker to your supervisor [2]."
        cleaned_remapped, cited_remapped = rag.validate_citations(mismatched_text, hits)
        self.assertIn("[3]", cleaned_remapped)
        self.assertEqual(cited_remapped, [3])

        # Ungrounded claim should have citation stripped
        fake_text = "- Space astronauts must wear insulated suits on Mars [1]."
        cleaned_fake, cited_fake = rag.validate_citations(fake_text, hits)
        self.assertNotIn("[1]", cleaned_fake)
        self.assertEqual(cited_fake, [])

        # Duplicate bullet points across sections should be removed
        dup_text = (
            "### Keyboard and Mouse\n"
            "- Maintain a straight wrist [2].\n"
            "- Use adjustable task lighting at document sources [4].\n\n"
            "### Lighting\n"
            "- Use adjustable task lighting at document sources [4]."
        )
        cleaned_dup, _ = rag.validate_citations(dup_text, hits)
        # Should appear only once in the entire text
        self.assertEqual(cleaned_dup.count("Use adjustable task lighting at document sources"), 1)

    def test_multi_topic_query_decomposition(self):
        """Test query expansion decomposes multi-component queries into sub-queries."""
        q = "Explain how to set up an ergonomic workstation, covering the chair, keyboard, mouse, monitor, and lighting."
        queries = rag.expand_queries(q, config.CHAT_MODEL)
        self.assertGreaterEqual(len(queries), 2)
        joined_queries = " ".join(queries).lower()
        # Ensure distinct component queries are present
        self.assertTrue("chair" in joined_queries)
        self.assertTrue("keyboard" in joined_queries or "mouse" in joined_queries)
        self.assertTrue("monitor" in joined_queries)
        self.assertTrue("light" in joined_queries)

    def test_multi_section_retrieval_coverage(self):
        """Test retrieval returns candidates covering all requested workstation components."""
        # Query covers chair, keyboard, mouse, monitor, lighting
        q = "Explain how to set up an ergonomic workstation, covering the chair, keyboard, mouse, monitor, and lighting."
        queries = [
            q,
            "chair adjustment seat height backrest",
            "keyboard and mouse positioning wrist posture elbow height",
            "monitor display position eye level distance glare",
            "lighting task lamps overhead glare"
        ]
        hits = self.retriever.search(queries, doc_id=2, top=20)
        retrieved_pages = {h.page for h in hits}
        self.assertIn(5, retrieved_pages, "Page 5 (Chair) must be retrieved")
        self.assertIn(7, retrieved_pages, "Page 7 (Keyboard & Mouse) must be retrieved")
        self.assertIn(9, retrieved_pages, "Page 9 (Monitor) must be retrieved")
        self.assertIn(11, retrieved_pages, "Page 11 (Lighting) must be retrieved")

    def test_missing_information_rejection(self):
        """Test that questions completely missing from the document return NOT_FOUND."""
        hits = self.retriever.search(["What is the capital of France and what currency is used?"], doc_id=2)
        # Dense score gate should either reject candidates or return hits where LLM returns NOT_FOUND
        if not hits:
            # Successfully rejected at retrieval gate
            self.assertEqual(len(hits), 0)
        else:
            context = "\n\n".join(f"[{n}] (file: {h.doc_name}, {h.label})\n{rag.tidy(h.text)}"
                                  for n, h in enumerate(hits[:config.TOP_K_FINAL], 1))
            res = rag.client.chat(model=config.CHAT_MODEL, messages=[
                {"role": "system", "content": rag.SYSTEM_PROMPT},
                {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: What is the capital of France and what currency is used?"}
            ], options={"temperature": 0, "num_ctx": config.NUM_CTX})
            ans = res["message"]["content"].strip()
            self.assertEqual(ans, rag.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
