import unittest

from text_processing.cleaners.academic import clean_academic_noise
from text_processing.normalizers.pdf import normalize_pdf_detection_text
from text_processing.segmenters.paragraphs import get_paragraphs
from text_processing.segmenters.sentences import split_sentences, split_sentences_with_offsets


class TextProcessingTests(unittest.TestCase):
    def test_normalize_pdf_detection_text_preserves_paragraph_breaks(self):
        text = "第一行\n第二行\n\n新段落"

        self.assertEqual(
            normalize_pdf_detection_text(text),
            "第一行第二行\n\n新段落",
        )

    def test_academic_cleaner_removes_citations_and_inline_formula(self):
        text = "引言 本文研究数字化营销策略[1]。$Y=X+1$ 研究结果表明效果提升。"

        cleaned = clean_academic_noise(text)

        self.assertIn("数字化营销策略", cleaned)
        self.assertIn("研究结果表明", cleaned)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("Y=X+1", cleaned)

    def test_sentence_segmenters_return_plain_sentences_and_offsets(self):
        text = "第一句。第二句！"

        self.assertEqual(split_sentences(text), ["第一句。", "第二句！"])

        spans = split_sentences_with_offsets(text)
        self.assertEqual(spans[0]["text"], "第一句。")
        self.assertEqual(spans[0]["start"], 0)
        self.assertEqual(spans[1]["text"], "第二句！")

    def test_get_paragraphs_falls_back_to_sentence_windows(self):
        text = "第一句很长很长。第二句也很长很长。第三句继续补充内容。"

        paragraphs = get_paragraphs(text, min_chars=10, max_count=2)

        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(all(len(paragraph) >= 10 for paragraph in paragraphs))


if __name__ == "__main__":
    unittest.main()
