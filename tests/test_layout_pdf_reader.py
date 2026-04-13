import os
import re
import unittest

from layout_pdf_reader import (
    LayoutBlock,
    _bbox_overlap_ratio,
    classify_layout_blocks,
    read_pdf_for_detection,
)


class LayoutPdfReaderTests(unittest.TestCase):
    def test_bbox_overlap_ratio_uses_block_area(self):
        block = (10.0, 10.0, 30.0, 30.0)
        table = (0.0, 0.0, 20.0, 20.0)

        self.assertAlmostEqual(_bbox_overlap_ratio(block, table), 0.25)

    def test_classify_layout_blocks_marks_table_overlap_and_body(self):
        blocks = [
            LayoutBlock(
                text="GRU 2.279 2.384 11.260 12.314 12.758",
                page_index=0,
                bbox=(100.0, 100.0, 500.0, 120.0),
                page_width=595.0,
                page_height=842.0,
            ),
            LayoutBlock(
                text="实验结果表明，所提模型具有最佳性能。",
                page_index=0,
                bbox=(80.0, 300.0, 520.0, 330.0),
                page_width=595.0,
                page_height=842.0,
            ),
        ]

        classified = classify_layout_blocks(
            blocks,
            table_bboxes_by_page={0: [(90.0, 90.0, 530.0, 160.0)]},
        )

        self.assertEqual(classified[0].kind, "table")
        self.assertEqual(classified[1].kind, "body")

    def test_known_pdf_numeric_table_noise_is_removed(self):
        data_dir = "data"
        if not os.path.isdir(data_dir):
            self.skipTest("data directory is not available")

        k_pdf = next(
            (
                os.path.join(data_dir, name)
                for name in os.listdir(data_dir)
                if name.startswith("K") and name.lower().endswith(".pdf")
            ),
            None,
        )
        diabetes_pdf = next(
            (
                os.path.join(data_dir, name)
                for name in os.listdir(data_dir)
                if name.lower().endswith(".pdf")
                and os.path.getsize(os.path.join(data_dir, name)) == 4475171
            ),
            None,
        )
        if not k_pdf or not diabetes_pdf:
            self.skipTest("sample PDFs are not available")

        k_text = re.sub(r'\s+', ' ', read_pdf_for_detection(k_pdf))
        diabetes_text = re.sub(r'\s+', ' ', read_pdf_for_detection(diabetes_pdf))

        self.assertIn("市场规模分析", k_text)
        self.assertNotIn("2.282", k_text)
        self.assertNotIn("21Q1", k_text)
        self.assertNotIn("23.0%", k_text)

        self.assertIn("实验结果表明", diabetes_text)
        self.assertNotIn("12.314", diabetes_text)
        self.assertNotIn("15.764", diabetes_text)


if __name__ == "__main__":
    unittest.main()
