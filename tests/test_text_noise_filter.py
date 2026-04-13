import unittest

from text_noise_filter import filter_detection_text_blocks, is_numeric_table_noise


class TextNoiseFilterTests(unittest.TestCase):
    def test_regression_table_row_is_noise(self):
        sample = "社会环境 0.162 0.071 0.148 2.282 0.025* 3.88"

        self.assertTrue(is_numeric_table_noise(sample))

    def test_metric_result_table_row_is_noise(self):
        sample = "GRU 2.279 2.384 11.260 12.314 12.758 12.815 13.981 14.020"

        self.assertTrue(is_numeric_table_noise(sample))

    def test_chart_axis_and_percent_series_are_noise(self):
        sample = (
            "23.0% 21.9% 22.7% 23.3% 23.6% 22.9% 23.0% "
            "21Q121Q221Q321Q422Q122Q222Q322Q4"
        )

        self.assertTrue(is_numeric_table_noise(sample))

    def test_narrative_sentence_with_numbers_is_kept(self):
        sample = "2023 年全年收入为97.71 亿元，同比增长19.01%，但整体营收依然实现增长。"

        self.assertFalse(is_numeric_table_noise(sample))

    def test_filter_detection_text_blocks_keeps_narrative_blocks(self):
        blocks = [
            "GRU 2.279 2.384 11.260 12.314 12.758 12.815 13.981 14.020",
            "实验结果表明，所提模型 TF-CNN 具有最佳性能。",
        ]

        self.assertEqual(
            filter_detection_text_blocks(blocks),
            ["实验结果表明，所提模型 TF-CNN 具有最佳性能。"],
        )


if __name__ == "__main__":
    unittest.main()
