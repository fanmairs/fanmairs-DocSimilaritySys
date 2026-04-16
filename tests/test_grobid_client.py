import unittest

from document_readers.pdf.grobid_backend import extract_body_text_from_tei


class GrobidClientTests(unittest.TestCase):
    def test_extract_body_text_from_tei_skips_tables_figures_and_references(self):
        tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <front>
              <abstract><p>摘要不进入正文查重。</p></abstract>
            </front>
            <body>
              <div>
                <head>第一章 绪论</head>
                <p>本文围绕数字化营销策略展开研究，重点分析用户触达路径。</p>
                <figure><p>图1 结构示意图</p></figure>
                <table><row><cell>GRU</cell><cell>12.314</cell></row></table>
                <formula>Y=0.1X+2</formula>
                <p>研究结果表明，该策略能够提升医生端内容触达效率。</p>
              </div>
            </body>
            <back>
              <listBibl><bibl>参考文献不进入正文查重。</bibl></listBibl>
            </back>
          </text>
        </TEI>
        """

        text = extract_body_text_from_tei(tei)

        self.assertIn("第一章 绪论", text)
        self.assertIn("数字化营销策略", text)
        self.assertIn("医生端内容触达效率", text)
        self.assertNotIn("摘要不进入正文查重", text)
        self.assertNotIn("12.314", text)
        self.assertNotIn("参考文献", text)


if __name__ == "__main__":
    unittest.main()
