import unittest

from text_processing.cleaners.academic import clean_academic_noise


class AcademicCleanerTests(unittest.TestCase):
    def test_cleaner_uses_body_heading_instead_of_toc_entry(self):
        chapter_title = "\u7b2c\u4e00\u7ae0 \u7eea\u8bba"
        body_sentence = (
            "\u968f\u7740\u533b\u836f\u884c\u4e1a\u7ade\u4e89\u7684\u65e5\u8d8b\u767d\u70ed\u5316\uff0c"
            "\u4f01\u4e1a\u9700\u8981\u4e0d\u65ad\u4f18\u5316\u8425\u9500\u7b56\u7565\u3002"
        )
        body = "\n\n".join(
            [
                chapter_title,
                chapter_title,
                "\n".join([body_sentence] * 30),
                "1.1 \u7814\u7a76\u80cc\u666f",
                "\n".join(
                    [
                        "\u672c\u7814\u7a76\u65e8\u5728\u4e3a\u6570\u5b57\u5316\u8425\u9500\u5b9e\u8df5\u63d0\u4f9b\u7406\u8bba\u652f\u6301\u3002"
                    ]
                    * 20
                ),
            ]
        )
        toc = "\n".join(
            [
                "\u76ee \u5f55",
                f"{chapter_title}.................................... 1",
                "1.1 \u7814\u7a76\u80cc\u666f .................................. 2",
                "\u53c2\u8003\u6587\u732e ..................................... 69",
            ]
        )
        raw = "\n\n".join(
            [
                "\u4f5c\u8005\u59d3\u540d\n:\u80e1\u4f73\u7eaf",
                toc,
                body,
                "\u53c2\u8003\u6587\u732e\n[1] \u5f20\u4e09. \u793a\u4f8b\u6587\u732e.",
            ]
        )

        cleaned = clean_academic_noise(raw)

        self.assertIn(body_sentence, cleaned)
        self.assertIn("\u672c\u7814\u7a76\u65e8\u5728", cleaned)
        self.assertNotIn("....................................", cleaned)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("\u793a\u4f8b\u6587\u732e", cleaned)


if __name__ == "__main__":
    unittest.main()
