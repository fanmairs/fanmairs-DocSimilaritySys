import os
import tempfile
import unittest

from document_readers.base import UnsupportedDocumentTypeError
from document_readers.factory import read_document_by_type


class DocumentReadersFactoryTests(unittest.TestCase):
    def test_reads_txt_by_extension(self):
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".txt",
            encoding="utf-8",
            delete=False,
        ) as file:
            file.write("hello document reader")
            path = file.name

        try:
            self.assertEqual(read_document_by_type(path), "hello document reader")
        finally:
            os.remove(path)

    def test_rejects_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".unknown",
            encoding="utf-8",
            delete=False,
        ) as file:
            file.write("unsupported")
            path = file.name

        try:
            with self.assertRaises(UnsupportedDocumentTypeError):
                read_document_by_type(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
