import unittest
from unittest.mock import patch, mock_open

from app.helpers.strategies_poc import TextFileManager, WordFileManager, PDFFileManager, FileReader, clean_text


class TestFileManager(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="Sample text file")
    def test_text_file_manager(self, mock_file):
        manager = TextFileManager("test.txt")
        self.assertEqual(manager.read(), "Sample text file")

    @patch("docx.Document")
    def test_word_file_manager(self, mock_doc):
        # Simular contenido del archivo Word
        mock_doc.return_value.paragraphs = [type("Paragraph", (), {"text": "Sample paragraph"})]

        manager = WordFileManager("test.docx")
        self.assertEqual(manager.read(), "Sample paragraph")

    @patch("builtins.open", new_callable=mock_open, read_data="dummy content")
    @patch("PyPDF2.PdfReader")
    def test_pdf_file_manager(self, mock_pdf, mock_file):
        # Mockear las páginas del lector PDF
        mock_pdf.return_value.pages = [type("Page", (), {"extract_text": lambda: "Sample PDF text"})]

        manager = PDFFileManager("test.pdf")
        self.assertEqual(manager.read(), "Sample PDF text")

    def test_file_reader_unsupported_extension(self):
        with self.assertRaises(ValueError):
            FileReader("test.csv")

    @patch("builtins.open", new_callable=mock_open, read_data="Sample text file")
    def test_file_reader_text_file(self, mock_file):
        reader = FileReader("test.txt")
        self.assertEqual(reader.read_file(), "Sample text file")

    @patch("docx.Document")
    def test_file_reader_word_file(self, mock_doc):
        # Simular contenido del archivo Word
        mock_doc.return_value.paragraphs = [type("Paragraph", (), {"text": "Sample paragraph"})]

        reader = FileReader("test.docx")
        self.assertEqual(reader.read_file(), "Sample paragraph")

    @patch("builtins.open", new_callable=mock_open, read_data="dummy content")
    @patch("PyPDF2.PdfReader")
    def test_file_reader_pdf_file(self, mock_pdf, mock_file):
        # Mockear las páginas del lector PDF
        mock_pdf.return_value.pages = [type("Page", (), {"extract_text": lambda: "Sample PDF text"})]

        reader = FileReader("test.pdf")
        self.assertEqual(reader.read_file(), "Sample PDF text")

    def test_clean_text(self):
        # Prueba de limpieza de texto
        raw_text = "This  is  a \ntest\n\nstring with\t\tspaces."
        expected = "This is a test\nstring with spaces."
        self.assertEqual(clean_text(raw_text), expected)


if __name__ == "_main_":
    unittest.main()