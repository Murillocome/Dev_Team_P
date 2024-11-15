import unittest
from unittest.mock import MagicMock
from app.helpers.vectorize_documents import get_openai_embeddings, document_to_vectors, chunk_text


class TestVectorizeDocuments(unittest.TestCase):

    def setUp(self):
        # Configuración inicial para las pruebas
        self.fake_openai_client = MagicMock()
        self.fake_openai_client._client.embeddings.create.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2, 0.3]}]
        )

    def test_get_openai_embeddings(self):
        # Prueba para get_openai_embeddings
        text = "Prueba de texto"
        result = get_openai_embeddings(text, self.fake_openai_client)
        self.fake_openai_client._client.embeddings.create.assert_called_once_with(
            input=text,
            model="text-embedding-ada-002"
        )
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_document_to_vectors(self):
        # Prueba para document_to_vectors
        content = "Este es un texto largo que necesita ser dividido en fragmentos."
        with unittest.mock.patch('app.helpers.vectorize_documents.chunk_text', return_value=["chunk1", "chunk2"]):
            result = document_to_vectors(content, self.fake_openai_client)

        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        self.fake_openai_client._client.embeddings.create.assert_any_call(
            input="chunk1",
            model="text-embedding-ada-002"
        )
        self.fake_openai_client._client.embeddings.create.assert_any_call(
            input="chunk2",
            model="text-embedding-ada-002"
        )

    def test_chunk_text(self):
        # Prueba para chunk_text
        text = "Este es un texto largo que será fragmentado."
        max_tokens = 5
        with unittest.mock.patch('tiktoken.get_encoding') as mock_get_encoding:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            mock_tokenizer.decode.side_effect = lambda x: " ".join(map(str, x))
            mock_get_encoding.return_value = mock_tokenizer

            result = chunk_text(text, max_tokens)

        self.assertEqual(result, ["1 2 3 4 5", "6 7 8 9 10"])
        mock_get_encoding.assert_called_once_with("cl100k_base")
        mock_tokenizer.encode.assert_called_once_with(text)


if __name__ == "_main_":
    unittest.main()