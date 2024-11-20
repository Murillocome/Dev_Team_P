import unittest
from unittest.mock import MagicMock, patch
from app.helpers.vectorize_documents import document_to_vectors, chunk_text


class TestVectorizeDocuments(unittest.TestCase):

    def setUp(self):
        # Configuración inicial para las pruebas
        self.fake_openai_client = MagicMock()
        self.fake_openai_client._client.embeddings.create.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2, 0.3]}]
        )



    def test_document_to_vectors(self):
        # Prueba para document_to_vectors
        content = "Este es un texto largo que necesita ser dividido en fragmentos."

        # Simula la división del texto en fragmentos
        with patch('app.helpers.vectorize_documents.chunk_text', return_value=["chunk1", "chunk2"]):
            with patch('app.helpers.vectorize_documents.get_openai_embeddings') as mock_get_embeddings:
                mock_get_embeddings.side_effect = [
                    [0.1, 0.2, 0.3],  # Embedding para el primer fragmento
                    [0.4, 0.5, 0.6],  # Embedding para el segundo fragmento
                ]
                result = document_to_vectors(content, self.fake_openai_client)

        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_get_embeddings.assert_any_call("chunk1", self.fake_openai_client)
        mock_get_embeddings.assert_any_call("chunk2", self.fake_openai_client)

    def test_chunk_text(self):
        # Prueba para chunk_text
        text = "Este es un texto largo que será fragmentado."
        max_tokens = 5
        with patch('tiktoken.get_encoding') as mock_get_encoding:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            mock_tokenizer.decode.side_effect = lambda x: " ".join(map(str, x))
            mock_get_encoding.return_value = mock_tokenizer

            result = chunk_text(text, max_tokens)

        self.assertEqual(result, ["1 2 3 4 5", "6 7 8 9 10"])
        mock_get_encoding.assert_called_once_with("cl100k_base")
        mock_tokenizer.encode.assert_called_once_with(text)


if __name__ == "__main__":
    unittest.main()
