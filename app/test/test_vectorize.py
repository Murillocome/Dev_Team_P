import unittest
from unittest.mock import MagicMock
from app.helpers.vectorize_documents import get_openai_embeddings, document_to_vectors, chunk_text


class TestVectorizeDocument(unittest.TestCase):

    # Prueba para la función chunk_text
    def test_chunk_text(self):
        text = "This is a test document that is quite long and will be chunked into multiple pieces."
        max_tokens = 10
        chunks = chunk_text(text, max_tokens)

        # Verifica que el texto se ha dividido en fragmentos correctos
        self.assertTrue(len(chunks) > 1)  # Debe haber más de un fragmento
        self.assertTrue(
            all(len(t.split()) <= max_tokens for t in chunks))  # Verifica que no haya fragmentos con más de 10 palabras

    # Prueba para la función document_to_vectors

    # Prueba para la función get_openai_embeddings
    def test_get_openai_embeddings(self):
        # Simular respuesta del cliente de OpenAI
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.4, 0.5, 0.6])]  # Simula la estructura de la respuesta
        mock_client._client.embeddings.create.return_value = mock_response

        text = "This is a test"
        embeddings = get_openai_embeddings(text, mock_client)

        # Verifica que los embeddings sean los esperados
        self.assertEqual(embeddings, [0.4, 0.5, 0.6])
        mock_client._client.embeddings.create.assert_called_once_with(
            input=text, model="text-embedding-ada-002"
        )


if __name__ == "_main_":
    unittest.main()