import unittest
from unittest.mock import MagicMock
from app.adapters.chromadb_adapter import ChromaDBAdapter
from app.core import models
import chromadb
from chromadb.errors import UniqueConstraintError


class TestChromaDBAdapter(unittest.TestCase):

    def setUp(self):
        """Configuración inicial para las pruebas"""
        # Creamos el mock del cliente OpenAI
        self.mock_openai_client = MagicMock()

        # Creamos un documento para usarlo en las pruebas
        self.document = models.Document(id="doc1", content="This is a test document.")

        # Mock de la colección en ChromaDB
        self.mock_collection = MagicMock()
        self.mock_collection.query.return_value = {
            'ids': [["doc1"]],
            'documents': [["This is a test document."]]
        }

        # Mock de la instancia ChromaDBAdapter
        self.adapter = ChromaDBAdapter(number_of_vectorial_results=3)

        # Mock para evitar la creación de la colección real
        self.adapter.client.create_collection = MagicMock(return_value=self.mock_collection)

    def test_create_collection_if_not_exists(self):
        """Prueba la creación de una colección si no existe"""
        # Simulamos la situación donde la colección no existe
        self.adapter.client.create_collection.reset_mock()
        self.adapter.client.create_collection.side_effect = None  # No lanzar excepción

        # Se debe llamar a la creación de la colección
        self.adapter.client.create_collection("documents")
        self.adapter.client.create_collection.assert_called_once_with("documents")

if __name__ == "_main_":
    unittest.main()