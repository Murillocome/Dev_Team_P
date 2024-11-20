import unittest
from unittest.mock import MagicMock, patch
from app.core.models import User, Document
from app.adapters.database_adapter import MongoDbAdapter


class TestMongoDbAdapter(unittest.TestCase):
    def setUp(self):
        # Inicializa el adaptador con un cliente simulado
        self.mock_client = MagicMock()
        self.adapter = MongoDbAdapter()
        self.adapter.client = self.mock_client
        self.adapter.db = self.mock_client["rag_db"]
        self.adapter.users = self.adapter.db["users"]
        self.adapter.documents = self.adapter.db["documents"]

    def test_save_user(self):
        # Prueba para save_user
        self.adapter.save_user("test_user", "secure_password")
        self.adapter.users.insert_one.assert_called_once_with(
            {"username": "test_user", "password": "secure_password"}
        )

    def test_get_user(self):
        # Simula el retorno de un usuario
        self.adapter.users.find_one.return_value = {"username": "test_user", "password": "secure_password"}

        result = self.adapter.get_user("test_user")
        self.adapter.users.find_one.assert_called_once_with({"username": "test_user"})
        self.assertEqual(result, User(username="test_user", password="secure_password"))

    def test_save_document(self):
        # Prueba para save_document
        document = Document(document_id="doc1", nombre="test_doc", ruta="/path/to/doc")
        self.adapter.save_document(document)
        self.adapter.documents.insert_one.assert_called_once_with(
            {"document_id": "doc1", "nombre": "test_doc", "ruta": "/path/to/doc"}
        )

    def test_get_document_found(self):
        # Simula que se encuentra un documento
        self.adapter.documents.find_one.return_value = {
            "document_id": "doc1",
            "nombre": "test_doc",
            "ruta": "/path/to/doc",
        }

        result = self.adapter.get_document("doc1")
        self.adapter.documents.find_one.assert_called_once_with({"document_id": "doc1"})
        self.assertEqual(result, Document(document_id="doc1", nombre="test_doc", ruta="/path/to/doc"))

    def test_get_document_not_found(self):
        # Simula que no se encuentra un documento
        self.adapter.documents.find_one.return_value = None

        result = self.adapter.get_document("non_existent_doc")
        self.adapter.documents.find_one.assert_called_once_with({"document_id": "non_existent_doc"})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
