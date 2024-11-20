import pytest
from unittest.mock import MagicMock
from app.core import models
from app.core.ports import DocumentRepositoryPort

class TestDocumentRepositoryPort:
    @pytest.fixture
    def mock_document_repo(self):
        # Mock de un repositorio de documentos
        mock_repo = MagicMock(spec=DocumentRepositoryPort)
        return mock_repo

    def test_save_document(self, mock_document_repo):
        # Preparar datos de prueba
        document = models.Document(document_id="123", content="Test document content")
        mock_openai_client = MagicMock()

        # Ejecutar método
        mock_document_repo.save_document(document, "Test content", mock_openai_client)

        # Verificar que el método save_document se haya llamado correctamente
        mock_document_repo.save_document.assert_called_once_with(document, "Test content", mock_openai_client)

    def test_get_documents(self, mock_document_repo):
        # Preparar datos de prueba
        mock_openai_client = MagicMock()
        mock_document_repo.get_documents.return_value = [models.Document(document_id="123", content="Test document content")]

        query = "Test query"

        # Ejecutar método
        result = mock_document_repo.get_documents(query, mock_openai_client)

        # Verificar que el método get_documents se haya llamado correctamente
        mock_document_repo.get_documents.assert_called_once_with(query, mock_openai_client)
        assert len(result) == 1
        assert result[0].document_id == "123"
        assert result[0].content == "Test document content"

    def test_get_vectors(self, mock_document_repo):
        # Preparar datos de prueba
        mock_document_repo.get_vectors.return_value = {'ids': ["123"], 'embeddings': [[0.1, 0.2]], 'documents': ["Test document content"]}

        # Ejecutar método
        result = mock_document_repo.get_vectors()

        # Verificar que el método get_vectors se haya llamado correctamente
        mock_document_repo.get_vectors.assert_called_once()

        assert result == {'ids': ["123"], 'embeddings': [[0.1, 0.2]], 'documents': ["Test document content"]}
