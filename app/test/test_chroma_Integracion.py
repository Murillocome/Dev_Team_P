import pytest
from unittest.mock import MagicMock
from app.adapters.chromadb_adapter import ChromaDBAdapter
from app.core.models import Document

@pytest.fixture
def chroma_db_adapter():
    """Inicializa el adaptador de ChromaDB."""
    return ChromaDBAdapter(number_of_vectorial_results=5)

@pytest.fixture
def openai_client_mock():
    """Crea un cliente de OpenAI simulado."""
    client = MagicMock()
    client.embed_text.side_effect = lambda text: [[0.1, 0.2, 0.3, 0.4]]  # Embedding fijo para cualquier texto
    return client

def test_integration_save_and_query_document(chroma_db_adapter, openai_client_mock):
    """Prueba la integración completa de guardar y consultar documentos."""

    # Crea un documento de prueba
    document = Document(
        document_id="doc_test_1",
        nombre="test_document.txt",
        ruta="/test/path/test_document.txt"
    )
    content = "Este es un documento de prueba para ChromaDB."

    # Guarda el documento en ChromaDB
    chroma_db_adapter.save_document(document, content, openai_client_mock)

    # Verifica que el documento fue almacenado correctamente
    data = chroma_db_adapter.get_vectors()
    assert len(data['ids']) == 1
    assert data['ids'][0] == "doc_test_1"
    assert data['documents'][0] == content
    assert isinstance(data['embeddings'][0], list)  # Verifica que el embedding es una lista
    assert data['embeddings'][0] == [0.1, 0.2, 0.3, 0.4]

    # Realiza una consulta en la base de datos
    query = "Documento de prueba"
    results = chroma_db_adapter.get_documents(query, openai_client_mock, n_results=1)

    # Verifica que el resultado contiene el documento correcto
    assert len(results) == 1
    assert results[0].document_id == "doc_test_1"
    assert results[0].content == content
