import pytest
from unittest.mock import MagicMock
from app.core.models import Document
from app.adapters.chromadb_adapter import ChromaDBAdapter  # Asegúrate de que la ruta es correcta

# Fixture para inicializar el adaptador con un número de resultados predeterminado
@pytest.fixture
def chroma_db_adapter():
    return ChromaDBAdapter(number_of_vectorial_results=5)

# Fixture para simular el cliente de OpenAI
@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.embed_text.return_value = [[0.1, 0.2, 0.3, 0.4]]  # Simula un embedding de prueba
    return client

def test_save_document(chroma_db_adapter, mock_openai_client):
    """Prueba que el documento se guarde en ChromaDB con embeddings generados."""
    document = Document(
        document_id="test_doc_123",
        nombre="test_document.txt",
        ruta="/media/test_document.txt"
    )
    content = "Este es un contenido de prueba para embeddings."

    # Guarda el documento
    chroma_db_adapter.save_document(document, content, mock_openai_client)

    # Verifica que el documento se haya guardado en ChromaDB
    data = chroma_db_adapter.get_vectors()
    assert "test_doc_123" in data['ids']
    assert data['documents'][0] == content
    assert isinstance(data['embeddings'][0], list)  # Confirma que el embedding es una lista

def test_get_documents(chroma_db_adapter, mock_openai_client):
    """Prueba la recuperación de documentos relevantes basados en la consulta."""
    # Crea un documento de prueba y guárdalo
    document = Document(
        document_id="test_doc_456",
        nombre="relevant_doc.txt",
        ruta="/media/relevant_doc.txt"
    )
    content = "Este es el contenido relevante para una consulta específica."
    chroma_db_adapter.save_document(document, content, mock_openai_client)

    # Realiza una búsqueda con una consulta de prueba
    query = "consulta específica"
    retrieved_documents = chroma_db_adapter.get_documents(query, mock_openai_client, n_results=1)

    # Verifica que se haya recuperado el documento relevante
    assert len(retrieved_documents) == 1
    assert retrieved_documents[0].document_id == "test_doc_456"
    assert retrieved_documents[0].content == content

def test_get_vectors(chroma_db_adapter, mock_openai_client):
    """Prueba que `get_vectors` devuelva todos los documentos y embeddings en el sistema."""
    # Guarda un documento para asegurarse de que haya datos en la colección
    document = Document(
        document_id="test_doc_789",
        nombre="vector_doc.txt",
        ruta="/media/vector_doc.txt"
    )
    content = "Este documento se usa para probar get_vectors."
    chroma_db_adapter.save_document(document, content, mock_openai_client)

    # Recupera los datos almacenados en ChromaDB
    data = chroma_db_adapter.get_vectors()

    # Validaciones
    assert isinstance(data, dict)
    assert "ids" in data
    assert "embeddings" in data
    assert "documents" in data
    assert len(data["ids"]) > 0  # Asegura que haya al menos un documento
    assert data["documents"][0] == content
