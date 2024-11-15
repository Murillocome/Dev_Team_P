import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import UploadFile
from io import BytesIO
from app.core.models import Document
from app.usecases import RAGService

@pytest.fixture
def mock_db():
    #Mock del puerto de base de datos."""
    db = MagicMock()
    db.save_user = MagicMock()
    db.get_document = MagicMock(return_value=Document(nombre="test_doc", ruta="media/test_doc.txt"))
    db.save_document = MagicMock()
    return db

@pytest.fixture
def mock_document_repo():
    #Mock del repositorio de documents
    repo = MagicMock()
    repo.get_documents = MagicMock(return_value=[Document(nombre="doc1", content="contenido 1")])
    repo.save_document = MagicMock()
    repo.get_vectors = MagicMock(return_value=[1, 2, 3])
    return repo

@pytest.fixture
def mock_llm_adapter():
    #Mock del adaptor de LLM
    adapter = MagicMock()
    adapter.generate_text = MagicMock(return_value="respuesta generada")
    return adapter

@pytest.fixture
def rag_service(mock_db, mock_document_repo, mock_llm_adapter):
    #Instancia del service RAG con dependencias simuladas."""
    return RAGService(db=mock_db, document_repo=mock_document_repo, openai_adapter=mock_llm_adapter)

def test_generate_answer(rag_service):
    #Prueba del metodo generate_answer
    query = "¿Qué es FastAPI?"
    respuesta = rag_service.generate_answer(query)
    assert respuesta == "respuesta generada"

def test_save_document(rag_service):
    #Prueba del metodo save_document.
    file_content = b"Contenido del archivo"
    file = UploadFile(filename="test.txt", file=BytesIO(file_content))
    rag_service.save_document(file)
    rag_service.db.save_document.assert_called_once()
    rag_service.document_repo.save_document.assert_called_once()

def test_sing_up(rag_service):
    #Prueba del metodo sing_up
    rag_service.sing_up("usuario", "contraseña")
    rag_service.db.save_user.assert_called_once_with("usuario", "contraseña")

def test_get_document(rag_service):
    #Prueba del metodo get_document.
    doc = rag_service.get_document("123")
    assert doc.nombre == "test_doc"
    assert doc.ruta == "media/test_doc.txt"

def test_get_vectors(rag_service):
    #Prueba del metodo get_vectors.
    vectors = rag_service.get_vectors()
    assert vectors == [1, 2, 3]