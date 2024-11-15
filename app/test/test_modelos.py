import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.api.routers import rag_router
from app.usecases import RAGService

# Crear un cliente de pruebas utilizando el router de FastAPI
@pytest.fixture
def client():
    # Crear un cliente de prueba de FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    return TestClient(app)

@pytest.fixture
def mock_rag_service():
    """Simula el servicio RAG para pruebas unitarias."""
    rag_service = MagicMock(RAGService)
    return rag_service

def test_save_document(client, mock_rag_service):
    """Prueba unitaria para la ruta save_document."""
    file_content = b"Contenido de prueba"
    mock_file = UploadFile(filename="test_doc.txt", file=file_content)

    # Mockear la dependencia
    rag_service_instance = mock_rag_service
    rag_service_instance.save_document = MagicMock()

    # Hacer la solicitud POST
    response = client.post("/save-document/", files={"file": mock_file})

    # Verificar que el servicio fue llamado
    rag_service_instance.save_document.assert_called_once()

    # Verificar la respuesta
    assert response.status_code == 201
    assert response.json() == {"status": "Document saved successfully"}

def test_generate_answer(client, mock_rag_service):
    """Prueba unitaria para la ruta generate_answer."""
    query = "¿Qué es FastAPI?"
    mock_rag_service.generate_answer.return_value = "respuesta generada"

    # Mockear la dependencia
    rag_service_instance = mock_rag_service

    # Hacer la solicitud GET
    response = client.get("/generate-answer/", params={"query": query})

    # Verificar que el servicio fue llamado
    rag_service_instance.generate_answer.assert_called_once_with(query)

    # Verificar la respuesta
    assert response.status_code == 201
    assert response.json() == "respuesta generada"

def test_get_document(client, mock_rag_service):
    """Prueba unitaria para la ruta get_document."""
    mock_document = {"nombre": "test_doc.txt", "ruta": "media/test_doc.txt"}
    mock_rag_service.get_document.return_value = mock_document

    # Mockear la dependencia
    rag_service_instance = mock_rag_service

    # Hacer la solicitud GET
    response = client.get("/get-document/", params={"document_id": "test_doc.txt"})

    # Verificar que el servicio fue llamado
    rag_service_instance.get_document.assert_called_once_with("test_doc.txt")

    # Verificar la respuesta
    assert response.status_code == 200
    assert response.json() == mock_document

def test_sing_up(client, mock_rag_service):
    """Prueba unitaria para la ruta sing_up."""
    username = "test_user"
    password = "secure_password"

    # Mockear la dependencia
    rag_service_instance = mock_rag_service

    # Hacer la solicitud POST
    response = client.post("/sing-up/", params={"username": username, "password": password})

    # Verificar que el servicio fue llamado
    rag_service_instance.sing_up.assert_called_once_with(username, password)

    # Verificar la respuesta
    assert response.status_code == 201
    assert response.json() == {"status": "User created successfully"}

def test_get_vectors(client, mock_rag_service):
    """Prueba unitaria para la ruta get_vectors."""
    mock_vectors = [{"vector_id": 1, "value": "vector1"}, {"vector_id": 2, "value": "vector2"}]
    mock_rag_service.get_vectors.return_value = mock_vectors

    # Mockear la dependencia
    rag_service_instance = mock_rag_service

    # Hacer la solicitud GET
    response = client.get("/get-vectors/")

    # Verificar que el servicio fue llamado
    rag_service_instance.get_vectors.assert_called_once()

    # Verificar la respuesta
    assert response.status_code == 201
    assert response.json() == mock_vectors