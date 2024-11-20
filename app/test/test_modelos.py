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



