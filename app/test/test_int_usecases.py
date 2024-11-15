import pytest
from app.core.models import Document, User, generate_uuid

def test_generate_uuid():
    """Prueba unitaria para verificar la generación de UUIDs."""
    uuid1 = generate_uuid()
    uuid2 = generate_uuid()

    # Verifica que los UUIDs generados sean únicos
    assert uuid1 != uuid2
    assert len(uuid1) == 36  # Longitud del UUID
    assert len(uuid2) == 36  # Longitud del UUID

def test_document_creation():
    """Prueba unitaria para la creación de un Document."""
    document = Document(nombre="Test Document", ruta="path/to/document.txt", content="Contenido del documento.")

    # Verificar que se asigna un document_id automáticamente
    assert document.document_id is not None
    assert len(document.document_id) == 36  # Verificar la longitud del UUID

    # Verificar los otros campos
    assert document.nombre == "Test Document"
    assert document.ruta == "path/to/document.txt"
    assert document.content == "Contenido del documento."

def test_user_creation():
    """Prueba unitaria para la creación de un User."""
    user = User(username="testuser", password="securepassword")

    # Verificar que los campos del usuario estén correctamente asignados
    assert user.id is None  # El ID debe ser None por defecto
    assert user.username == "testuser"
    assert user.password == "securepassword"