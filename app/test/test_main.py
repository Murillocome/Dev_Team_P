# test_main.py
from fastapi.testclient import TestClient
from app.main import app  # Asegúrate de importar tu app desde main.py


def test_read_main():
    # Se crea el cliente de test
    client = TestClient(app)

    # Realiza una solicitud GET a la ruta raíz ("/") o a cualquier otra ruta definida en tu API
    response = client.get("/")

    # Verifica que la respuesta tenga el código de estado 200
    assert response.status_code == 200

    # Verifica que el contenido de la respuesta sea el esperado
    # (Modifica esta verificación según lo que se espera en la respuesta)
    assert response.json() == {"message": "Bienvenido a la API de RG"}  # Asegúrate de que coincida con la respuesta esperada.
