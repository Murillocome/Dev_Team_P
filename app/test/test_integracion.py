import pytest
from pymongo import MongoClient
from app.core.models import Document  # Asegúrate que esta ruta es correcta
from app.adapters.database_adapter import MongoDbAdapter  # Ajusta si la ruta es distinta

# URL del contenedor de Docker (cambia según tu configuración)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "test_db"  # Usaremos una base temporal para las pruebas

@pytest.fixture(scope="module")
def mongo_client():
    """Conecta a MongoDB y devuelve el cliente."""
    client = MongoClient(MONGO_URI)
    yield client
    client.drop_database(DB_NAME)  # Limpia la base de datos después de las pruebas
    client.close()

@pytest.fixture
def db_adapter(mongo_client):
    """Crea una instancia del adaptador para la base de datos."""
    return MongoDbAdapter(client=mongo_client, db_name=DB_NAME)

