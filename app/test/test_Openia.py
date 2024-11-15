import pytest
from unittest.mock import MagicMock
from app.core.ports import LlmPort
from app.adapters.openai_adapter import OpenAIAdapter

@pytest.fixture
def mock_openai_client():
    # Crea un mock para el cliente de OpenAI
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Generated response"))]
    )
    return mock_client

@pytest.fixture
def openai_adapter(mock_openai_client):
    # Crea una instancia del adaptador con el cliente mock
    adapter = OpenAIAdapter(
        api_key="test-api-key",
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7
    )
    adapter._client = mock_openai_client  # Reemplaza el cliente con el mock
    return adapter

def test_generate_text_success(openai_adapter, mock_openai_client):
    # Prueba que generate_text devuelve la respuesta correcta
    prompt = "What is AI?"
    retrieval_context = "Artificial Intelligence basics"

    result = openai_adapter.generate_text(prompt, retrieval_context)

    # Verifica que el mock fue llamado con los parámetros correctos
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "The context is: Artificial Intelligence basics, please respond to the following question: "},
            {"role": "user", "content": "What is AI?"},
        ],
        max_tokens=100,
        temperature=0.7
    )

    # Verifica que la respuesta sea la esperada
    assert result == "Generated response"

def test_generate_text_no_choices(openai_adapter, mock_openai_client):
    # Simula una respuesta sin elecciones
    mock_openai_client.chat.completions.create.return_value = MagicMock(choices=[])

    prompt = "What is AI?"
    retrieval_context = "Artificial Intelligence basics"

    with pytest.raises(IndexError):
        openai_adapter.generate_text(prompt, retrieval_context)

def test_generate_text_invalid_response(openai_adapter, mock_openai_client):
    # Simula una respuesta mal formada
    mock_openai_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock()])

    prompt = "What is AI?"
    retrieval_context = "Artificial Intelligence basics"

    with pytest.raises(AttributeError):
        openai_adapter.generate_text(prompt, retrieval_context)