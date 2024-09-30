import chromadb
import numpy as np
from typing import List
from app.core import ports
from app.core import models
from app.helpers.vectorize_documents import document_to_vectors


class ChromaDBAdapter(ports.DocumentRepositoryPort):
    def __init__(self, number_of_vectorial_results: int) -> None:
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        self._number_of_vectorial_results = number_of_vectorial_results

    # Guardar un documento con embeddings generados por OpenAI
    def save_document(self, document: models.Document, content: str, openai_client) -> None:
        embeddings_document = document_to_vectors(content, openai_client)

        # Si hay más de un embedding, combinarlo promediando
        if len(embeddings_document) > 1:
            combined_embedding = np.mean(embeddings_document, axis=0).tolist()
        else:
            combined_embedding = embeddings_document[0]

        # Agregar el documento a ChromaDB con su embedding
        self.collection.add(
            ids=[document.document_id],
            embeddings=[combined_embedding],  # Aseguramos que sea una lista de embeddings
            documents=[content]
        )

