import pydantic
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app import usecases
from app.api import dependencies

rag_router = APIRouter()


@rag_router.post("/sing-up/", status_code=201)
def sing_up(username: str, password: str,
            rag_service: usecases.RAGService = Depends(dependencies.RAGServiceSingleton.get_instance)):
    rag_service.sing_up(username, password)
    return {"status": "User created successfully"}


@rag_router.post("/generate-answer/", status_code=200)
def generate_answer(query: str,
                    rag_service: usecases.RAGService = Depends(dependencies.RAGServiceSingleton.get_instance)):
    return {"answer": rag_service.generate_answer(query)}


@rag_router.post("/save-document/", status_code=201)
def save_document(document: DocumentInput,
                  rag_service: usecases.RAGService = Depends(dependencies.RAGServiceSingleton.get_instance)):
    rag_service.save_document(content=document.content)
    return {"status": "Document saved successfully"}
