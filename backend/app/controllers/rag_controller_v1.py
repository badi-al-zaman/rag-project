from fastapi import APIRouter, HTTPException, Depends

import chromadb
from chromadb.config import Settings

from app.core.db import SessionDep
from app.services.embedding_service import load_and_chunk_documents
from app.services.embedding_service import setup_vector_database, process_user_query
from app.services.retriever_service import search_vector_database
from app.services.generator_service import llm_chat
from app.services.rag_service import run_complete_rag_pipeline
from app.models.conversation_models import Session as SessionModel
import app.models.conversation_crud as conversation_crud

from pydantic import BaseModel
from typing import List

router = APIRouter()


class Document(BaseModel):
    id: str
    title: str
    content: str
    similarity: float


def search_query_pipline(query: str):
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient("vector_db/chroma", settings=Settings(anonymized_telemetry=False))

    # Create collection (what is the collection name? | What similarity metric is used? | embedding_functions)
    collection = chroma_client.get_or_create_collection(name="wiki_articles_v1", metadata={
        "hnsw:space": "cosine"}, )

    # index documents if they are not indexed before
    if collection.count() == 0:
        # Step 1: Load and chunk documents to the vector client
        chunks = load_and_chunk_documents()

        # Step 2: Setup vector database client
        setup_vector_database(chunks)

    # Step 3: Process user query
    model, query_embedding = process_user_query(query)

    # Step 4: Search vector database
    search_results = search_vector_database(collection, query_embedding, top_k=3)
    return search_results


@router.post("/index")
def index_documents():
    # Step 1: Load and chunk documents to the vector client
    chunks = load_and_chunk_documents()

    # Step 2: Setup vector database client
    setup_vector_database(chunks)

    return {"message": "documents Indexed successfully"}


@router.post("/search")
def search(query: str):
    return search_query_pipline(query)


@router.post("/ask")
def ask(query: str):
    response = run_complete_rag_pipeline(query)
    return {"response": response}


@router.post("/chat")
def chat(query: str, session_id: str, db: SessionDep, activate_search: bool = False, ):
    new_message = conversation_crud.add_message(db, session_id, content=query, role="user", tokens=None, extra=None)
    if activate_search:
        search_results = search_query_pipline(query)
        conversation_crud.attach_retrieved_docs(db, new_message.message_id, search_results)
    session = conversation_crud.get_full_session(db, session_id)
    response = llm_chat(query, session)
    conversation_crud.add_message(db, session_id, content=response, role="assistant", tokens=None, extra=None)
    return {"response": response}

    # messages = chat_store.get_messages("conversation1")
    # if len(messages) == 0:
    #     messages = [
    #         ChatMessage(role="user", content="When did Adam graduate from college?"),
    #         ChatMessage(role="chatbot", content="1755."),
    #     ]
    #  Role should be 'system', 'developer', 'user', 'assistant', 'function', 'tool', 'chatbot' or 'model'
    #     chat_store.set_messages("conversation1", messages=messages)

    # session = crud.get_full_session(db, session_id)
