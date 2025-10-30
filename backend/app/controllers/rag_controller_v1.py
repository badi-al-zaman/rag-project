from fastapi import APIRouter, HTTPException, Depends

import chromadb
from chromadb.config import Settings

from app.core.db import SessionDep
from app.services.embedding_service import load_and_chunk_documents
from app.services.embedding_service import setup_vector_database, process_user_query
from app.services.retriever_service import search_vector_database, search_query_pipline
from app.services.generator_service import llm_chat, ask_agent_v1
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
async def chat(query: str, session_id: str, db: SessionDep, activate_search: bool = False, ):
    new_message = conversation_crud.add_message(db, session_id, content=query, role="user", tokens=None, extra=None)
    if activate_search:
        search_results = search_query_pipline(query)
        conversation_crud.attach_retrieved_docs(db, new_message.message_id, search_results)
    session = conversation_crud.get_full_session(db, session_id)
    response = await llm_chat(query, session)
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


@router.post("/agent/chat/{session_id}")
async def chat_with_agent(query: str, session_id: str, db: SessionDep):
    new_message = conversation_crud.add_message(db, session_id, content=query, role="user", tokens=None, extra=None)
    # search_results = search_query_pipline(query)
    # conversation_crud.attach_retrieved_docs(db, new_message.message_id, search_results)
    session = conversation_crud.get_full_session(db, session_id)
    response = await ask_agent_v1(query, session)
    # conversation_crud.add_message(db, session_id, content=response, role="assistant", tokens=None, extra=None)
    return {"response": response}

    chat_memory = Memory.from_defaults(
        token_limit=3000,
        async_engine=pg_engine,
        session_id=session_id,
        table_name="memory_table",
    )
    print(chat_memory)

    response = await ask_agent(query, chat_memory)
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
