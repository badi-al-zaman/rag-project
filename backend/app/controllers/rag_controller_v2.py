from fastapi import APIRouter, HTTPException

from app.services.embedding_service import load_and_chunk_documents_v2
from app.services.embedding_service import setup_vector_database_v2
from app.services.retriever_service import search_query_pipline_v2
from app.services.generator_service import llm_chat_v2, ask_agent
from app.services.rag_service import run_complete_rag_pipeline_v2
from app.core.config import settings as server_settings

router = APIRouter()


@router.post("/index")
def index_documents():
    # Step 1: Setup vector database client
    vector_store = setup_vector_database_v2()

    # Step 2: Load and chunk documents to the vector client
    load_and_chunk_documents_v2(vector_store)
    return {"message": "documents Indexed successfully"}


@router.post("/search")
def search(query: str):
    return search_query_pipline_v2(query)


@router.post("/ask")
def ask(query: str):
    response = run_complete_rag_pipeline_v2(query)
    return {"response": response}


from llama_index.core.llms import ChatMessage
# from llama_index.core.memory import ChatMemoryBuffer # deprecated
from llama_index.core.memory import Memory
from app.core.db import async_engine as pg_engine


@router.post("/chat/{session_id}")
async def chat(query: str, session_id: str):
    chat_memory = Memory.from_defaults(
        token_limit=3000,
        async_engine=pg_engine,
        # async_database_uri=str(server_settings.SQLALCHEMY_ASYN_DATABASE_URI),
        session_id=session_id,
        table_name="memory_table",
    )

    response = await llm_chat_v2(query, chat_memory)
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
async def chat_with_agent(query: str, session_id: str):
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
