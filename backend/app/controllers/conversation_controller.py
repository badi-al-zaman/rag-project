from fastapi import APIRouter
from app.core.db import SessionDep
import app.models.conversation_crud as conversation_crud
import app.schemas.request_schemas as schemas

session_router = APIRouter()


@session_router.post("/sessions")
def create_chat_session(payload: schemas.SessionCreate, db: SessionDep):
    return conversation_crud.create_session(db, **payload.model_dump())


@session_router.get("/sessions/{session_id}")
def get_session_messages(session_id: str, db: SessionDep, limit: int = 50):
    return conversation_crud.get_full_session(db, session_id)


@session_router.post("/sessions/{session_id}/messages")
def add_message(msg: schemas.MessageCreate, db: SessionDep):
    return conversation_crud.add_message(db, **msg.model_dump())
