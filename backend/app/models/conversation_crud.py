# crud.py
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import SQLAlchemyError
from app.models.conversation_models import Session as ChatSession, Message, RetrievedDoc, MessageRetrievedDoc
from typing import List
from datetime import datetime, timezone
from fastapi import HTTPException, status
from app.utils.logger import logger


def create_session(db: Session, title: str = None, user_id=None, metadata=None):
    new_session = ChatSession(title=title, meta=metadata, )  # user_id=user_id
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def get_full_session(db: Session, session_id: str):
    try:
        query = (
            select(ChatSession)
            .where(ChatSession.session_id == session_id)
            .options(
                selectinload(ChatSession.messages)
                .selectinload(Message.retrieved_docs)
                .selectinload(MessageRetrievedDoc.retrieved_doc)
            )
        )

        session_obj = db.exec(query).one_or_none()
        if not session_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with id={session_id} not found."
            )

        # Build response
        result = {
            "session_id": str(session_obj.session_id),
            "title": session_obj.title,
            "created_at": session_obj.created_at,
            "last_active_at": session_obj.last_active_at,
            "messages": []
        }

        for msg in sorted(session_obj.messages, key=lambda m: m.created_at):
            retrieved_docs = []
            for link in msg.retrieved_docs:
                if link.retrieved_doc:
                    rd = link.retrieved_doc
                    retrieved_docs.append({
                        "retrieved_doc_id": str(rd.retrieved_doc_id),
                        "title": rd.title,
                        "content": rd.snippet,
                        "score": rd.score,
                        "source_doc_id": rd.source_doc_id,
                        "metadata": rd.meta
                    })

            result["messages"].append({
                "message_id": str(msg.message_id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "retrieved_docs": retrieved_docs
            })
        return result
    except HTTPException:
        # Re-raise HTTPException so FastAPI can handle it properly
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        # Optionally log e for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while adding message."
        )
    except Exception as e:
        db.rollback()
        # Catch any other unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


def get_all_sessions(db: Session, limit: int = 10):
    sessions = db.exec(select(ChatSession).order_by(ChatSession.created_at.desc()).limit(limit)).all()
    return sessions[::-1]


def get_messages(db: Session, session_id: str, limit: int = 50):
    stmt = select(Message).where(Message.session_id == session_id).order_by(Message.created_at.desc()).limit(limit)
    return db.exec(stmt).all()[::-1]  # return ascending order


def add_message(db: Session, session_id, content: str, role="user", tokens=None, extra=None):
    try:
        # Get the chat session
        session = db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with id={session_id} not found."
            )
        # Create the message
        message = Message(session_id=session_id, role=role, content=content, tokens=tokens, extra=extra)
        db.add(message)

        # update session last active
        session.last_active_at = datetime.now(timezone.utc)
        # Commit the transaction
        db.commit()
        db.refresh(message)
        return message
    except HTTPException:
        # Re-raise HTTPException so FastAPI can handle it properly
        db.rollback()
        raise

    except SQLAlchemyError as e:
        logger.exception(f"Unexpected error: {str(e)}")
        db.rollback()
        # Optionally log e for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while adding message."
        )

    except Exception as e:
        db.rollback()
        # Catch any other unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


def attach_retrieved_docs(db: Session, message_id, docs: List[dict]):
    """
    docs: list of {source_doc_id, title, snippet, score, metadata, rank}
    This will insert retrieved_docs and link them to the message.
    """
    try:
        for d in docs:
            existing = db.exec(
                select(RetrievedDoc).where(
                    (RetrievedDoc.source_doc_id == d.get("id")) &
                    (RetrievedDoc.snippet == d.get("content"))
                )
            ).first()

            if existing:
                link = MessageRetrievedDoc(message_id=message_id, retrieved_doc_id=existing.retrieved_doc_id,
                                           rank=d.get("similarity"))
                db.add(link)
                continue
            else:
                rd = RetrievedDoc(
                    source_doc_id=d.get("id"),
                    title=d.get("title"),
                    snippet=d.get("content"),
                    score=d.get("similarity"),
                    meta=d.get("metadata")
                )

                db.add(rd)
                db.commit()
                db.refresh(rd)

                link = MessageRetrievedDoc(message_id=message_id, retrieved_doc_id=rd.retrieved_doc_id,
                                           rank=d.get("similarity"))
                db.add(link)
        db.commit()
    except HTTPException:
        # Re-raise HTTPException so FastAPI can handle it properly
        db.rollback()
        raise

    except SQLAlchemyError as e:
        db.rollback()
        # Optionally log e for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while adding retrieved_docs."
        )

    except Exception as e:
        db.rollback()
        # Catch any other unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
