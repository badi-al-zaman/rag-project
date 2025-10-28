# models.py
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field, Relationship, Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON


class Session(SQLModel, table=True):
    session_id: UUID = Field(default_factory=uuid4, primary_key=True)
    # user_id: UUID = Field(foreign_key="user.user_id", nullable=False, ondelete="CASCADE")
    title: str = "default_session_title"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    # user: "User" = Relationship(back_populates="sessions")
    messages: list["Message"] = Relationship(back_populates="session", cascade_delete=True)


class Message(SQLModel, table=True):
    message_id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.session_id", nullable=False, ondelete="CASCADE")
    role: str  # "user" | "assistant" | "system" | "tool"| "function"
    content: str
    tokens: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    session: Session = Relationship(back_populates="messages")
    retrieved_docs: list["MessageRetrievedDoc"] = Relationship(back_populates="message", cascade_delete=True)


class RetrievedDoc(SQLModel, table=True):
    retrieved_doc_id: UUID = Field(default_factory=uuid4, primary_key=True)
    source_doc_id: str = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: float = None
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # Prevent duplicate snippets from same source
        UniqueConstraint("source_doc_id", "snippet", name="uq_source_snippet"),
    )

    linked_messages: list["MessageRetrievedDoc"] = Relationship(back_populates="retrieved_doc", cascade_delete=True)


class MessageRetrievedDoc(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    message_id: UUID = Field(foreign_key="message.message_id", ondelete="CASCADE")
    retrieved_doc_id: UUID = Field(foreign_key="retrieveddoc.retrieved_doc_id", ondelete="CASCADE")
    rank: Optional[int] = None

    message: Message = Relationship(back_populates="retrieved_docs")
    retrieved_doc: RetrievedDoc = Relationship(back_populates="linked_messages")
