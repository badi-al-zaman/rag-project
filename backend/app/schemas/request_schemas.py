# schemas.py
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class MessageCreate(BaseModel):
    session_id: UUID
    content: str


class MessageRead(BaseModel):
    message_id: UUID
    session_id: UUID
    role: str
    content: str
    tokens: Optional[int]
    created_at: datetime
    extra: Optional[dict]
    #
    # class Config:
    #     orm_mode = True


class SessionCreate(BaseModel):
    # user_id: UUID
    title: Optional[str] = "default_title"
    metadata: Optional[dict] = None


class SessionRead(BaseModel):
    session_id: UUID
    # user_id: UUID
    title: Optional[str]
    created_at: datetime
    last_active_at: datetime
    metadata: Optional[dict]
    #
    # class Config:
    #     orm_mode = True
