from fastapi import APIRouter, Depends, HTTPException, status

from app.models import user_crud
from app.core.db import SessionDep
from app.models.user_models import UserCreate, User

user_router = APIRouter()


@user_router.post("/")
def create_user(user: UserCreate, db: SessionDep) -> User:
    user = user_crud.get_user_by_email(session=db, email=user.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    return user_crud.create_user(db, user)
