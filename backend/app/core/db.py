# database.py
from sqlmodel import create_engine, Session, select, SQLModel
from psycopg.rows import dict_row
import psycopg

from app.core.config import settings as server_settings
from fastapi import Depends
from typing import Annotated

# from app.models.user_models import User, UserCreate
# from app.models.user_crud import create_user
import app.models.conversation_models
import app.models.document_models

DATABASE_URI = str(server_settings.SQLALCHEMY_DATABASE_URI)
FIRST_SUPERUSER = server_settings.FIRST_SUPERUSER
FIRST_SUPERUSER_PASSWORD = server_settings.FIRST_SUPERUSER_PASSWORD

# psycopg3 works directly with SQLModel's create_engine
engine = create_engine(DATABASE_URI)  # echo=False


def get_db_session():
    with Session(engine) as session:
        yield session


def get_db_connection():
    with psycopg.connect(DATABASE_URI,
                         row_factory=dict_row) as conn:
        yield conn


def init_db() -> None:
    # make sure all SQLModel models are imported (app.models) before initializing DB
    # otherwise, SQLModel might fail to initialize relationships properly
    SQLModel.metadata.create_all(engine)

    # with Session(engine) as session:
    #     user = session.exec(
    #         select(User).where(User.email == FIRST_SUPERUSER)
    #     ).first()
    #     if not user:
    #         user_in = UserCreate(
    #             email=FIRST_SUPERUSER,
    #             password=FIRST_SUPERUSER_PASSWORD,
    #         )
    #         user = create_user(db=session, user=user_in)
    #         print(user)


SessionDep = Annotated[Session, Depends(get_db_session)]
