from fastapi import FastAPI
# from app.controllers.user_controller import user_router
from app.controllers.conversation_controller import session_router
from app.controllers.rag_controller_v1 import router as rag_router_v1
from app.controllers.rag_controller_v2 import router as rag_router_v2
from app.core.db import init_db

app = FastAPI()

# app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(session_router, tags=["Sessions | Conversations"])
app.include_router(rag_router_v1, prefix="/v1", tags=["Rag V1: Index, Search, Ask, Chat"])
app.include_router(rag_router_v2, prefix="/v2", tags=["Rag V2: Index, Search, Ask, Chat"])


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
async def root():
    return {"message": "Hello World!!!"}
