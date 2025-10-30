from fastapi import FastAPI
from app.controllers.conversation_controller import session_router
from app.controllers.rag_controller_v1 import router as rag_router_v1
from app.controllers.rag_controller_v2 import router as rag_router_v2
from app.core.db import init_db
from app.core.config import settings as server_settings
from starlette.middleware.cors import CORSMiddleware
from app.utils.logger import logger

app = FastAPI(title=server_settings.PROJECT_NAME, description="Chat with multiple wiki articles :articles:")

# Set all CORS enabled origins
if server_settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(session_router, tags=["Sessions | Conversations"])
app.include_router(rag_router_v1, prefix="/v1", tags=["Rag V1: Index, Search, Ask, Chat"])
app.include_router(rag_router_v2, prefix="/v2", tags=["Rag V2: Index, Search, Ask, Chat"])


@app.on_event("startup")
def on_startup():
    logger.info("Starting up app...")
    init_db()
    logger.info("Startup complete.")


@app.get("/")
async def root():
    return {"message": "Server is running. Get /docs to see the endpoints."}
