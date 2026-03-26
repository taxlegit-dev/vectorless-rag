import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from api.db import init_db
from api.rag_documents import router as rag_documents_router
from api.rag_query import router as rag_query_router

app = FastAPI(title="PageIndex RAG API")  # API server create karta hai.

_cors_env = os.getenv("CORS_ORIGINS", "")  # Environment variable se allowed domains read karta hai.
if _cors_env.strip():  # Agar .env me CORS defined hai -> use karo.
    _origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:  # Default frontend URLs.
    _origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(  # CORS middleware add karta hai.
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")  # Server start hone par:
# -> database tables create ho jate hain
def _startup() -> None:
    init_db()


app.include_router(rag_documents_router)
app.include_router(rag_query_router)
