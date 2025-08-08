from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as query_router

app = FastAPI(
    title="LLM Document Query System",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1/hackrx")