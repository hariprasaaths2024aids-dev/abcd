import os
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List
from embedding import process_documents

router = APIRouter()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post("/run", response_model=QueryResponse)
def run_query(
    payload: QueryRequest,
    authorization: str = Header(...)
):
    expected_token = os.getenv("team_token")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_token = authorization.replace("Bearer ", "")
    if provided_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")

    answers = process_documents(payload.documents, payload.questions)
    return {"answers": answers}