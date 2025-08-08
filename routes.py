from fastapi import APIRouter, HTTPException, status, Header
from pydantic import BaseModel
from typing import List
import tempfile
import requests
import os

from embedding import load_document, create_vectorstore
from decision import evaluate_with_llm

router = APIRouter()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post("/run", response_model=QueryResponse)
def run_query(payload: QueryRequest, authorization: str = Header(...)):
    expected_token = os.getenv("team_token")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
    token = authorization.split(" ")[1]
    if token != expected_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Bearer token")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = requests.get(payload.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        docs = load_document(tmp_path)
        vectorstore = create_vectorstore(docs)

        results = []
        for q in payload.questions:
            try:
                raw_answer = evaluate_with_llm(q, vectorstore)
                flat = raw_answer.get("justification", "No justification provided.")
                results.append(flat)
            except Exception as e:
                results.append(f"Error: {str(e)}")

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.get("/debug")
def debug():
    return {
        "env_team_token_set": bool(os.getenv("team_token")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "team_token_value": os.getenv("team_token"),
        "groq_key_prefix": os.getenv("GROQ_API_KEY", "")[:10] + "...",
        "status": "ok"
    }