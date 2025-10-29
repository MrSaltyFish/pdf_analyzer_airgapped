from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import os

from src.services.vector_store import VectorStore
from src.core.logger import get_logger
from src.core.config import MISTRAL_BACKEND_URL, UPLOAD_DIR

logger = get_logger(__name__)
router = APIRouter()
vector_store = VectorStore()


# ====== Request Schema ======
class QueryPayload(BaseModel):
    query: str
    persona: Optional[str] = "General user"
    top_k: Optional[int] = 5


# ====== Unified RAG → Mistral endpoint ======
@router.post("/")
async def rag_query(payload: QueryPayload):
    """
    Handles a user query end-to-end:
      1. Retrieve top-k relevant document chunks from FAISS.
      2. Build a contextualized prompt.
      3. Send to Mistral backend for synthesis.
      4. Return the final generated answer.
    """
    try:
        logger.info(f"Received query: '{payload.query}' (persona: {payload.persona})")

        # Step 1: Verify PDFs exist (so we have something to query)
        pdf_files = [os.path.join(UPLOAD_DIR, f)
                     for f in os.listdir(UPLOAD_DIR)
                     if f.endswith(".pdf")]
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No uploaded PDF files found.")

        # Step 2: Encode query and search FAISS
        query_embedding = vector_store.model.encode(payload.query)
        search_results = vector_store.query_faiss(query_embedding, top_k=payload.top_k)

        if not search_results:
            return {
                "final_answer": "No relevant information found in your uploaded documents.",
                "context_used": None,
                "sources": pdf_files,
                "confidence": 0.0
            }

        # Step 3: Prepare top contextual snippets
        context = "\n".join([r["text"] for r in search_results])

        # Step 4: Construct the intelligent prompt for Mistral
        mistral_prompt = f"""
        You are an AI assistant for a user with persona "{payload.persona}".
        User question: {payload.query}

        Here are the most relevant excerpts from their uploaded documents:
        {context}

        Please answer concisely, factually, and in a professional tone.
        """

        # Step 5: Forward to Mistral backend
        mistral_response = requests.post(
            f"{MISTRAL_BACKEND_URL}/chat/simple",
            json={"query": mistral_prompt},
            # timeout=30
        )

        if mistral_response.status_code != 200:
            logger.error(f"Mistral backend error: {mistral_response.text}")
            raise HTTPException(status_code=mistral_response.status_code,
                                detail="Mistral backend failed")

        mistral_json = mistral_response.json()
        model_answer = (
            mistral_json.get("answer") or
            mistral_json.get("response") or
            str(mistral_json)
        )

        # Step 6: Return unified JSON for frontend
        return {
            "final_answer": model_answer.strip(),
            "context_used": context[:1000],  # short preview
            "sources": pdf_files,
            "top_chunks": search_results,
        }

    except Exception as e:
        logger.exception(f"RAG query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
