from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.services.vector_store import VectorStore
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
vector_store = VectorStore()

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

@router.post("/")
async def query_document(query: QueryRequest):
    try:
        logger.info(f"Processing query: {query.question}")
        
        # Get query embedding
        query_embedding = vector_store.model.encode(query.question)
        
        # Search in vector store using class method
        search_results = vector_store.query_faiss(query_embedding, top_k=3)
        
        if not search_results:
            return {
                "answer": "No relevant information found in the documents.",
                "confidence": 0.0,
                "source_text": None
            }

        # Process results
        best_match = search_results[0]
        confidence = 1.0 - min(best_match["distance"], 1.0)
        
        # Combine context from top results
        context = "\n".join([r["text"] for r in search_results[:2]])
        
        # Format response
        answer = f"Based on the document: {context[:200]}..."
        
        return {
            "answer": answer,
            "confidence": confidence,
            "source_text": context[:1000]  # Limit context size
        }

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
async def health_check():
    return {"status": "healthy"}
@router.get("/health")
async def health_check():
    return {"status": "healthy"}
