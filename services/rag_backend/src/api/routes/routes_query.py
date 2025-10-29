from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import os
from src.services.vector_store import VectorStore
from src.core.logger import get_logger
from src.core.config import MISTRAL_BACKEND_URL

logger = get_logger(__name__)
router = APIRouter()
vector_store = VectorStore()

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

@router.post("/")
async def query_document(query: QueryRequest):
    try:
        query_docs = 5
        logger.info(f"Processing query: {query.question}")

        # Get query embedding
        query_embedding = vector_store.model.encode(query.question)
        
        # Search in vector store using class method
        search_results = vector_store.query_faiss(query_embedding, top_k=query_docs)
        
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
            "search_results": search_results,
            "source_text": None
        }

        # return {
        #     "answer": answer,
        #     "confidence": confidence,
        #     "source_text": context[:1000]  # Limit context size
        # }

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

# ✅ Define the expected JSON body
class ResponseRequest(BaseModel):
    user_query: str
    user_persona: Optional[str] = "General user"
    top_k: Optional[int] = 5

# ========== Endpoint Implementation ==========
@router.post("/response")
async def generate_response(payload: ResponseRequest):
    try:
        logger.info(f"Generating response for query: {payload.user_query}")

        # Step 1: Load files from updated_files folder
        upload_dir = "updated_files"
        pdf_files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".pdf")]

        if not pdf_files:
            raise HTTPException(status_code=400, detail="No uploaded PDF files found in updated_files/.")

        logger.info(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

        # # Step 2: Build or refresh vector store
        # vector_store.build_index_from_pdfs(pdf_files)
        # logger.info("Vector store successfully built/updated.")

        # Step 3: Get embedding for user query
        query_embedding = vector_store.model.encode(payload.user_query)

        # Step 4: Query FAISS index
        search_results = vector_store.query_faiss(query_embedding, top_k= payload.top_k)

        if not search_results:
            return {
                "answer": "No relevant chunks found in uploaded documents.",
                "confidence": 0.0,
                "context": None
            }

        # Step 5: Extract top context for synthesis
        context = "\n".join([r["text"] for r in search_results])

                # Step 4: Build a smart prompt for Mistral
        mistral_prompt = f"""
        You are an AI assistant for a user with persona: "{payload.user_persona or 'General user'}".

        User query: {payload.user_query}

        Context extracted from user's documents:
        {context}

        Based on the above context, generate a clear, factual, and concise answer.
        """

        # Step 5: Send request to Mistral backend
        response = requests.post(
            MISTRAL_BACKEND_URL,
            json={"query": mistral_prompt}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Mistral backend error")

        mistral_data = response.json()
        model_answer = mistral_data.get("answer") or mistral_data.get("response") or str(mistral_data)

        # Step 6: Return full enriched response
        return {
            "final_answer": model_answer,
            "source_context": context[:1000],
            "top_chunks": search_results,
            "source_docs": pdf_files,
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))