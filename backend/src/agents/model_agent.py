import numpy as np
from threading import Lock
from typing import List, Dict, Any

from src.core.logger import get_logger
from src.core import config
from src.agents.embedding_agent import EmbeddingAgent
from src.agents.llm_agent import LLMAgent
from src.services.vector_store import query_faiss

logger = get_logger(__name__)
_lock = Lock()

class ModelAgent:
    """Coordinates LLM + Embedding models for user queries."""
    _instance = None

    
    def __init__(self):
        self.embedding_agent = EmbeddingAgent()
        self.llm_agent = LLMAgent()
        self._user_query_embedding = None

    # --- Singleton (Optional, avoids reloading models per request) ---
    @classmethod
    def instance(cls):
        with _lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --- Query logic ---
    def set_user_query(self, persona: str, job: str):
        combined = f"Persona: {persona}. Task: {job}"
        logger.info(f"🧠 Creating embedding for user query → {combined}")
        self._user_query_embedding = self.embedding_agent.encode(combined)
        logger.info(f"✅ User query vector shape={self._user_query_embedding.shape}")

    def query_documents(self, top_k=config.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        if self._user_query_embedding is None:
            raise RuntimeError("User query not set. Call set_user_query() first.")
        logger.info("🔍 Querying FAISS...")
        return query_faiss(self._user_query_embedding, top_k=top_k)

    def refine_text(self, text: str) -> str:
        logger.info("💬 Sending text to LLM for refinement...")
        prompt = f"Refine and format this text clearly:\n\n{text.strip()}\n\nRefined output:"
        return self.llm_agent.generate(prompt)
