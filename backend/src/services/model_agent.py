# models.py
import numpy as np

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from typing import List
from .vector_store import query_faiss
from src.core.logger import get_logger

logger = get_logger(__name__)

class ModelAgent:
    _embedding_model = None
    _embedding_model_dimension = -1
    _llm_model = None

    _user_persona: str = ""
    _user_job_to_do: str = ""
    _user_query_embedding: List[float] = []

    @classmethod
    def initialize(cls):
        if cls._embedding_model is None:
            logger.info("----- Loading embedding model...")
            
            cls._embedding_model = SentenceTransformer(
                "./models/all-MiniLM-L12-v2", device='cpu'
            )
            cls._embedding_model.max_seq_length = 256
            cls._embedding_model_dimension = cls._embedding_model.get_sentence_embedding_dimension()

            logger.info(f"Embedding Model Dimensions: {cls._embedding_model_dimension}")

        if cls._llm_model is None:
            logger.info("----- Loading LLM model...")
            cls._llm_model = Llama(
                model_path="./models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
                n_ctx=2048,
                n_threads=8,
                n_batch=512
            )

    @classmethod
    def refine_text(cls, prompt: str) -> str:
        if cls._llm_model is None:
            raise RuntimeError("LLM model not initialized")

        full_prompt = f"Refine and format this input clearly and cleanly:\n\n{prompt.strip()}\n\nRefined output:"

        output = cls._llm_model(full_prompt, max_tokens=256)
        return output["choices"][0]["content"].strip()

    @classmethod
    def embed_chunks(cls, chunks: list[dict]) -> list[dict]:
        """Generate vector embeddings for a list of text chunks."""
        if cls._embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        content = [chunk["content"] for chunk in chunks]
        vectors = cls._embedding_model.encode(content, batch_size=32, show_progress_bar=True)

        return [
            {
                "vector": vector.tolist(),
                "metadata": chunk["metadata"],
                "content": chunk["content"]
            }
            for chunk, vector in zip(chunks, vectors)
        ]

    @classmethod
    def set_user_query(cls, persona: str, job: str):

        cls._user_persona = persona
        cls._user_job_to_do = job

        combined_query = f"Persona: {persona}. Task: {job}"
        cls._user_query_embedding = cls._embedding_model.encode(combined_query, convert_to_tensor=True)
        logger.info(f"✓ User query set: \"{combined_query}\"")
        logger.info(f"→ Query vector shape: {cls._user_query_embedding.shape}")

    @classmethod
    def get_user_query_vector(cls):
        logger.info(f"!> User Job-to-do Vector: {cls._user_query_embedding}")
        return cls._user_query_embedding
    
    @classmethod
    def query_documents(cls, top_k: int = 5):
        """
        Query the FAISS index using the current user query embedding.

        Args:
            top_k (int): Number of top results to retrieve.

        Returns:
            list[dict]: Ranked search results with metadata and distances.
        """
        if cls._user_query_embedding is None or len(cls._user_query_embedding) == 0:
            raise RuntimeError("User query vector not set. Call set_user_query() first.")

        # Ensure numpy array in correct shape
        query_vec = np.array(cls._user_query_embedding, dtype="float32")
        results = query_faiss(query_vec, top_k=top_k)

        # for r in results:
        #     logger.info(f"[{r['rank']}] {r['pdf']} (chunk {r['chunk_id']}) → {r['distance']:.4f}")
        #     logger.info(r["content"])
        #     logger.info("----")

        return results
