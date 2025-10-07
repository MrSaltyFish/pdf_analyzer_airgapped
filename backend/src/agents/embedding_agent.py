import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.logger import get_logger
from src.core import config

logger = get_logger(__name__)

class EmbeddingAgent:
    """Handles all embedding operations."""

    def __init__(self):
        logger.info(f"Loading embedding model from: {config.EMBEDDING_MODEL_PATH}")
        self.model = SentenceTransformer(str(config.EMBEDDING_MODEL_PATH), device=config.EMBEDDING_DEVICE)
        self.model.max_seq_length = config.CHUNK_SIZE
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded ({self.dim} dims, device={config.EMBEDDING_DEVICE})")

    def encode(self, texts, normalize=True):
        """Encode list of strings into embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        return np.array(
            self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=config.VERBOSE,
                normalize_embeddings=normalize
            ),
            dtype=np.float32
        )
