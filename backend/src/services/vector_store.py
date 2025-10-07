import faiss
import numpy as np
import pickle

from src.core.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from src.core.logger import get_logger
logger = get_logger(__name__)

# -------------------------------
# Internal helpers
# -------------------------------
def _load_faiss_index_and_metadata():
    """Load FAISS index and metadata from disk if they exist."""
    if not FAISS_INDEX_PATH.exists() or not FAISS_METADATA_PATH.exists():
        return None, []
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def _save_faiss_index_and_metadata(index, metadata):
    """Persist FAISS index and metadata to disk."""
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

# -------------------------------
# Public API
# -------------------------------
def save_embedded_document_in_faiss(embeddings: np.ndarray, metadatas: list, normalize: bool = True):
    """
    Save document embeddings into FAISS index.

    Args:
        embeddings (np.ndarray): Shape (num_vectors, vector_dim).
        metadatas (list[dict]): Metadata for each vector.
        normalize (bool): Whether to normalize embeddings for cosine similarity.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype="float32")
    else:
        embeddings = embeddings.astype("float32", copy=False)

    if normalize:
        faiss.normalize_L2(embeddings)

    vector_dim = embeddings.shape[1]

    # Load existing index or create new one
    index, metadata_store = _load_faiss_index_and_metadata()
    if index is None:
        index = faiss.IndexFlatL2(vector_dim)
        metadata_store = []

    index.add(embeddings)
    metadata_store.extend(metadatas)

    _save_faiss_index_and_metadata(index, metadata_store)
    logger.debug(f"✅ Added {len(embeddings)} vectors. Total vectors: {index.ntotal}")

def query_faiss(query_vector: list[float] | np.ndarray, top_k: int = 5, normalize: bool = True):
    """
    Query FAISS index for nearest neighbors.

    Args:
        query_vector (list[float] | np.ndarray): The query embedding.
        top_k (int): Number of results to retrieve.
        normalize (bool): Whether to normalize query vector for cosine similarity.

    Returns:
        list[dict]: Ranked search results with metadata and distances.
    """
    index, metadata_store = _load_faiss_index_and_metadata()
    if index is None:
        raise FileNotFoundError("FAISS index or metadata not found.")

    # Ensure it's a 2D float32 array
    query_vector = np.asarray(query_vector, dtype="float32")
    if query_vector.ndim == 1:  # single vector → make it shape (1, d)
        query_vector = query_vector[np.newaxis, :]

    if normalize:
        faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < len(metadata_store):
            results.append({
                "rank": rank + 1,
                "distance": float(distances[0][rank]),
                **metadata_store[idx]
            })

    return results
