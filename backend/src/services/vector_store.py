import gc
from typing import Dict, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import uuid
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
    if normalize:
        faiss.normalize_L2(embeddings)

    vector_dim = embeddings.shape[1]
    
    # Use IndexIVFFlat for better memory efficiency with large datasets
    index, metadata_store = _load_faiss_index_and_metadata()
    if index is None:
        quantizer = faiss.IndexFlatL2(vector_dim)
        index = faiss.IndexIVFFlat(quantizer, vector_dim, min(embeddings.shape[0], 100))
        index.train(embeddings)
        metadata_store = []

    # Add vectors in smaller batches
    batch_size = 1000
    for i in range(0, len(embeddings), batch_size):
        batch_end = min(i + batch_size, len(embeddings))
        index.add(embeddings[i:batch_end])
        metadata_store.extend(metadatas[i:batch_end])
        
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

class VectorStore:
    _instance = None
    _initialized = False
    processing_status = {}
    BATCH_SIZE = 32  # Process chunks in batches
    MAX_CHUNKS = 500  # Limit total chunks per document
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info("Initializing VectorStore...")
            # Use a smaller, more efficient model
            self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            self.documents = {}
            # Create data directory if it doesn't exist
            FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.info("VectorStore initialized successfully")

    def _process_chunks_in_batches(self, chunks: List[str]) -> np.ndarray:
        """Process chunks in batches to manage memory."""
        all_embeddings = []
        
        # Limit chunks to prevent memory issues
        chunks = chunks[:self.MAX_CHUNKS]
        
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i:i + self.BATCH_SIZE]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)
            
            # Force garbage collection after each batch
            gc.collect()
        
        return np.vstack(all_embeddings)

    async def store_document(self, doc_content: Dict, filename: str) -> str:
        try:
            doc_id = str(uuid.uuid4())
            self.processing_status[doc_id] = "processing"
            logger.info(f"Processing document: {filename} (ID: {doc_id})")

            # Validate document content
            if not doc_content or "chunks" not in doc_content:
                raise ValueError("Invalid document content structure")

            chunks = doc_content["chunks"]
            if not chunks:
                raise ValueError("No content chunks found in document")

            # Process in batches
            embeddings = self._process_chunks_in_batches(chunks)
            
            # Prepare metadata (only for processed chunks)
            metadatas = [{
                "doc_id": doc_id,
                "filename": filename,
                "text": chunk,
                "chunk_index": i
            } for i, chunk in enumerate(chunks[:self.MAX_CHUNKS])]
            
            # Save to FAISS with memory optimization
            try:
                save_embedded_document_in_faiss(embeddings, metadatas)
            finally:
                # Clear memory
                del embeddings
                gc.collect()
            
            # Store document reference
            self.documents[doc_id] = {
                "filename": filename,
                "num_chunks": len(metadatas)
            }
            
            self.processing_status[doc_id] = "completed"
            logger.info(f"Document {filename} processed successfully")
            return doc_id
            
        except Exception as e:
            self.processing_status[doc_id] = "failed"
            logger.error(f"Failed to process document {filename}: {str(e)}")
            raise

    def query_faiss(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Query the FAISS index for similar chunks."""
        try:
            # Ensure proper shape
            if query_vector.ndim == 1:
                query_vector = query_vector[np.newaxis, :]
            query_vector = query_vector.astype('float32')
            
            # Normalize query vector
            faiss.normalize_L2(query_vector)
            
            # Load index and query
            index, metadata_store = _load_faiss_index_and_metadata()
            if index is None:
                raise FileNotFoundError("No processed documents found")
                
            distances, indices = index.search(query_vector, top_k)
            
            results = []
            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(metadata_store):
                    results.append({
                        "rank": rank + 1,
                        "distance": float(distance),
                        **metadata_store[idx]
                    })
            
            return results
        except Exception as e:
            logger.error(f"FAISS query error: {str(e)}")
            raise
