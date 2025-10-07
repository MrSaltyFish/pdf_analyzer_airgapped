from pathlib import Path

# Misc flags
NORMALIZE_EMBEDDINGS = True
VERBOSE = False

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
VECTORDDB_DIR = PROJECT_ROOT / "vectorDB_data"

# Logging configs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "DEBUG" if VERBOSE else "INFO"
LOG_FILE = LOG_DIR / "run.log"

# Chunking configs
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", " ", ""]

# Model configs
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models/all-MiniLM-L12-v2"
LLM_MODEL_PATH = PROJECT_ROOT / "models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
EMBEDDING_DEVICE = "cpu"
LLM_THREADS = 8
LLM_CONTEXT = 2048
LLM_BATCH = 512

# Paths for FAISS index and metadata
FAISS_DIR = PROJECT_ROOT / "vectorDB_data/FAISS_cache"
FAISS_INDEX_PATH = FAISS_DIR / "vector_index.faiss"
FAISS_METADATA_PATH = FAISS_DIR / "vector_metadata.pkl"

TOP_K_RESULTS = 5

# TODO: modify below settings to support .env
CLEAR_OLD_FAISS = True
