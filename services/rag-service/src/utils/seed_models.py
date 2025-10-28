from sentence_transformers import SentenceTransformer
from src.core.config import EMBEDDING_MODEL_PATH

model_name = 'paraphrase-MiniLM-L3-v2'
model = SentenceTransformer(model_name)
MODEL_PATH = EMBEDDING_MODEL_PATH.parent / model_name
print(MODEL_PATH)
model.save(str(MODEL_PATH))
