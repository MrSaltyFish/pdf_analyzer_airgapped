from src.agents.embedding_agent import EmbeddingAgent

class ModelSession:
    def __init__(self, persona: str, task: str, pdf_vectors: list[dict]):
        self.persona = persona
        self.task = task
        self.user_query_vector = EmbeddingAgent.encode(persona, task)
        self.faiss_index = self._create_faiss_index(pdf_vectors)

    def _create_faiss_index(self, vectors):
        import faiss, numpy as np
        dim = len(vectors[0]['vector'])
        index = faiss.IndexFlatL2(dim)
        vecs = np.array([v['vector'] for v in vectors], dtype=np.float32)
        index.add(vecs)
        return index

    def query_top_k(self, k=5):
        import numpy as np
        query_vec = np.array(self.user_query_vector).reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vec, k)
        return [{"index": i, "distance": d} for i, d in zip(indices[0], distances[0])]
