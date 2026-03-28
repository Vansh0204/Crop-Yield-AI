import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KNOWLEDGE_CHUNKS = [
    "Cassava: Grows best in tropical climates with 1000-2000mm rainfall.",
    "Maize: Needs consistent moisture during silking. Optimal range: 18-27°C.",
    "Wheat: Requires cool growing season. Sensitive to extreme heat.",
    "Rice (Paddy): Continuous flooding or saturated soil required.",
    "Soybeans: High demand for Phosphorus and Potassium. pH: 6.0-6.8.",
    "Potatoes: Prefers slightly acidic soil. Steady moisture is critical.",
    "Sorghum: Highly drought tolerant. Withstands temperatures up to 40°C.",
    "Irrigation: Drip irrigation reduces water consumption by 40%.",
    "Soil Health: Cover cropping and zero-tillage increase water-holding capacity."
]

class AgronomicRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunks = KNOWLEDGE_CHUNKS
        self._build_index()

    def _build_index(self):
        embeddings = self.model.encode(self.chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        query_vec = self.model.encode([query])
        _, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]

_engine = None
def retrieve_context(query: str, k: int = 3):
    global _engine
    if _engine is None: _engine = AgronomicRAG()
    return _engine.retrieve(query, k)
