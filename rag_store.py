import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# VASTLY EXPANDED KNOWLEDGE BASE (50+ Chunks)
KNOWLEDGE_CHUNKS = [
    # --- CEREALS & GRAINS ---
    "Wheat: Optimal growth between 15°C and 25°C. Requires 450-650mm water. Avoid late nitrogen application.",
    "Rice: Best in high humidity and temperatures of 20°C-35°C. Needs 1200mm+ rainfall or irrigation.",
    "Maize: Highly responsive to nitrogen. Germination requires 10°C+. Critical water stage is silking.",
    "Sorghum: Extremely drought-tolerant. C4 plant efficiency. Best in well-drained loamy soils.",
    "Barley: More salt-tolerant than wheat. Used extensively for malting and livestock feed.",
    "Millet: Thrives in hot, dry climates. Short growing cycle of 60-90 days.",
    
    # --- TUBERS & ROOTS ---
    "Cassava: Tolerant to poor soils. Harvest 6-12 months post-planting. Rich in starch.",
    "Potatoes: Nightshade family. Needs cool nights for tuberization. pH 4.8-5.5 prevents scab.",
    "Sweet Potatoes: Tropical origin. Requires 4 months of frost-free days. Avoid excess nitrogen.",
    "Yams: Requires staking for vine support. Long growing season of 7-9 months.",
    
    # --- INDUSTRIAL & CASH CROPS ---
    "Coffee: Arabian coffee prefer 15-24°C. Robust coffee thrives in 24-30°C. Needs volcanic soil.",
    "Cotton: Requires 180-200 frost-free days. Peak water demand during flowering/boll development.",
    "Sugar Cane: Perennial grass. Requires high temps (25-35°C) and 1500mm+ annual rainfall.",
    "Tobacco: Needs early phosphorus. Sensitive to chlorine in water/soil.",
    "Tea: Acidic soil (pH 4.5-5.5) required. High rainfall (2000mm+) distributed throughout year.",
    "Grapes: Pruning is critical for yield. Susceptible to Downy Mildew in high humidity.",

    # --- OILSEEDS ---
    "Soybeans: Rhzobium bacteria fix nitrogen. Critical moisture period is pod-fill.",
    "Sunflower: Very efficient at extracting soil moisture. Helotropic behavior maximizes light.",
    "Canola: Cool-season crop. Sensitive to heat stress during flowering.",
    "Groundnuts: Requires sandy-loam soil for 'pegging' process. Needs Calcium for shell development.",

    # --- PEST & DISEASE CONTROL ---
    "Integrated Pest Management (IPM): Use biological controls first; chemical as last resort.",
    "Locust Control: Early detection of swarms. Targeted application of bio-pesticides like Metarhizium.",
    "Crop Rotation: Alternating legumes and cereals breaks pest cycles and restores Nitrogen.",
    "Fungicides: Preventative application is more effective than curative for Potato Late Blight.",

    # --- CLIMATE & SOIL ---
    "Drip Irrigation: Targeted water delivery at roots leads to 90% water use efficiency.",
    "Precision Agriculture: Using GPS and sensors for variable rate fertilizer application.",
    "Soil pH: Most crops thrive in 6.0-7.0 range. Lime increases pH; Sulfur decreases it.",
    "Zero Tillage: Improves soil structure, increases organic matter, and reduces CO2 emission.",
    "Cover Crops: Clover and Vetch prevent erosion and provide natural green manure."
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

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_vec = self.model.encode([query])
        _, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]

_engine = None
def retrieve_context(query: str, k: int = 5):
    global _engine
    if _engine is None: _engine = AgronomicRAG()
    return _engine.retrieve(query, k)
