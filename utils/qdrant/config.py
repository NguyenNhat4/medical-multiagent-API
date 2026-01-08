import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv(override=False)

# Qdrant Connection
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Embedding Models
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE_PATH", "./models")

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
LATE_INTERACTION_MODEL_NAME = "colbert-ir/colbertv2.0"

DENSE_VECTOR_SIZE = 384
LATE_INTERACTION_VECTOR_SIZE = 128

# Vector Names
VECTOR_DENSE = "all-MiniLM-L6-v2"
VECTOR_SPARSE = "bm25"
VECTOR_LATE = "colbertv2.0"

# Collection Names
COLLECTION_BNDTD = "bndtd"
COLLECTION_BSNT = "bsnt"
COLLECTION_BNRHM = "bnrhm"
COLLECTION_BSRHM = "bsrhm"
COLLECTION_USER_MEMORY = "user_memory"

ALL_COLLECTIONS = [
    COLLECTION_BNDTD,
    COLLECTION_BSNT,
    COLLECTION_BNRHM,
    COLLECTION_BSRHM,
    COLLECTION_USER_MEMORY
]

# Collection Configuration
# Maps collection name to its CSV file source (for loading)
COLLECTION_SOURCES = {
    COLLECTION_BNDTD: "medical_knowledge_base/Bệnh nhân đái tháo đường.csv",
    COLLECTION_BSNT: "medical_knowledge_base/Bác sĩ nội tiết.csv",
    COLLECTION_BNRHM: "medical_knowledge_base/Bệnh nhân răng hàm mặt.csv",
    COLLECTION_BSRHM: "medical_knowledge_base/Bác sĩ răng hàm mặt.csv",
}

# Vector Configuration for Qdrant Collections
def get_vector_params(models_module):
    """
    Returns the vector parameters using Qdrant models.
    Pass 'qdrant_client.models' as the argument to avoid circular imports or hard dependency here.
    """
    return {
        VECTOR_DENSE: models_module.VectorParams(
            size=DENSE_VECTOR_SIZE,
            distance=models_module.Distance.COSINE,
        ),
        VECTOR_LATE: models_module.VectorParams(
            size=LATE_INTERACTION_VECTOR_SIZE,
            distance=models_module.Distance.COSINE,
            multivector_config=models_module.MultiVectorConfig(
                comparator=models_module.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models_module.HnswConfigDiff(m=0)  # Disable HNSW for reranking
        ),
    }

def get_sparse_vector_params(models_module):
    return {
        VECTOR_SPARSE: models_module.SparseVectorParams(
            modifier=models_module.Modifier.IDF
        )
    }
