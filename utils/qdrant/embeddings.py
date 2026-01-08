import logging
import os
import shutil
import time
from typing import Tuple, Generator, Any

from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from utils.qdrant.config import (
    FASTEMBED_CACHE,
    DENSE_MODEL_NAME,
    SPARSE_MODEL_NAME,
    LATE_INTERACTION_MODEL_NAME
)

logger = logging.getLogger(__name__)

class EmbeddingService:
    _dense_model = None
    _sparse_model = None
    _late_interaction_model = None
    _lock = logging.getLogger(__name__) # Just using a logger as placeholder, actually we need threading lock if concurrent load is expected, but usually it's once.
    # Actually let's use a proper lock
    import threading
    _lock = threading.Lock()

    @classmethod
    def get_models(cls) -> Tuple[Any, Any, Any]:
        """
        Lazy load embedding models (singleton pattern) with auto-recovery from corruption.
        Returns: (dense_model, sparse_model, late_interaction_model)
        """
        if cls._dense_model is None:
            with cls._lock:
                if cls._dense_model is None:
                    cls._load_models_with_retry()

        return cls._dense_model, cls._sparse_model, cls._late_interaction_model

    @classmethod
    def _load_models(cls):
        """Internal method to load models."""
        return (
            TextEmbedding(DENSE_MODEL_NAME, cache_dir=FASTEMBED_CACHE, providers=['CPUExecutionProvider']),
            SparseTextEmbedding(SPARSE_MODEL_NAME, cache_dir=FASTEMBED_CACHE),
            LateInteractionTextEmbedding(LATE_INTERACTION_MODEL_NAME, cache_dir=FASTEMBED_CACHE)
        )

    @classmethod
    def _clear_cache(cls):
        """Clear all model caches in case of corruption."""
        model_patterns = [
            f"models--{DENSE_MODEL_NAME.replace('/', '--')}",
            f"models--{DENSE_MODEL_NAME.split('/')[-1]}-onnx", # Handle onnx variation naming if needed, though fastembed usually uses standardized paths
            f"models--{SPARSE_MODEL_NAME.replace('/', '--')}",
            f"models--{LATE_INTERACTION_MODEL_NAME.replace('/', '--')}",
        ]

        # Also add some hardcoded known variations seen in logs
        model_patterns.extend([
            "models--sentence-transformers--all-MiniLM-L6-v2",
            "models--qdrant--all-MiniLM-L6-v2-onnx",
            "models--Qdrant--bm25",
            "models--colbert-ir--colbertv2.0",
        ])

        for pattern in model_patterns:
            model_path = os.path.join(FASTEMBED_CACHE, pattern)
            if os.path.exists(model_path):
                logger.warning(f"[EmbeddingService] Deleting potentially corrupted cache: {model_path}")
                try:
                    shutil.rmtree(model_path)
                except Exception as rm_error:
                    logger.error(f"[EmbeddingService] Could not delete {model_path}: {rm_error}")

    @classmethod
    def _load_models_with_retry(cls):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[EmbeddingService] Loading models (attempt {attempt}/{max_retries})...")
                cls._dense_model, cls._sparse_model, cls._late_interaction_model = cls._load_models()
                logger.info("[EmbeddingService] ✅ Embedding models loaded successfully")
                return

            except Exception as e:
                error_msg = str(e)
                is_corruption = any(keyword in error_msg.lower() for keyword in [
                    'modelproto does not have a graph',
                    'onnxruntimeerror',
                    'corrupted',
                    'download',
                    'incomplete',
                    'could not download'
                ])

                if is_corruption:
                    logger.warning(f"[EmbeddingService] ⚠️ Model corruption/download error detected: {e}")
                    if attempt < max_retries:
                        logger.info(f"[EmbeddingService] 🧹 Clearing model caches and retrying...")
                        cls._clear_cache()
                        wait_time = 2 ** attempt
                        logger.info(f"[EmbeddingService] ⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"[EmbeddingService] ❌ Failed to load models after {max_retries} attempts")
                        raise RuntimeError(f"Failed to load embedding models: {e}") from e
                else:
                    logger.error(f"[EmbeddingService] ❌ Failed to load embedding models: {e}")
                    raise

    @classmethod
    def embed_query(cls, query: str, use_late_interaction: bool = True):
        """
        Embeds a single query string.
        Returns dictionary with 'dense', 'sparse', and optionally 'late' vectors.
        """
        dense_model, sparse_model, late_model = cls.get_models()

        dense_vec = next(dense_model.query_embed(query))
        sparse_vec = next(sparse_model.query_embed(query))
        late_vec = None
        if use_late_interaction:
            late_vec = next(late_model.query_embed(query))

        return {
            'dense': dense_vec,
            'sparse': sparse_vec,
            'late': late_vec
        }

    @classmethod
    def embed_documents(cls, documents: list[str]):
        """
        Embeds a list of documents.
        Returns generators for vectors.
        """
        dense_model, sparse_model, late_model = cls.get_models()

        return {
            'dense': dense_model.embed(documents),
            'sparse': sparse_model.embed(documents),
            'late': late_model.embed(documents)
        }
