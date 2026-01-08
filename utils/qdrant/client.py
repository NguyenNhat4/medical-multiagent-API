import logging
import threading
from typing import Optional
from qdrant_client import QdrantClient
from utils.qdrant.config import QDRANT_URL, QDRANT_API_KEY

logger = logging.getLogger(__name__)

class QdrantClientManager:
    _instance: Optional[QdrantClient] = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls) -> QdrantClient:
        """
        Returns a singleton instance of QdrantClient.
        Thread-safe instantiation.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        logger.info(f"🔗 Connecting to Qdrant at {QDRANT_URL}")
                        cls._instance = QdrantClient(
                            url=QDRANT_URL,
                            api_key=QDRANT_API_KEY,
                            # Connection reuse settings if applicable, though QdrantClient handles this mostly internally
                        )
                    except Exception as e:
                        logger.error(f"❌ Failed to connect to Qdrant: {e}")
                        raise
        return cls._instance

    @classmethod
    def reset_client(cls):
        """
        Resets the client instance, forcing a reconnection on next access.
        """
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.close()
                except Exception as e:
                    logger.warning(f"Error closing Qdrant client: {e}")
            cls._instance = None

def get_client() -> QdrantClient:
    """Convenience function to get the Qdrant client."""
    return QdrantClientManager.get_client()
