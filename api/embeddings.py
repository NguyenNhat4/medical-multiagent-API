"""
Embeddings loading API endpoints for managing Qdrant collections
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from utils.timezone_utils import get_vietnam_time
import os
import socket
from qdrant_client import QdrantClient
from utils.knowledge_base.loadvector_qdrant import (
    EmbeddingModels,
    load_all_collections,
    COLLECTION_CONFIGS,
    QDRANT_URL
)


# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


# Pydantic models
class CollectionLoadRequest(BaseModel):
    collections: Optional[List[str]] = Field(
        None,
        description="List of specific collections to load. If None, all collections will be loaded.",
        example=["bndtd", "bsnt" , "bsrhm","bnrhm"]
    )
    recreate: bool = Field(
        False,
        description="Whether to recreate collections if they already exist"
    )
    qdrant_url: Optional[str] = Field(
       QDRANT_URL,
        description="Optional Qdrant server URL override"
    )


class CollectionStatus(BaseModel):
    name: str = Field(..., description="Collection name")
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    points_count: Optional[int] = Field(None, description="Number of points in the collection")


class LoadResponse(BaseModel):
    status: str = Field(..., description="Overall operation status")
    timestamp: str = Field(..., description="Operation timestamp")
    collections: List[CollectionStatus] = Field(..., description="Status of each collection")
    total_processed: int = Field(..., description="Total number of collections processed")
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")


class AvailableCollectionsResponse(BaseModel):
    collections: List[str] = Field(..., description="List of available collection names")
    timestamp: str = Field(..., description="Response timestamp")


@router.get("/collections", response_model=AvailableCollectionsResponse)
async def get_available_collections():
    """
    Get list of available collections that can be loaded into Qdrant

    Returns the collection names defined in the system configuration.
    """
    try:
        from utils.knowledge_base.loadvector_qdrant import COLLECTION_CONFIGS

        return AvailableCollectionsResponse(
            collections=list(COLLECTION_CONFIGS.keys()),
            timestamp=get_vietnam_time().isoformat()
        )
    except Exception as e:
        logger.error(f"❌ Error getting available collections: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available collections: {str(e)}"
        )


@router.post("/load", response_model=LoadResponse)
async def load_embeddings(
    request: CollectionLoadRequest,
    background_tasks: BackgroundTasks
):
    """
    Load CSV files into Qdrant collections with hybrid search embeddings

    This endpoint loads medical knowledge base CSV files into separate Qdrant collections,
    creating embeddings using dense, sparse (BM25), and late interaction (ColBERT) models.

    - **collections**: Optional list of specific collections to load (default: all)
    - **recreate**: Whether to recreate existing collections (default: false)
    - **qdrant_url**: Optional Qdrant server URL override

    The operation will skip collections that already have data unless recreate=true.
    """
    try:
       

        logger.info(f"📥 Loading embeddings request received")
        logger.info(f"   Collections: {request.collections or 'all'}")
        logger.info(f"   Recreate: {request.recreate}")

        # Validate requested collections
        if request.collections:
            invalid_collections = [
                col for col in request.collections
                if col not in COLLECTION_CONFIGS
            ]
            if invalid_collections:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid collection names: {invalid_collections}. "
                           f"Available collections: {list(COLLECTION_CONFIGS.keys())}"
                )
        
        # Determine Qdrant URL
        qdrant_url = request.qdrant_url if request.qdrant_url and request.qdrant_url != "string" else QDRANT_URL
        if not qdrant_url:
            raise HTTPException(
                status_code=400,
                detail="Qdrant URL not provided and QDRANT_URL environment variable not set"
            )

        # Initialize Qdrant client
        logger.info(f"🔗 Connecting to Qdrant at {qdrant_url}")
        try:
            client = QdrantClient(qdrant_url)
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to Qdrant at {qdrant_url}: {str(e)}"
            )

        # Load embedding models
        logger.info("  Loading embedding models...")
        models = EmbeddingModels()
        models.load()
        logger.info("✅ Embedding models loaded successfully")

        # Load collections
        logger.info("📚 Starting collection loading...")
        results = load_all_collections(
            client=client,
            models=models,
            collections=request.collections,
            recreate=request.recreate
        )

        # Prepare response
        collection_statuses = []
        successful_count = 0
        failed_count = 0

        for collection_name, success in results.items():
            # Get points count if successful
            points_count = None
            message = "Failed to load"

            if success:
                try:
                    collection_info = client.get_collection(collection_name)
                    points_count = collection_info.points_count

                    if request.recreate:
                        message = f"Successfully recreated with {points_count} points"
                    else:
                        message = f"Loaded or already exists with {points_count} points"
                    successful_count += 1
                except Exception as e:
                    message = f"Loaded but failed to get stats: {str(e)}"
                    successful_count += 1
            else:
                failed_count += 1
                message = "Failed to load collection"

            collection_statuses.append(
                CollectionStatus(
                    name=collection_name,
                    success=success,
                    message=message,
                    points_count=points_count
                )
            )

        overall_status = "success" if failed_count == 0 else "partial" if successful_count > 0 else "failed"

        logger.info(f"✅ Embedding loading completed")
        logger.info(f"   Status: {overall_status}")
        logger.info(f"   Successful: {successful_count}")
        logger.info(f"   Failed: {failed_count}")

        return LoadResponse(
            status=overall_status,
            timestamp=get_vietnam_time().isoformat(),
            collections=collection_statuses,
            total_processed=len(results),
            successful=successful_count,
            failed=failed_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error loading embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading embeddings: {str(e)}"
        )


@router.get("/status/{collection_name}")
async def get_collection_status(collection_name: str):
    """
    Get status information for a specific collection

    Returns information about whether the collection exists and how many points it contains.
    """
    try:
        import os
        from qdrant_client import QdrantClient
        from utils.knowledge_base.loadvector_qdrant import (
            COLLECTION_CONFIGS,
            collection_has_data,
            QDRANT_URL
        )

        # Validate collection name
        if collection_name not in COLLECTION_CONFIGS:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found. "
                       f"Available collections: {list(COLLECTION_CONFIGS.keys())}"
            )

        # Connect to Qdrant
        qdrant_url = QDRANT_URL
        if not qdrant_url:
            raise HTTPException(
                status_code=500,
                detail="QDRANT_URL environment variable not set"
            )

        client = QdrantClient(qdrant_url)

        # Check collection status
        has_data, points_count = collection_has_data(client, collection_name)

        return {
            "collection_name": collection_name,
            "exists": has_data,
            "points_count": points_count,
            "status": "loaded" if has_data else "not_loaded",
            "timestamp": get_vietnam_time().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting collection status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection status: {str(e)}"
        )


@router.get("/dns-test")
async def dns_test(hostname: str = "qdrant", port: int = 6333):
    """
    Test DNS resolution and basic TCP connectivity from inside the API container.

    - hostname: the host to resolve (default: "qdrant")
    - port: TCP port to try connecting (default: 6333)
    """
    resolved_ip = None
    resolved = False
    connect_ok = False
    connect_error = None

    try:
        resolved_ip = socket.gethostbyname(hostname)
        resolved = True
    except Exception as e:
        connect_error = f"DNS error: {e}"

    if resolved:
        try:
            with socket.create_connection((resolved_ip, port), timeout=2):
                connect_ok = True
        except Exception as e:
            connect_error = f"TCP connect error: {e}"

    return {
        "hostname": hostname,
        "port": port,
        "resolved": resolved,
        "ip": resolved_ip,
        "connect_ok": connect_ok,
        "error": connect_error,
        "env_QDRANT_URL": os.getenv("QDRANT_URL"),
        "timestamp": get_vietnam_time().isoformat(),
    }
