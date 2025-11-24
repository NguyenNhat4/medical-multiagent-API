"""
Retrieval API endpoint - Direct access to knowledge base search
"""

import logging
import os
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from utils.role_enum import RoleEnum, ROLE_TO_CSV
from utils.knowledge_base.qdrant_retrieval import retrieve_from_qdrant

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


# Pydantic models
class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Search query")
    role: str = Field(
        default=RoleEnum.PATIENT_DENTAL.value,
        description="User's role to select appropriate knowledge base"
    )
    top_k: int = Field(default=5, description="Number of results to return")
    demuc: Optional[str] = Field(None, description="Filter by category (DEMUC)")
    chu_de_con: Optional[str] = Field(None, description="Filter by sub-category (CHUDECON)")


class RetrievalResult(BaseModel):
    id: int | str
    score: float
    question: str = Field(..., alias="CAUHOI")
    answer: str = Field(..., alias="CAUTRALOI")
    category: str = Field(..., alias="DEMUC")
    subcategory: str = Field(..., alias="CHUDECON")
    explanation: str = Field(default="", alias="GIAITHICH")
    
    class Config:
        populate_by_name = True


class RetrievalResponse(BaseModel):
    results: List[RetrievalResult]
    total: int
    collection_used: str


@router.post("/search", response_model=RetrievalResponse)
async def search_knowledge_base(
    request: RetrievalRequest,
):
    """
    Search the knowledge base directly based on user query and role.
    
    - **query**: The question or search term
    - **role**: The user role (determines which collection to search)
    - **top_k**: Number of results (default: 5)
    """
    try:
        # Validate and normalize role
        role_name = request.role
        valid_roles = [role.value for role in RoleEnum]
        
        if role_name not in valid_roles:
            # Try to match by enum name if value fails
            try:
                role_enum = RoleEnum[role_name]
                role_name = role_enum.value
            except KeyError:
                logger.warning(f"⚠️ Invalid role '{role_name}', using default '{RoleEnum.PATIENT_DENTAL.value}'")
                role_name = RoleEnum.PATIENT_DENTAL.value

        # Determine collection name from role
        # ROLE_TO_CSV maps role to filename (e.g., "bndtd.csv")
        # We assume collection name is filename without extension (e.g., "bndtd")
        csv_filename = ROLE_TO_CSV.get(role_name, "bnrhm.csv")
        collection_name = csv_filename.replace(".csv", "")
        
        logger.info(f"🔍 Searching in collection '{collection_name}' for query: '{request.query}'")

        # Execute search
        raw_results = retrieve_from_qdrant(
            query=request.query,
            demuc=request.demuc,
            chu_de_con=request.chu_de_con,
            top_k=request.top_k,
            collection_name=collection_name
        )

        # Format results
        formatted_results = []
        for res in raw_results:
            formatted_results.append(RetrievalResult(
                id=res.get("id"),
                score=res.get("score"),
                CAUHOI=res.get("CAUHOI", ""),
                CAUTRALOI=res.get("CAUTRALOI", ""),
                DEMUC=res.get("DEMUC", ""),
                CHUDECON=res.get("CHUDECON", ""),
                GIAITHICH=res.get("GIAITHICH", "")
            ))

        return RetrievalResponse(
            results=formatted_results,
            total=len(formatted_results),
            collection_used=collection_name
        )

    except Exception as e:
        logger.error(f"❌ Error in retrieval endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

