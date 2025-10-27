"""
Clean and refactored chat API endpoints for thread and message management
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List

from database.db import get_db
from utils.auth import get_current_user
from database.models import Users
from services.chat_service import ChatService
from schemas.chat_schemas import (
    ThreadSchema,
    ThreadWithMessagesSchema,
    ThreadMessagesResponse,
    CreateThreadRequest,
    RenameThreadRequest,
    SendMessageRequest,
)

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/threads", tags=["chat"])


def get_current_user_id(current_user: Users = Depends(get_current_user)) -> int:
    """Extract user ID from authenticated user"""
    return current_user.id


def get_chat_service(db: Session = Depends(get_db)) -> ChatService:
    """Dependency to get chat service instance"""
    return ChatService(db)


@router.get("/", response_model=List[ThreadSchema])
async def get_threads(
    chat_service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
):
    """
    Get all chat threads for the current user
    
    Returns threads ordered by most recently updated first.
    Requires authentication via JWT token.
    """
    try:
        logger.info(f"Getting threads for user {user_id}")
        threads = chat_service.get_user_threads(user_id)
        logger.info(f"Retrieved {len(threads)} threads for user {user_id}")
        return threads
    
    except Exception as e:
        logger.error(f"Error getting threads for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve threads"
        )


@router.post("/", response_model=ThreadSchema, status_code=status.HTTP_201_CREATED)
async def create_thread(
    request: CreateThreadRequest,
    chat_service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
):
    """
    Create a new chat thread with welcome message
    
    Creates a new thread and automatically adds a welcome message.
    Requires authentication via JWT token.
    """
    try:
        logger.info(f"Creating thread '{request.name}' for user {user_id}")
        thread = chat_service.create_thread(user_id, request.name)
        logger.info(f"Created thread {thread.id} for user {user_id}")
        return thread
    
    except Exception as e:
        logger.error(f"Error creating thread for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create thread"
        )


@router.get("/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(
    thread_id: str,
    page: int = 1,
    limit: int = None,
    chat_service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
):
    """
    Get messages for a specific thread with pagination
    
    Args:
        thread_id: The thread identifier
        page: Page number for pagination (default: 1)
        limit: Number of messages per page (default: 50, max: 200)
    
    Returns messages in chronological order with pagination info.
    Only accessible by the thread owner.
    Requires authentication via JWT token.
    """
    try:
        logger.info(f"Getting messages for thread {thread_id}, page {page}, limit {limit}")
        result = chat_service.get_thread_messages_paginated(thread_id, user_id, page, limit)
        logger.info(f"Retrieved {len(result.messages)} messages for thread {thread_id}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages for thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve thread messages"
        )


@router.get("/{thread_id}", response_model=ThreadWithMessagesSchema)
async def get_thread(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
):
    """
    Get a specific chat thread with all messages
    
    Returns the thread details along with all messages in chronological order.
    Use this endpoint when you need all messages without pagination.
    Only accessible by the thread owner.
    Requires authentication via JWT token.
    """
    try:
        logger.info(f"Getting thread {thread_id} with all messages")
        result = chat_service.get_thread_with_messages(thread_id, user_id)
        logger.info(f"Retrieved thread {thread_id} with {result.total_messages} messages")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve thread"
        )


@router.put("/{thread_id}/rename", response_model=ThreadSchema)
async def rename_thread(
    thread_id: str,
    request: RenameThreadRequest,
    chat_service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
):
    """
    Rename a chat thread
    
    Updates the thread name and the updated_at timestamp.
    Only accessible by the thread owner.
    Requires authentication via JWT token.
    """
    try:
        logger.info(f"Renaming thread {thread_id} to '{request.name}'")
        result = chat_service.rename_thread(thread_id, user_id, request.name)
        logger.info(f"Renamed thread {thread_id}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rename thread"
        )


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
):
    """
    Delete a chat thread
    
    Permanently deletes the thread and all associated messages.
    Only accessible by the thread owner.
    Requires authentication via JWT token.
    """
    try:
        logger.info(f"Deleting thread {thread_id}")
        chat_service.delete_thread(thread_id, user_id)
        logger.info(f"Deleted thread {thread_id}")
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete thread"
        )
