"""
Chat API endpoints for thread and message management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

from database.db import get_db
from database.models import Users, ChatThread, ChatMessage

# Pydantic models for request/response
class MessageModel(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime
    apiRole: Optional[str] = None
    suggestions: Optional[List[str]] = None
    summary: Optional[str] = None
    needClarify: Optional[bool] = None
    inputType: Optional[str] = None

    class Config:
        orm_mode = True

class ThreadModel(BaseModel):
    id: str
    name: str
    createdAt: datetime
    updatedAt: datetime
    
    class Config:
        orm_mode = True

class ThreadWithMessagesModel(ThreadModel):
    messages: List[MessageModel]

class CreateThreadRequest(BaseModel):
    name: str = Field(..., min_length=1)

class RenameThreadRequest(BaseModel):
    name: str = Field(..., min_length=1)

class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    role: Optional[str] = None

# Create router
router = APIRouter(prefix="/api/threads")

# Use the authentication from utils.auth
from utils.auth import get_current_user

# Helper function to get user_id from current authenticated user using JWT
def get_current_user_id(current_user: Users = Depends(get_current_user)) -> int:
    """
    Get the current user's ID from JWT authentication
    """
    return current_user.id

@router.get("/", response_model=List[ThreadModel])
def get_threads(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Get all chat threads for the current user
    """
    threads = db.query(ChatThread).filter(ChatThread.user_id == user_id).order_by(
        ChatThread.updated_at.desc()
    ).all()
    
    return [
        ThreadModel(
            id=thread.id,
            name=thread.name,
            createdAt=thread.created_at,
            updatedAt=thread.updated_at
        ) for thread in threads
    ]

@router.post("/", response_model=ThreadModel, status_code=status.HTTP_201_CREATED)
def create_thread(
    request: CreateThreadRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Create a new chat thread
    """
    # Create thread with welcome message
    thread_id = str(uuid.uuid4())
    new_thread = ChatThread(
        id=thread_id,
        user_id=user_id,
        name=request.name,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    
    # Add welcome message
    welcome_message = ChatMessage(
        id=str(uuid.uuid4()),
        thread_id=thread_id,
        role="bot",
        content="Xin chào! Tôi là trợ lý AI của bạn. Rất vui được hỗ trợ bạn - Bạn cần tôi giúp gì hôm nay?",
        timestamp=datetime.now(),
    )
    
    db.add(new_thread)
    db.add(welcome_message)
    db.commit()
    
    return ThreadModel(
        id=new_thread.id,
        name=new_thread.name,
        createdAt=new_thread.created_at,
        updatedAt=new_thread.updated_at
    )

@router.get("/{thread_id}", response_model=ThreadWithMessagesModel)
def get_thread(
    thread_id: str,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Get a specific chat thread with all messages
    """
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id,
        ChatThread.user_id == user_id
    ).first()
    
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.thread_id == thread_id
    ).order_by(
        ChatMessage.timestamp.asc()
    ).all()
    
    return ThreadWithMessagesModel(
        id=thread.id,
        name=thread.name,
        createdAt=thread.created_at,
        updatedAt=thread.updated_at,
        messages=[
            MessageModel(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                apiRole=msg.api_role,
                suggestions=msg.suggestions,
                summary=msg.summary,
                needClarify=msg.need_clarify,
                inputType=msg.input_type
            ) for msg in messages
        ]
    )

@router.put("/{thread_id}/rename", response_model=ThreadModel)
def rename_thread(
    thread_id: str,
    request: RenameThreadRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Rename a chat thread
    """
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id,
        ChatThread.user_id == user_id
    ).first()
    
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    
    thread.name = request.name
    thread.updated_at = datetime.now()
    db.commit()
    
    return ThreadModel(
        id=thread.id,
        name=thread.name,
        createdAt=thread.created_at,
        updatedAt=thread.updated_at
    )

@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: str,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Delete a chat thread
    """
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id,
        ChatThread.user_id == user_id
    ).first()
    
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    
    db.delete(thread)
    db.commit()
    
    return None
