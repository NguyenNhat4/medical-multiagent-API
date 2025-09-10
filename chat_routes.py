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
class MessageInfo(BaseModel):
    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user/bot)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    api_role: Optional[str] = Field(None, description="API role used for user messages")
    suggestions: Optional[List[str]] = Field(None, description="Bot message suggestions")
    need_clarify: Optional[bool] = Field(None, description="Whether response needs clarification")
    input_type: Optional[str] = Field(None, description="Classified input type")


class ThreadMessagesResponse(BaseModel):
    thread_id: str = Field(..., description="Thread identifier")
    thread_name: str = Field(..., description="Thread name")
    messages: List[MessageInfo] = Field(..., description="List of messages in chronological order")
    total_messages: int = Field(..., description="Total number of messages")
    user_id: int = Field(..., description="User ID")
    created_at: str = Field(..., description="Thread creation timestamp")
    updated_at: str = Field(..., description="Thread last update timestamp")


@router.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(
    thread_id: str,
    page: int = 1,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get all messages for a specific conversation thread
    
    - **thread_id**: The thread/session identifier
    - **page**: Page number for pagination (default: 1)
    - **limit**: Number of messages per page (default: 50, max: 200)
    
    Returns all messages in chronological order for the specified thread.
    Only accessible by the thread owner. Perfect for loading full conversation
    when user clicks on a thread from the sidebar.
    Requires authentication via JWT token.
    """
    try:
        # Validate limit
        if limit > 200:
            limit = 200
        if limit < 1:
            limit = 1
        if page < 1:
            page = 1
            
        # Verify thread exists and belongs to current user
        thread = db.query(ChatThread).filter(
            ChatThread.id == thread_id,
            ChatThread.user_id == current_user.id
        ).first()
        
        if not thread:
            raise HTTPException(
                status_code=404,
                detail="Thread not found or you don't have permission to access it"
            )
        
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Get total message count
        total_messages = db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id
        ).count()
        
        # Get messages for this thread with pagination
        messages = db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id
        ).order_by(ChatMessage.timestamp.asc()).offset(offset).limit(limit).all()
        
        # Format messages
        formatted_messages = []
        for message in messages:
            message_info = MessageInfo(
                id=message.id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp.isoformat(),
                api_role=message.api_role,
                suggestions=message.suggestions,
                need_clarify=message.need_clarify,
                input_type=message.input_type
            )
            formatted_messages.append(message_info)
        
        return ThreadMessagesResponse(
            thread_id=thread.id,
            thread_name=thread.name,
            messages=formatted_messages,
            total_messages=total_messages,
            user_id=current_user.id,
            created_at=thread.created_at.isoformat(),
            updated_at=thread.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting thread messages: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving thread messages: {str(e)}"
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
