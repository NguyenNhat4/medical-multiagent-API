"""
FastAPI server for Medical Conversation System
Exposes the PocketFlow medical agent as REST API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, status
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from passlib.hash import bcrypt
from sqlalchemy.orm import Session
from database.db import get_db, Users, ChatMessage, ChatThreads
from database.models import ChatThread
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uvicorn
import os
import uuid
from dotenv import load_dotenv

# Import chat routes
from chat_routes import router as chat_router

# Import our flow and conversation logger
from flow import create_med_agent_flow
from utils.conversation_logger import conversation_logger
from utils.response_parser import (
    parse_medical_response,
    handle_greeting_response,
    handle_statement_response,
)
from utils.helpers import serialize_conversation_history
from utils.role_ENUM import RoleEnum, ROLE_DISPLAY_NAME, ROLE_DESCRIPTION, get_role_name

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Conversation API",
    description="AI-powered medical consultation system using PocketFlow",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/redoc",
    swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
        "clientId": "",
        "clientSecret": "",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router with prefix
router = APIRouter(prefix="/api")

# Initialize the medical flow
med_flow = create_med_agent_flow()


# Pydantic models for request/response
class ConversationRequest(BaseModel):
    message: str = Field(
        ..., 
        description="User's message"
    )
    role: str = Field(
        default=RoleEnum.PATIENT_DENTAL.value,
        description="User's role in the medical context (can be role name or role ID)",
    )
    session_id: Optional[str] = Field(
        None, description="Session identifier for conversation tracking"
    )


class ConversationResponse(BaseModel):
    explanation: str = Field(..., description="Main explanation/answer content")
    questionSuggestion: Optional[List[str]] = Field(None, description="Follow-up question suggestions")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    input_type: Optional[str] = Field(None, description="Classified input type")
    need_clarify: Optional[bool] = Field(None, description="Whether response needs clarification")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str


class RoleInfo(BaseModel):
    id: str = Field(..., description="Role identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Role description")


class RolesResponse(BaseModel):
    roles: List[RoleInfo] = Field(..., description="Available roles")
    default_role: str = Field(..., description="Default role identifier")
    timestamp: str = Field(..., description="Response timestamp")


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=200)


class LoginReq(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


class DeleteUserResponse(BaseModel):
    message: str
    deleted_user: UserOut
    timestamp: str


class ConversationThreadInfo(BaseModel):
    thread_id: str = Field(..., description="Thread/session identifier")
    name: str = Field(..., description="Thread name")
    preview: str = Field(..., description="Preview of first message (truncated)")
    message_count: int = Field(..., description="Total number of messages in thread")
    created_at: str = Field(..., description="Thread creation timestamp")
    updated_at: str = Field(..., description="Thread last update timestamp")
    last_activity: str = Field(..., description="Last message timestamp")


class ConversationHistoryResponse(BaseModel):
    conversations: List[ConversationThreadInfo] = Field(..., description="List of conversation threads")
    total: int = Field(..., description="Total number of conversations")
    user_id: int = Field(..., description="User ID")


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


# API Endpoints


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Conversation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", timestamp=datetime.now().isoformat(), version="1.0.0"
    )


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    # duplicate check
    if db.query(Users).filter(Users.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already exists")

    hashed = bcrypt.hash(payload.password)
    user = Users(email=payload.email, password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, email=user.email)


from utils.auth import create_access_token, Token, get_current_user
from fastapi.security import OAuth2PasswordRequestForm

@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginReq, db: Session = Depends(get_db)):
    user = db.query(Users).filter(Users.email == body.email).first()
    if not user or not bcrypt.verify(body.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create access token
    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserOut(id=user.id, email=user.email)
    )

@router.post("/auth/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login, get an access token for future requests
    This endpoint is used by Swagger UI for authorization
    """
    user = db.query(Users).filter(Users.email == form_data.username).first()
    if not user or not bcrypt.verify(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data)
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users", response_model=List[UserOut])
def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all users with pagination

    - **skip**: Number of users to skip (default: 0)
    - **limit**: Maximum number of users to return (default: 100, max: 1000)
    """
    if limit > 1000:
        limit = 1000

    users = db.query(Users).offset(skip).limit(limit).all()
    return [UserOut(id=user.id, email=user.email) for user in users]


@router.get("/users/me", response_model=UserOut)
def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Get current user information
    
    Returns the profile information of the currently authenticated user.
    Requires valid JWT token in Authorization header.
    """
    return UserOut(id=current_user.id, email=current_user.email)


@router.delete("/users/{user_id}", response_model=DeleteUserResponse)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete user by ID

    - **user_id**: The ID of the user to delete
    """
    user = db.query(Users).filter(Users.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Store user info before deletion
    deleted_user_info = UserOut(id=user.id, email=user.email)

    db.delete(user)
    db.commit()

    return DeleteUserResponse(
        message=f"User {user_id} deleted successfully",
        deleted_user=deleted_user_info,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/roles", response_model=RolesResponse)
async def get_available_roles():
    """
    Get available user roles for medical conversations

    Returns list of available roles with descriptions to help users choose
    the appropriate role for their medical consultation context.
    """
    try:
        roles = [
            RoleInfo(
                id=r.value,
                name=ROLE_DISPLAY_NAME[r],
                description=ROLE_DESCRIPTION[r],
            )
            for r in RoleEnum
        ]

        return RolesResponse(
            roles=roles,
            default_role=RoleEnum.PATIENT_DENTAL.value,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"❌ Error getting roles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving roles: {str(e)}")


@router.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Main chat endpoint for medical conversations

    - **message**: User's input message
    - **role**: User's role in medical context
    - **session_id**: Optional session identifier (thread_id from database)
    
    Requires authentication via JWT token
    """
    try:
        # Get user_id from the authenticated user
        user_id = current_user.id
        
        # Make sure we have a valid thread_id (session_id)
        thread_id = request.session_id
        if not thread_id:
            raise HTTPException(
                status_code=400, 
                detail="session_id (thread_id) is required"
            )
            
        # Verify that the thread belongs to the current user
        thread = db.query(ChatThread).filter(
            ChatThread.id == thread_id,
            ChatThread.user_id == user_id
        ).first()
        
        if not thread:
            raise HTTPException(
                status_code=404,
                detail="Thread not found or you don't have permission to access it"
            )
            
        # Convert role ID to role name if needed
        role_name = get_role_name(request.role)
        logger.info(
            f"🔥 New chat request - Role: {role_name} (from: {request.role}), Message: {request.message[:50]}..."
        )

        # Store user message in database
        user_message_id = str(uuid.uuid4())
        user_message = ChatMessage(
            id=user_message_id,
            thread_id=thread_id,
            role="user",
            content=request.message.strip(),
            timestamp=datetime.now(),
            api_role=request.role
        )
        db.add(user_message)
        db.commit()

        # Serialize conversation history for the flow
        conversation_history = serialize_conversation_history(thread.messages)

        # Prepare shared data for the flow
        shared = {
            "role": role_name,
            "input": request.message.strip(),
            "query": "",
            "explain": "",
            "conversation_history": conversation_history,
            "session_id": request.session_id,
        }
        # run chat flow  -> updating shared store
        med_flow.run(shared)
        
        explanation = shared.get("explain")
        if not explanation or not isinstance(explanation, str) or not explanation.strip():
            explanation = "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại hoặc đặt câu hỏi cụ thể hơn."
        suggestion_questions = shared.get("suggestion_questions", [])
        input_type = shared.get("input_type")
        need_clarify = shared.get("need_clarify", False)
        logger.info(f"✅ Flow completed - explanation: {explanation}, Need clarify: {need_clarify}")

        response = ConversationResponse(
            explanation=explanation,
            questionSuggestion=suggestion_questions,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat(),
            input_type=input_type,
            need_clarify=need_clarify
        )
        
        # Store bot message in database
        bot_message = ChatMessage(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role="bot",
            content=explanation,
            timestamp=datetime.now(),
            suggestions=suggestion_questions,
            summary=explanation,
            need_clarify=need_clarify,
            input_type=input_type
        )
        db.add(bot_message)
        
        # Update thread's updated_at timestamp
        thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if thread:
            thread.updated_at = datetime.now()
        
        db.commit()

        # Log conversation in background (async)
        background_tasks.add_task(
            log_conversation_background,
            request.message,
            explanation,
            request.session_id,
        )

        return response

    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/conversation-history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    limit: int = 50, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get user's conversation threads for chat history sidebar
    
    - **limit**: Number of recent threads to return (default: 50)
    
    Returns a list of conversation threads with summary information for displaying
    in a chat history sidebar. Each thread includes the first user message as preview.
    Requires authentication via JWT token.
    """
    try:
        # Get user's threads ordered by most recent activity
        threads = db.query(ChatThread).filter(
            ChatThread.user_id == current_user.id
        ).order_by(ChatThread.updated_at.desc()).limit(limit).all()
        
        if not threads:
            return {
                "conversations": [],
                "total": 0,
                "user_id": current_user.id
            }
        
        conversations = []
        for thread in threads:
            # Get the first user message as preview
            first_message = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread.id,
                ChatMessage.role == "user"
            ).order_by(ChatMessage.timestamp.asc()).first()
            
            # Get total message count for this thread
            message_count = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread.id
            ).count()
            
            # Get the last message timestamp
            last_message = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread.id
            ).order_by(ChatMessage.timestamp.desc()).first()
            
            conversation_info = {
                "thread_id": thread.id,
                "name": thread.name,
                "preview": first_message.content[:100] + "..." if first_message and len(first_message.content) > 100 else (first_message.content if first_message else "New conversation"),
                "message_count": message_count,
                "created_at": thread.created_at.isoformat(),
                "updated_at": thread.updated_at.isoformat(),
                "last_activity": last_message.timestamp.isoformat() if last_message else thread.updated_at.isoformat()
            }
            conversations.append(conversation_info)

        return ConversationHistoryResponse(
            conversations=conversations,
            total=len(conversations),
            user_id=current_user.id
        )

    except Exception as e:
        logger.error(f"❌ Error reading conversation history: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error reading conversation history: {str(e)}"
        )


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


@router.post("/clear-history")
async def clear_conversation_history():
    """Clear conversation history log file"""
    try:
        if os.path.exists("conversation.log"):
            # Backup current log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"conversation_backup_{timestamp}.log"
            os.rename("conversation.log", backup_name)
            logger.info(f"📦 Conversation history backed up to {backup_name}")

        # Create new log
        conversation_logger._ensure_log_file()

        return {
            "message": "Conversation history cleared successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error clearing conversation history: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing conversation history: {str(e)}"
        )


# Background tasks
async def log_conversation_background(
    user_message: str, bot_response: str, session_id: Optional[str]
):
    """Background task to log conversation"""
    try:
        # Add session info if available
        if session_id:
            conversation_logger._write_to_log(f"[Session: {session_id}]")

        conversation_logger.log_exchange(user_message, bot_response)
        logger.info("📝 Conversation logged successfully")

    except Exception as e:
        logger.error(f"❌ Error logging conversation: {str(e)}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
        },
    )


# Include router after defining routes
app.include_router(router)

# Include chat threads router
app.include_router(chat_router)

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"🚀 Starting Medical Conversation API on {host}:{port}")
    logger.info(f"📖 API Documentation: http://{host}:{port}/docs")

    uvicorn.run("api:app", host=host, port=port, reload=debug, log_level="info")
