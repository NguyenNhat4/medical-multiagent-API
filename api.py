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
from utils.response_parser import (
    parse_medical_response,
    handle_greeting_response,
    handle_statement_response,
)
from utils.helpers import serialize_conversation_history
from utils.role_ENUM import RoleEnum, ROLE_DISPLAY_NAME, ROLE_DESCRIPTION

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

# Startup event to load knowledge base
@app.on_event("startup")
async def startup_event():
    """Load knowledge base and initialize components at startup"""
    logger.info("🚀 Starting Medical Conversation API...")
    logger.info("🔄 Loading knowledge base...")
    
    try:
        from utils.kb import get_kb
        kb = get_kb()  # This will trigger the loading
        logger.info(f"✅ Knowledge base loaded successfully!")
        logger.info(f"📊 Total records: {len(kb.df)}")
        logger.info(f"📁 Role-specific dataframes: {list(kb.role_dataframes.keys())}")
        logger.info(f"🔧 Vectorizers created: {list(kb.role_vectorizers.keys())}")
        
        # Test a quick retrieval to ensure everything works
        from utils.kb import retrieve
        test_results, test_score = retrieve("test", top_k=1)
        logger.info(f"🧪 Test retrieval successful: {len(test_results)} results, score: {test_score:.4f}")
        
        logger.info("🎉 API startup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge base: {str(e)}")
        logger.error("⚠️  API will continue but chat functionality may be limited")
        raise e

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
    explanation: str = Field(..., description="Main explanation/answer content ")
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

        # Lấy 3 message gần nhất của thread
        if thread:
            thread.messages = sorted(thread.messages, key=lambda m: m.timestamp, reverse=True)[:3][::-1]
        if not thread:
            raise HTTPException(
                status_code=404,
                detail="Thread not found or you don't have permission to access it"
            )
            
        role_name = request.role
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
