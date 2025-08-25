"""
FastAPI server for Medical Conversation System
Exposes the PocketFlow medical agent as REST API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv

# Import our flow and conversation logger
from flow import create_med_agent_flow
from utils.conversation_logger import conversation_logger
from utils.response_parser import parse_medical_response, handle_greeting_response, handle_statement_response

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Conversation API",
    description="AI-powered medical consultation system using PocketFlow",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/redoc"
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

# Role mapping helper
def get_role_name(role_input: str) -> str:
    """
    Convert role ID to role name, or return the input if it's already a role name
    
    Args:
        role_input: Role ID or role name
        
    Returns:
        Standard role name for the medical flow
    """
    role_mapping = {
        "patient_dental": "Bệnh nhân nha khoa",
        "patient_diabetes": "Bệnh nhân đái tháo đường", 
        "doctor_dental": "Bác sĩ nha khoa",
        "doctor_endocrine": "Bác sĩ nội tiết"
    }
    
    # If it's a role ID, convert to role name
    if role_input in role_mapping:
        return role_mapping[role_input]
    
    # If it's already a role name, return as is
    valid_role_names = list(role_mapping.values())
    if role_input in valid_role_names:
        return role_input
    
    # Default fallback
    logger.warning(f"Unknown role: {role_input}, using default")
    return "Bệnh nhân nha khoa"

# Pydantic models for request/response
class ConversationRequest(BaseModel):
    message: str = Field(..., description="User's message", min_length=1, max_length=1000)
    role: str = Field(
        default="Bệnh nhân nha khoa",
        description="User's role in the medical context (can be role name or role ID)"
    )
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")

class ConversationResponse(BaseModel):
    explanation: str = Field(..., description="Main explanation/answer content")
    summary: str = Field(..., description="Brief summary of key points")
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

# API Endpoints

@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Conversation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
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
                id="patient_dental",
                name="Bệnh nhân nha khoa",
                description="Dành cho người cần tư vấn về các vấn đề răng miệng, nha chu, và chăm sóc sức khỏe răng miệng"
            ),
            RoleInfo(
                id="patient_diabetes",
                name="Bệnh nhân đái tháo đường",
                description="Dành cho người mắc đái tháo đường cần tư vấn về mối liên hệ giữa bệnh đái tháo đường và sức khỏe răng miệng"
            ),
            RoleInfo(
                id="doctor_dental",
                name="Bác sĩ nha khoa",
                description="Dành cho bác sĩ nha khoa cần tư vấn về tác động của đái tháo đường đến điều trị nha khoa"
            ),
            RoleInfo(
                id="doctor_endocrine",
                name="Bác sĩ nội tiết",
                description="Dành cho bác sĩ nội tiết cần hiểu về biến chứng răng miệng ở bệnh nhân đái tháo đường"
            )
        ]
        
        return RolesResponse(
            roles=roles,
            default_role="patient_dental",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting roles: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving roles: {str(e)}"
        )

@router.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for medical conversations
    
    - **message**: User's input message
    - **role**: User's role in medical context
    - **session_id**: Optional session identifier
    """
    try:
        # Convert role ID to role name if needed
        role_name = get_role_name(request.role)
        logger.info(f"🔥 New chat request - Role: {role_name} (from: {request.role}), Message: {request.message[:50]}...")
        
        # Prepare shared data for the flow
        shared = {
            "role": role_name,
            "input": request.message,
            "query": "",
            "answer": "",
            "history": [],  # TODO: Implement session-based history
            "session_id": request.session_id
        }
        
        # Run the medical flow
        med_flow.run(shared)
        
        # Extract results
        raw_response = shared.get("answer", "Xin lỗi, tôi không thể trả lời lúc này.")
        raw_suggestions = shared.get("suggestions", [])
        input_type = shared.get("input_type")
        need_clarify = shared.get("need_clarify", False)
        
        # Parse response based on input type
        if input_type == "greeting":
            explanation, summary, question_suggestions = handle_greeting_response(raw_response)
        elif input_type == "statement":
            explanation, summary, question_suggestions = handle_statement_response(raw_response, raw_suggestions)
        else:
            # Medical question or other types
            explanation, summary, question_suggestions = parse_medical_response(raw_response, raw_suggestions)
        
        # Create structured response
        response = ConversationResponse(
            explanation=explanation,
            summary=summary,
            questionSuggestion=question_suggestions,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat(),
            input_type=input_type,
            need_clarify=need_clarify
        )
        
        # Log conversation in background (async)
        background_tasks.add_task(
            log_conversation_background,
            request.message,
            explanation,  # Use the parsed explanation instead
            request.session_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/conversation-history")
async def get_conversation_history(limit: int = 5):
    """
    Get recent conversation history from log file
    
    - **limit**: Number of recent exchanges to return (default: 50)
    """
    try:
        # Read conversation log
        if not os.path.exists("conversation.log"):
            return {"exchanges": [], "total": 0}
        
        with open("conversation.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Parse exchanges
        exchanges = []
        current_exchange = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("user:"):
                if current_exchange:
                    exchanges.append(current_exchange)
                current_exchange = {"user": line[5:].strip()}
            elif line.startswith("bot:"):
                if "user" in current_exchange:
                    current_exchange["bot"] = line[4:].strip()
            elif line == "" and current_exchange:
                exchanges.append(current_exchange)
                current_exchange = {}
        
        # Add last exchange if exists
        if current_exchange:
            exchanges.append(current_exchange)
        
        # Return recent exchanges
        recent_exchanges = exchanges[-limit:] if len(exchanges) > limit else exchanges
        
        return {
            "exchanges": recent_exchanges,
            "total": len(exchanges),
            "showing": len(recent_exchanges)
        }
        
    except Exception as e:
        logger.error(f"❌ Error reading conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading conversation history: {str(e)}"
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
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing conversation history: {str(e)}"
        )

# Background tasks
async def log_conversation_background(user_message: str, bot_response: str, session_id: Optional[str]):
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
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Include router after defining routes
app.include_router(router)

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"🚀 Starting Medical Conversation API on {host}:{port}")
    logger.info(f"📖 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
