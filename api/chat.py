"""
Chat API endpoint - Main conversation handling
"""

import uuid
import logging
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database.db import get_db
from database.models import ChatMessage, ChatThread
from utils.auth import get_current_user
from utils.timezone_utils import get_vietnam_time
from utils.helpers import serialize_conversation_history
from utils.role_enum import RoleEnum
from config.timeout_config import timeout_config
from contextlib import contextmanager
from datetime import datetime
import threading

from core.flows import create_med_agent_flow, create_oqa_orthodontist_flow

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize flows
med_flow = create_med_agent_flow()
oqa_flow = create_oqa_orthodontist_flow()


# Pydantic models
class ConversationRequest(BaseModel):
    message: str = Field(..., description="User's message")
    role: str = Field(
        default=RoleEnum.PATIENT_DENTAL.value,
        description="User's role in the medical context",
    )
    session_id: str = Field(..., description="Session identifier for conversation tracking")


class ConversationResponse(BaseModel):
    explanation: str = Field(..., description="Main explanation/answer content")
    questionSuggestion: List[str] | None = Field(None, description="Follow-up question suggestions")
    session_id: str | None = Field(None, description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    input_type: str | None = Field(None, description="Classified input type")
    need_clarify: bool | None = Field(None, description="Whether response needs clarification")


class FlowTimeoutError(Exception):
    """Raised when flow execution exceeds the configured timeout limit"""
    pass


def _create_timeout_checker(start_time: datetime, timeout_seconds: int, timeout_flag: list) -> callable:
    """Create a closure that checks if timeout has occurred"""
    def check_timeout():
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed >= timeout_seconds:
            timeout_flag[0] = True
            logger.error(
                f"⏱️ Flow execution timeout detected: {elapsed:.1f}s >= {timeout_seconds}s"
            )
    return check_timeout


@contextmanager
def flow_timeout(timeout_seconds: int = None):
    """Context manager to enforce timeout on flow execution (cross-platform)"""
    if timeout_seconds is None:
        timeout_seconds = timeout_config.FLOW_EXECUTION_TIMEOUT

    start_time = datetime.now()
    timeout_occurred = [False]

    # Create and start timeout checker
    timeout_checker = _create_timeout_checker(start_time, timeout_seconds, timeout_occurred)
    timer = threading.Timer(timeout_seconds, timeout_checker)
    timer.daemon = True
    timer.start()

    try:
        yield
        # Check if timeout occurred during execution
        if timeout_occurred[0]:
            raise FlowTimeoutError(
                f"Flow execution exceeded {timeout_seconds} seconds timeout"
            )
    finally:
        timer.cancel()


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
    - **session_id**: Required session identifier (thread_id from database)

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

        # Get recent messages (last 6 messages - 3 pairs)
        recent_messages = sorted(thread.messages, key=lambda m: m.timestamp, reverse=True)[:6][::-1]

        # Validate and normalize role
        role_name = request.role

        # Check if role is valid, if not use default
        valid_roles = [role.value for role in RoleEnum]
        if role_name not in valid_roles:
            logger.warning(f"⚠️  Invalid role '{role_name}', using default role '{RoleEnum.PATIENT_DENTAL.value}'")
            role_name = RoleEnum.PATIENT_DENTAL.value

        logger.info(
            f"🔥 New chat request - Role: {role_name}, Message: {request.message[:50]}..."
        )

        # Store user message in database
        user_message_id = str(uuid.uuid4())
        user_message = ChatMessage(
            id=user_message_id,
            thread_id=thread_id,
            role="user",
            content=request.message.strip(),
            timestamp=get_vietnam_time(),
            api_role=request.role
        )

        # Serialize conversation history for the flow
        conversation_history = serialize_conversation_history(recent_messages)

        # Prepare shared data for the flow
        shared = {
            "role": role_name,
            "input": request.message.strip(),
            "query": "",
            "explain": "",
            "conversation_history": conversation_history,
            "session_id": request.session_id,
        }

        # Run chat flow with timeout protection
        try:
            with flow_timeout():
                if role_name == RoleEnum.ORTHODONTIST.value:
                    logger.info(
                        f"🔥 Running OQA flow (timeout: {timeout_config.FLOW_EXECUTION_TIMEOUT}s)"
                    )
                    oqa_flow.run(shared)
                else:
                    logger.info(
                        f"🔥 Running medical flow (timeout: {timeout_config.FLOW_EXECUTION_TIMEOUT}s)"
                    )
                    med_flow.run(shared)
        except FlowTimeoutError as e:
            logger.error(f"⏱️ Flow execution timeout: {e}")
            # Provide graceful timeout response to user
            shared["explain"] = timeout_config.get_timeout_message()
            shared["suggestion_questions"] = []
            shared["input_type"] = "timeout"
            shared["need_clarify"] = False

        explanation = shared.get("explain")
        if not explanation or not isinstance(explanation, str) or not explanation.strip():
            explanation = "Xin lỗi, tôi không thể trả lời câu hỏi ngay lúc này. Bạn chờ một xíu rồi và thử gửi lại câu hỏi cho tôi nhé!"

        suggestion_questions = shared.get("suggestion_questions", [])
        input_type = shared.get("input_type")
        need_clarify = shared.get("need_clarify", False)
        logger.info(f"✅ Flow completed - Need clarify: {need_clarify}")

        response = ConversationResponse(
            explanation=explanation,
            questionSuggestion=suggestion_questions,
            session_id=request.session_id,
            timestamp=get_vietnam_time().isoformat(),
            input_type=input_type,
            need_clarify=need_clarify
        )

        # Create bot message
        bot_message = ChatMessage(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role="bot",
            content=explanation,
            timestamp=get_vietnam_time(),
            suggestions=suggestion_questions,
            need_clarify=need_clarify,
            input_type=input_type
        )

        # Single transaction: create user_message, create bot_message, update thread timestamp
        try:
            db.add(user_message)
            db.add(bot_message)

            # Update thread's updated_at timestamp
            thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
            if thread:
                thread.updated_at = get_vietnam_time()

            db.commit()
        except Exception as e:
            db.rollback()
            raise e

        return response

    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
