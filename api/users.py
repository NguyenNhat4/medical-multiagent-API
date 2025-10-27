"""
User management API endpoints
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from database.db import get_db
from database.models import Users
from utils.auth import safe_hash_password, get_current_user
from utils.timezone_utils import get_vietnam_time

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/users", tags=["users"])


# Pydantic models
class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str | None = None
    avatar: str | None = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=200)


class DeleteUserResponse(BaseModel):
    message: str
    deleted_user: UserOut
    timestamp: str


@router.post("", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user account

    - **email**: User email address
    - **password**: Password (min 6 characters)
    """
    # Duplicate check
    if db.query(Users).filter(Users.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already exists")

    hashed = safe_hash_password(payload.password)
    user = Users(email=payload.email, password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, email=user.email)


@router.get("", response_model=List[UserOut])
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


@router.get("/me", response_model=UserOut)
def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Get current user information

    Returns the profile information of the currently authenticated user.
    Requires valid JWT token in Authorization header.
    """
    return UserOut(id=current_user.id, email=current_user.email)


@router.delete("/{user_id}", response_model=DeleteUserResponse)
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
        timestamp=get_vietnam_time().isoformat(),
    )
