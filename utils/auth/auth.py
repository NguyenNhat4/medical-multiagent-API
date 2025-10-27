"""
Authentication utilities for JWT token handling
"""
import os
import time
from datetime import datetime, timedelta
from utils.timezone_utils import get_vietnam_time
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt  # For explicit bcrypt operations
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Users

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "my-super-secret-key-please-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction from requests
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Models for authentication
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

def verify_password(plain_password, hashed_password):
    """Verify password against hashed value"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password for storing"""
    return pwd_context.hash(password)

def safe_hash_password(password: str) -> str:
    """
    Safely hash a password using bcrypt, truncating to 72 bytes to avoid backend errors.
    Note: bcrypt ignores bytes beyond 72; truncation ensures consistent behavior across backends.
    """
    password_bytes = password.encode("utf-8")[:72]
    # Convert back to string for passlib bcrypt
    truncated_password = password_bytes.decode("utf-8", errors="ignore")
    return bcrypt.hash(truncated_password)

def safe_verify_password(password: str, hashed: str) -> bool:
    """
    Safely verify a password against a bcrypt hash, truncating to 72 bytes first.
    Returns False on any verification error.
    """
    try:
        password_bytes = password.encode("utf-8")[:72]
        # Convert back to string for passlib bcrypt
        truncated_password = password_bytes.decode("utf-8", errors="ignore")
        return bcrypt.verify(truncated_password, hashed)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT token"""
    to_encode = data.copy()
    # Note: JWT token expiration should use UTC as per RFC 7519 standard
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get the current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Convert to integer (user ID is stored as str in JWT)
        user_id = int(user_id)
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    except ValueError:
        raise credentials_exception
    
    # Fetch the user from database
    user = db.query(Users).filter(Users.id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
        
    return user
