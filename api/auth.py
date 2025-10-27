"""
Authentication API endpoints
"""

import os
import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from database.db import get_db
from database.models import Users
from utils.auth import (
    safe_hash_password,
    safe_verify_password,
    create_access_token,
    Token,
)

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/auth", tags=["authentication"])


# Pydantic models
class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str | None = None
    avatar: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


class GoogleLoginReq(BaseModel):
    googleIdToken: str


class LoginReq(BaseModel):
    email: EmailStr
    password: str


@router.post("/google", response_model=TokenResponse)
def login_with_google(payload: GoogleLoginReq, db: Session = Depends(get_db)):
    """
    Login with Google ID token.
    If user doesn't exist, create a new account automatically.
    """
    try:
        idinfo = id_token.verify_oauth2_token(
            payload.googleIdToken, google_requests.Request(), os.getenv("GOOGLE_CLIENT_ID")
        )
        email = idinfo.get("email")
        name = idinfo.get("name", "")
        logger.info(f"Google login attempt for email: {email}, name: {name}")

        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google token: missing email")

        user = db.query(Users).filter(Users.email == email).first()
        if not user:
            random_password = uuid.uuid4().hex
            hashed_password = safe_hash_password(random_password)
            user = Users(email=email, password=hashed_password)
            db.add(user)
            db.commit()
            db.refresh(user)

        token_data = {"sub": str(user.id)}
        access_token = create_access_token(token_data)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserOut(
                id=user.id,
                email=user.email,
                name=name,
                avatar=idinfo.get("picture", ""),
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Google token: {e}")
    except Exception as e:
        logger.error(f"Error in Google login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/login", response_model=TokenResponse)
def login(body: LoginReq, db: Session = Depends(get_db)):
    """
    Login with email and password
    """
    user = db.query(Users).filter(Users.email == body.email).first()
    if not user or not safe_verify_password(body.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create access token
    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserOut(id=user.id, email=user.email)
    )


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    This endpoint is used by Swagger UI for authorization.
    """
    user = db.query(Users).filter(Users.email == form_data.username).first()
    if not user or not safe_verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data)

    return {"access_token": access_token, "token_type": "bearer"}
