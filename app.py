"""
FastAPI server for Medical Conversation System
Main application entry point with simplified structure
"""

import logging
import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from utils.timezone_utils import get_vietnam_time, setup_vietnam_logging
from config.logging_config import logging_config

# Configure logging
if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(
        __name__,
        level=getattr(logging, logging_config.LOG_LEVEL.upper()),
        format_str=logging_config.LOG_FORMAT
    )
else:
    logging.basicConfig(
        level=getattr(logging, logging_config.LOG_LEVEL.upper()),
        format=logging_config.LOG_FORMAT
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


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load knowledge base and initialize components at startup"""
    logger.info("🚀 Starting Medical Conversation API...")
    logger.info("🔄 Loading knowledge base...")

    try:
        from utils.knowledge_base import get_kb, retrieve
        kb = get_kb()
        logger.info(f"✅ Knowledge base loaded successfully!")
        logger.info(f"📊 Total records: {len(kb.df)}")
        logger.info(f"📁 Role-specific dataframes: {list(kb.role_dataframes.keys())}")
        logger.info(f"🔧 BM25 indices created: {list(kb.role_bm25s.keys())}")

        # Test retrieval
        test_results, test_score = retrieve("test", top_k=1)
        logger.info(f"🧪 Test retrieval successful: {len(test_results)} results, score: {test_score:.4f}")

        logger.info("🎉 Main KB startup completed successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load knowledge base: {str(e)}")
        logger.error("⚠️  API will continue but chat functionality may be limited")

    # Load OQA index
    logger.info("🔄 Loading OQA vector index...")
    try:
        from utils.knowledge_base import preload_oqa_index
        preload_oqa_index()
        logger.info("✅ OQA index preloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to preload OQA index: {e}")
        logger.info("⚠️  OQA will be lazy-loaded on first request")

    logger.info("🎉 All startup tasks completed!")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "timestamp": get_vietnam_time().isoformat(),
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
            "timestamp": get_vietnam_time().isoformat(),
        },
    )


# Root endpoint
@app.get("/api")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Conversation API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health",
    }


# Include routers
from api import auth_router, users_router, health_router, chat_router, threads_router, embeddings_router

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(threads_router)
app.include_router(embeddings_router)


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"🚀 Starting Medical Conversation API on {host}:{port}")
    logger.info(f"📖 API Documentation: http://{host}:{port}/api/docs")

    uvicorn.run("app:app", host=host, port=port, reload=debug, log_level="info")
