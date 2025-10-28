from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import api_router
from app.core.config import settings
from app.services.model_service import model_manager
from app.services.rate_limiter import rate_limiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Insurance Policy Summarization API...")
    model_manager.load_model()
    
    print(f"✨ API ready at http://localhost:8000")
    print(f"📚 Docs available at http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    print("👋 Shutting down gracefully...")

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Production-ready API for insurance policy document analysis with chat context.",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting"""
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW}s"
            },
            headers={
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(settings.RATE_LIMIT_WINDOW)
            }
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    return response

# Include all API routes
app.include_router(api_router)