from fastapi import APIRouter
from datetime import datetime

from app.models.api_models import HealthResponse
from app.services.model_service import model_manager
from app.services.cache_service import response_cache
from app.services.chat_service import chat_service
from app.core.config import settings

router = APIRouter()

@router.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Chat with follow-up context",
            "Batch querying",
            "Streaming responses",
            "Intelligent response caching",
            "Rate limiting protection",
            "Multi-query type support"
        ]
    }

@router.get("/health", response_model=HealthResponse, tags=["Admin"])
async def health_check():
    """Health check with detailed system information"""
    uptime = (datetime.now() - model_manager.load_time).total_seconds() if model_manager.load_time else 0
    
    return HealthResponse(
        status="healthy" if model_manager.model else "unhealthy",
        model_loaded=model_manager.model is not None,
        uptime_seconds=round(uptime, 2),
        total_requests=model_manager.total_requests,
        active_conversations=chat_service.get_active_count(),
        response_cache_stats=response_cache.get_stats(),
        system_info={
            "model_path": settings.MODEL_PATH.split("\\")[-1], # Show only model name
            "context_window": settings.N_CTX,
            "gpu_layers": settings.N_GPU_LAYERS,
            "cpu_threads": settings.N_THREADS
        }
    )

@router.post("/cache/clear", tags=["Admin"])
async def clear_cache():
    """Clear all cached responses"""
    old_stats = response_cache.get_stats()
    response_cache.clear()
    
    return {
        "message": "Response cache cleared successfully",
        "previous_stats": old_stats,
        "current_stats": response_cache.get_stats()
    }

@router.get("/cache/stats", tags=["Admin"])
async def get_cache_stats():
    """Get detailed cache statistics"""
    return {
        "response_cache_stats": response_cache.get_stats(),
        "cache_enabled": settings.CACHE_ENABLED,
        "ttl_hours": settings.CACHE_TTL_HOURS,
        "chat_history_stats": {
            "active_conversations": chat_service.get_active_count(),
            "max_size": chat_service.max_size,
            "ttl_hours": chat_service.ttl.total_seconds() / 3600
        }
    }

@router.get("/query-types", tags=["General"])
async def get_query_types():
    """Get supported query types and their descriptions"""
    # This data is static, so just return it directly
    return {
        "query_types": [
            {"type": "descriptive", "description": "Summarize key features"},
            {"type": "detail", "description": "Extract specific details"},
            {"type": "comparative", "description": "Compare options"},
            {"type": "concept", "description": "Explain insurance jargon"},
            {"type": "coverage", "description": "Extract coverage limits"},
            {"type": "financial", "description": "Summarize financial terms"}
        ]
    }