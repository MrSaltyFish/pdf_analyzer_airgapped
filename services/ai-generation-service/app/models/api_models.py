from typing import List, Dict, Optional, Any  # <-- Added Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# ============================================================================
# ENUMS
# ============================================================================

class QueryType(str, Enum):
    """Supported query types based on your fine-tuning"""
    DESCRIPTIVE = "descriptive"
    DETAIL = "detail"
    COMPARATIVE = "comparative"
    CONCEPT = "concept"
    COVERAGE = "coverage"
    FINANCIAL = "financial"

# ============================================================================
# CHAT / QUERY MODELS (Replaces InferenceRequest)
# ============================================================================

class ChatRequest(BaseModel):
    """
    Request for a single chat query.
    Supports new conversations or follow-ups.
    """
    policy_text: Optional[str] = Field(None, min_length=10, max_length=50000,
                                       description="Insurance policy document text (REQUIRED for new chats)")
    query: str = Field(..., min_length=5, max_length=500,
                       description="Question about the policy")
    conversation_id: Optional[str] = Field(None, 
                                           description="ID of an existing conversation for follow-up questions")
    
    query_type: Optional[QueryType] = Field(None, 
                                            description="Type of query for optimized prompting")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=50, le=4096)
    stream: bool = Field(False, description="Enable streaming response")
    use_cache: bool = Field(True, description="Use cached results if available")
    
    class Config:
        json_schema_extra = {
            "example_new_chat": {
                "policy_text": "This comprehensive life insurance policy provides...",
                "query": "What are the premium payment options available?",
                "query_type": "detail",
                "stream": False,
            },
            "example_follow_up": {
                "query": "What about annual payments?",
                "conversation_id": "a1b2c3d4-..."
            }
        }

class ChatResponse(BaseModel):
    """Response for single chat query, includes conversation_id"""
    query: str
    response: str
    conversation_id: str
    query_type: Optional[str]
    processing_time_ms: float
    cached: bool = False
    model_info: Dict[str, Any]  # <-- FIX: Was 'any'

# ============================================================================
# BATCH MODELS (Unchanged)
# ============================================================================

class BatchQueryRequest(BaseModel):
    """Batch inference for multiple queries on same policy"""
    policy_text: str = Field(..., min_length=10, max_length=50000)
    queries: List[str] = Field(..., min_items=1, max_items=10,
                               description="Multiple questions to process")
    query_types: Optional[List[QueryType]] = None
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, ge=50, le=4096)
    use_cache: bool = Field(True)
    
    @field_validator('query_types')
    def validate_query_types(cls, v, info):
        if 'values' in info:  # Pydantic v2 compatibility
             queries = info.data.get('queries', [])
        else:
            queries = info.get('queries', [])

        if v is not None and len(v) != len(queries):
            raise ValueError("query_types length must match queries length")
        return v

class BatchQueryResponse(BaseModel):
    """Response for a single query within a batch"""
    query: str
    response: str
    query_type: Optional[str]
    processing_time_ms: float
    cached: bool = False
    model_info: Dict[str, Any]  # <-- FIX: Was 'any'

class BatchResponse(BaseModel):
    """Response for batch inference"""
    results: List[BatchQueryResponse]
    total_processing_time_ms: float
    queries_processed: int

# ============================================================================
# HEALTH & ADMIN MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    active_conversations: int
    response_cache_stats: Dict[str, Any]  # <-- FIX: Was 'any'
    system_info: Dict[str, Any]           # <-- FIX: Was 'any'