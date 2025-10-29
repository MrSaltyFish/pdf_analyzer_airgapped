import time
from fastapi import APIRouter, HTTPException
from app.models.api_models import BatchQueryRequest, BatchResponse, BatchQueryResponse
from app.services.model_service import model_manager
from app.services.cache_service import response_cache
from app.core.config import settings

router = APIRouter()

@router.post("/batch-query", response_model=BatchResponse, tags=["Inference"])
async def batch_query(request: BatchQueryRequest):
    """
    Batch inference for multiple queries on the same policy document.
    This endpoint is stateless and does not use chat history.
    """
    
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    results = []
    
    # Process each query sequentially
    for idx, query in enumerate(request.queries):
        query_start = time.time()
        
        query_type = request.query_types[idx] if request.query_types else None
        
        params = {
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'query_type': query_type
        }
        
        # Check cache
        cached_response = None
        if request.use_cache:
            cached_response = response_cache.get(request.policy_text, query, params)
        
        if cached_response:
            response_text = cached_response
            is_cached = True
        else:
            # Generate (using the *non-chat* prompt creator)
            prompt = model_manager.create_chat_prompt(
                request.policy_text, 
                query, 
                history=[],  # No history for batch
                query_type=query_type
            )
            result = model_manager.generate(
                prompt, 
                request.temperature, 
                request.max_tokens, 
                stream=False
            )
            response_text = result['choices'][0]['text'].strip()
            
            # Cache
            if request.use_cache:
                response_cache.set(request.policy_text, query, params, response_text)
            
            is_cached = False
        
        query_time = (time.time() - query_start) * 1000
        
        results.append(BatchQueryResponse(
            query=query,
            response=response_text,
            query_type=query_type,
            processing_time_ms=round(query_time, 2),
            cached=is_cached,
            model_info={
                "model": "mistral-7b-insurance-finetune",
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchResponse(
        results=results,
        total_processing_time_ms=round(total_time, 2),
        queries_processed=len(request.queries)
    )