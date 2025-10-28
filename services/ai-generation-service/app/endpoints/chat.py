import time
import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse

from app.models.api_models import ChatRequest, ChatResponse
from app.services.model_service import model_manager
from app.services.cache_service import response_cache
from app.services.chat_service import chat_service
from app.core.config import settings

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat_with_policy(request: ChatRequest):
    """
    Main endpoint for policy queries.
    
    Supports:
    - **New conversations**: Provide `policy_text` and `query`.
    - **Follow-up questions**: Provide `conversation_id` and `query`.
    - Intelligent caching for repeated queries.
    - Streaming responses.
    """
    
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    conv_id = request.conversation_id
    
    # --- 1. Identify Conversation ---
    if conv_id:
        chat_data = await chat_service.get_chat(conv_id)
        if not chat_data:
            raise HTTPException(status_code=404, 
                detail=f"Conversation ID '{conv_id}' not found or expired.")
        policy_text, history = chat_data
    
    elif request.policy_text:
        conv_id = await chat_service.start_chat(request.policy_text)
        policy_text = request.policy_text
        history = []
    
    else:
        raise HTTPException(status_code=400, 
            detail="Must provide 'policy_text' for a new conversation or a valid 'conversation_id' for a follow-up.")

    # --- 2. Prepare Parameters ---
    temperature = request.temperature or settings.TEMPERATURE
    max_tokens = request.max_tokens or settings.MAX_TOKENS
    
    params = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'query_type': request.query_type
    }
    
    # --- 3. Check Response Cache ---
    cached_response = None
    if request.use_cache:
        cached_response = response_cache.get(policy_text, request.query, params)
    
    if cached_response:
        # Add cached interaction to history
        await chat_service.add_message(conv_id, "user", request.query)
        await chat_service.add_message(conv_id, "assistant", cached_response)
        
        processing_time = (time.time() - start_time) * 1000
        return ChatResponse(
            query=request.query,
            response=cached_response,
            conversation_id=conv_id,
            query_type=request.query_type,
            processing_time_ms=round(processing_time, 2),
            cached=True,
            model_info=params
        )

    # --- 4. Generate Prompt ---
    prompt = model_manager.create_chat_prompt(
        policy_text, 
        request.query, 
        history, 
        request.query_type
    )
    
    # --- 5. Handle Streaming ---
    if request.stream:
        async def stream_generator():
            full_response = ""
            try:
                for chunk in model_manager.generate(prompt, temperature, max_tokens, stream=True):
                    if 'choices' in chunk:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            full_response += text
                            yield f"data: {json.dumps({'text': text})}\n\n"
            
            except Exception as e:
                print(f"Streaming Error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            finally:
                # Add full interaction to history
                await chat_service.add_message(conv_id, "user", request.query)
                await chat_service.add_message(conv_id, "assistant", full_response)
                
                # Cache the complete response
                if request.use_cache:
                    response_cache.set(policy_text, request.query, params, full_response)
                
                # Send final metadata
                processing_time = (time.time() - start_time) * 1000
                yield f"data: {json.dumps({'done': True, 'conversation_id': conv_id, 'processing_time_ms': round(processing_time, 2)})}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    
    # --- 6. Non-Streaming Inference ---
    try:
        result = model_manager.generate(prompt, temperature, max_tokens, stream=False)
        response_text = result['choices'][0]['text'].strip()
        
        # Add to history
        await chat_service.add_message(conv_id, "user", request.query)
        await chat_service.add_message(conv_id, "assistant", response_text)

        # Cache result
        if request.use_cache:
            response_cache.set(policy_text, request.query, params, response_text)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            query=request.query,
            response=response_text,
            conversation_id=conv_id,
            query_type=request.query_type,
            processing_time_ms=round(processing_time, 2),
            cached=False,
            model_info=params
        )

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")