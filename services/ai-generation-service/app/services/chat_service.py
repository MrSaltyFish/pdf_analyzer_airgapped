import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import asyncio
from pydantic import BaseModel, Field  # <-- FIX: Added imports
from app.core.config import settings

# Structure for a single message in the history
class ChatMessage(BaseModel):  # <-- FIX: Needs BaseModel
    role: str # "user" or "assistant"
    content: str

# Structure for storing a full conversation
class Conversation(BaseModel):  # <-- FIX: Needs BaseModel
    conversation_id: str
    policy_text: str
    history: List[ChatMessage] = Field(default_factory=list)  # <-- FIX: Needs List, Field
    last_access: datetime = Field(default_factory=datetime.now)  # <-- FIX: Needs datetime, Field

class ChatHistoryService:
    """
    Manages conversation history for follow-up questions.
    Uses an LRU cache with TTL to manage memory.
    """
    
    def __init__(self, max_size: int, ttl_hours: int):
        self.conversations: Dict[str, Conversation] = {}
        self.access_order: deque = deque()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = asyncio.Lock() # Protects shared state
    
    async def _evict(self):
        """Evict oldest conversations if over max_size or TTL"""
        now = datetime.now()
        
        # Evict by TTL
        expired_keys = [
            cid for cid, conv in self.conversations.items() 
            if now - conv.last_access > self.ttl
        ]
        for cid in expired_keys:
            if cid in self.conversations:
                del self.conversations[cid]
            if cid in self.access_order:
                self.access_order.remove(cid)

        # Evict by LRU
        while len(self.conversations) > self.max_size:
            try:
                oldest_cid = self.access_order.popleft()
                if oldest_cid in self.conversations:
                    del self.conversations[oldest_cid]
            except IndexError:
                break # Queue is empty
    
    async def start_chat(self, policy_text: str) -> str:
        """Creates a new chat session and returns its ID"""
        async with self.lock:
            await self._evict() # Clean up before adding
            
            conv_id = str(uuid.uuid4())
            conversation = Conversation(
                conversation_id=conv_id,
                policy_text=policy_text
            )
            
            self.conversations[conv_id] = conversation
            self.access_order.append(conv_id)
            return conv_id

    async def get_chat(self, conv_id: str) -> Optional[Tuple[str, List[Dict]]]:
        """Gets the policy text and history for a conversation"""
        async with self.lock:
            if conv_id in self.conversations:
                conversation = self.conversations[conv_id]
                
                # Check TTL
                if datetime.now() - conversation.last_access > self.ttl:
                    del self.conversations[conv_id]
                    self.access_order.remove(conv_id)
                    return None # Conversation expired
                
                # Update LRU
                conversation.last_access = datetime.now()
                self.access_order.remove(conv_id)
                self.access_order.append(conv_id)
                
                history_dicts = [msg.model_dump() for msg in conversation.history] # Use .model_dump() for Pydantic v2
                return conversation.policy_text, history_dicts
            
            return None # Not found
    
    async def add_message(self, conv_id: str, role: str, content: str):
        """Adds a new message (user or assistant) to a conversation"""
        async with self.lock:
            if conv_id in self.conversations:
                conversation = self.conversations[conv_id]
                conversation.history.append(ChatMessage(role=role, content=content))
                conversation.last_access = datetime.now()
    
    def get_active_count(self) -> int:
        """Returns the number of active conversations"""
        return len(self.conversations)

# Single instance for the app
chat_service = ChatHistoryService(
    max_size=settings.CHAT_HISTORY_MAX_SIZE,
    ttl_hours=settings.CHAT_HISTORY_TTL_HOURS
)