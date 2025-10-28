import time
import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from app.core.config import settings

class ResponseCacheService:
    """
    LRU cache with TTL for identical (policy, query, params) requests.
    Speeds up repeated queries significantly.
    """
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Dict] = {}
        self.access_order: deque = deque()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, policy_text: str, query: str, params: Dict) -> str:
        """Generate cache key from request parameters"""
        # Create deterministic hash
        content = f"{policy_text[:500]}|{query}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, policy_text: str, query: str, params: Dict) -> Optional[str]:
        """Retrieve from cache if exists and not expired"""
        if not settings.CACHE_ENABLED:
            return None
            
        key = self._generate_key(policy_text, query, params)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            if datetime.now() - entry['timestamp'] < self.ttl:
                self.hits += 1
                # Update LRU order
                self.access_order.remove(key)
                self.access_order.append(key)
                return entry['response']
            else:
                # Expired
                del self.cache[key]
                self.access_order.remove(key)
        
        self.misses += 1
        return None
    
    def set(self, policy_text: str, query: str, params: Dict, response: str):
        """Store in cache with LRU eviction"""
        if not settings.CACHE_ENABLED:
            return
            
        key = self._generate_key(policy_text, query, params)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                 del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_stats(self) -> Dict:
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': round(hit_rate, 2),
            'size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0

# Single instance for the app
response_cache = ResponseCacheService(
    max_size=settings.CACHE_MAX_SIZE, 
    ttl_hours=settings.CACHE_TTL_HOURS
)