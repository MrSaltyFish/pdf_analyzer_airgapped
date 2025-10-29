import time
from typing import Dict
from collections import defaultdict, deque
from app.core.config import settings

class RateLimiter:
    """Token bucket rate limiter per client IP"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.clients: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        """Check if request is allowed, return (allowed, remaining)"""
        now = time.time()
        client_history = self.clients[client_ip]
        
        # Remove old requests outside window
        while client_history and client_history[0] < now - self.window:
            client_history.popleft()
        
        if len(client_history) < self.max_requests:
            client_history.append(now)
            return True, self.max_requests - len(client_history)
        
        return False, 0

# Single instance for the app
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS, 
    window_seconds=settings.RATE_LIMIT_WINDOW
)