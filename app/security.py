import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, status

from app.config import get_settings


class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)
        self._lock = Lock()

    def check(self, client_key: str) -> None:
        now = time.time()
        window_start = now - self.window_seconds
        with self._lock:
            queue = self._requests[client_key]
            while queue and queue[0] < window_start:
                queue.popleft()
            if len(queue) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please retry later.",
                )
            queue.append(now)


settings = get_settings()
rate_limiter = InMemoryRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def require_api_key(x_api_key: str = Header(default="")) -> str:
    if x_api_key != settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    rate_limiter.check(x_api_key)
    return x_api_key
