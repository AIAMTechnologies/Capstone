import json
import time
import logging
from typing import Optional, Dict, Any
from db import settings

logger = logging.getLogger("lead_allocation")

class AIClient:
    """OpenAI API wrapper with caching and error handling."""

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._cache_ttl = 1800  # 30 minutes
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        return self._client

    def _get_cache(self, key: str) -> Optional[dict]:
        if key in self._cache:
            result, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return result
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: dict):
        self._cache[key] = (value, time.time())
        # Evict old entries if cache grows too large
        if len(self._cache) > 500:
            cutoff = time.time() - self._cache_ttl
            self._cache = {k: v for k, v in self._cache.items() if v[1] > cutoff}

    def call_json(self, system: str, user: str, model: str = "gpt-4o-mini", cache_key: str = None) -> Optional[dict]:
        """Call OpenAI and parse JSON response. Returns None on failure."""
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set, skipping AI call")
            return None

        # Check cache
        if cache_key:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        client = self._get_client()
        if not client:
            return None

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                result = json.loads(content)
                if cache_key:
                    self._set_cache(cache_key, result)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"AI JSON parse error (attempt {attempt+1}): {e}")
            except Exception as e:
                logger.error(f"AI API error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
        return None

ai_client = AIClient()
