import json
import time
import logging
from typing import Optional, Dict, Any
from access_control import AIOperationsPausedError, assert_ai_operations_enabled
from cost_control import (
    SpendLimitExceededError,
    assert_within_spend_limits,
    log_spend_limit_block,
    record_cost_usage,
)
from db import settings
from services.model_router import route_json_completion
from audit_logger import calculate_cost_cad, log_event, timed_call_latency_ms, timed_call_start

logger = logging.getLogger("lead_allocation")

class AIClient:
    """OpenAI API wrapper with caching and error handling.

    Uses model: gpt-4o-mini (default).
    Used by: ai_leads.py, ai_insights.py routes.
    Email intelligence handles its own model routing in email_intel.py.
    """

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._cache_ttl = 1800  # 30 minutes
        self._client = None
        self.last_call_meta: Dict[str, Any] = {}
        self.last_error_meta: Dict[str, Any] = {}

    def _get_client(self):
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

    def call_json(
        self,
        system: str,
        user: str,
        model: str = "gpt-4o-mini",
        cache_key: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        request_timeout: float = 20.0,
        retries: int = 3,
        actor: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        event_type: str = "OPENAI_API_CALL",
        payload: Optional[dict] = None,
        task_type: str = "reasoning",
    ) -> Optional[dict]:
        """Call OpenAI and parse JSON response. Returns None on failure."""
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set, skipping AI call")
            self.last_error_meta = {}
            return None

        # Check cache
        if cache_key:
            cached = self._get_cache(cache_key)
            if cached:
                self.last_call_meta = {}
                self.last_error_meta = {}
                return cached

        self.last_call_meta = {}
        self.last_error_meta = {}
        for attempt in range(retries):
            try:
                assert_ai_operations_enabled()
                assert_within_spend_limits(model)
                started_at = timed_call_start()
                routed = route_json_completion(
                    task_type=task_type,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    override_model=model if model and model != "gpt-4o-mini" else None,
                )
                response = routed.response
                latency_ms = timed_call_latency_ms(started_at)
                usage = response.usage
                prompt_tokens = (usage.prompt_tokens or 0) if usage else 0
                completion_tokens = (usage.completion_tokens or 0) if usage else 0
                total_tokens = prompt_tokens + completion_tokens
                cost_cad = calculate_cost_cad(routed.model_used, prompt_tokens, completion_tokens)
                self.last_call_meta = {
                    "model_used": routed.model_used,
                    "provider": routed.provider,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_used": total_tokens,
                    "cost_cad": cost_cad,
                    "latency_ms": latency_ms,
                }
                log_event(
                    event_type=event_type,
                    entity_type=entity_type or "ai_operation",
                    entity_id=entity_id,
                    actor=actor,
                    model_used=routed.model_used,
                    tokens_used=total_tokens,
                    cost_cad=cost_cad,
                    latency_ms=latency_ms,
                    payload={
                        "attempt": attempt + 1,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "cache_key": cache_key,
                        "task_type": task_type,
                        "provider": routed.provider,
                        **(payload or {}),
                    },
                )
                record_cost_usage(routed.model_used, prompt_tokens, completion_tokens, cost_cad)
                content = response.choices[0].message.content
                result = json.loads(content)
                if cache_key:
                    self._set_cache(cache_key, result)
                return result
            except AIOperationsPausedError as exc:
                self.last_call_meta = {}
                self.last_error_meta = exc.to_payload()
                log_event(
                    event_type="AI_KILL_SWITCH_BLOCK",
                    entity_type=entity_type or "ai_operation",
                    entity_id=entity_id,
                    actor=actor,
                    model_used=model,
                    payload={**self.last_error_meta, **(payload or {})},
                )
                logger.info(exc.message_text)
                return None
            except SpendLimitExceededError as exc:
                self.last_call_meta = {}
                self.last_error_meta = exc.to_payload()
                log_spend_limit_block(
                    actor=actor,
                    entity_type=entity_type or "ai_operation",
                    entity_id=entity_id,
                    model_used=model,
                    meta=self.last_error_meta,
                    extra_payload=payload,
                )
                logger.warning(exc.message())
                return None
            except json.JSONDecodeError as e:
                logger.error(f"AI JSON parse error (attempt {attempt+1}): {e}")
            except Exception as e:
                logger.error(f"AI API error (attempt {attempt+1}): {e}")
                self.last_call_meta = {}
                self.last_error_meta = {}
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1))
        return None

ai_client = AIClient()
