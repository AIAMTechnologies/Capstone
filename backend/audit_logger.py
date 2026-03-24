import logging
import os
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from db import settings

logger = logging.getLogger("lead_allocation")

USD_TO_CAD_RATE = Decimal(os.getenv("AUDIT_USD_TO_CAD_RATE", "1.38"))

# USD pricing per 1M tokens.
MODEL_PRICING_USD = {
    "gpt-4.1-nano": {"input": Decimal("0.10"), "output": Decimal("0.40")},
    "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gpt-4.1-mini": {"input": Decimal("0.40"), "output": Decimal("1.60")},
    "gpt-4.1": {"input": Decimal("2.00"), "output": Decimal("8.00")},
    "gpt-4o": {"input": Decimal("2.50"), "output": Decimal("10.00")},
    "gpt-5-nano": {"input": Decimal("0.10"), "output": Decimal("0.40")},
    "gpt-5-mini": {"input": Decimal("0.30"), "output": Decimal("1.20")},
}


def calculate_cost_cad(
    model_used: Optional[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Optional[float]:
    if not model_used:
        return None

    pricing = MODEL_PRICING_USD.get(model_used)
    if not pricing:
        return None

    input_cost_usd = (Decimal(prompt_tokens or 0) * pricing["input"]) / Decimal(1_000_000)
    output_cost_usd = (Decimal(completion_tokens or 0) * pricing["output"]) / Decimal(1_000_000)
    total_cad = (input_cost_usd + output_cost_usd) * USD_TO_CAD_RATE
    return float(total_cad.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def log_event(
    event_type: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    actor: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_used: Optional[int] = None,
    cost_cad: Optional[float] = None,
    latency_ms: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
) -> bool:
    try:
        conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO audit_log (
                        event_type,
                        entity_type,
                        entity_id,
                        actor,
                        model_used,
                        tokens_used,
                        cost_cad,
                        latency_ms,
                        payload,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """,
                    (
                        event_type,
                        entity_type,
                        str(entity_id) if entity_id is not None else None,
                        actor,
                        model_used,
                        tokens_used,
                        cost_cad,
                        latency_ms,
                        Json(payload or {}),
                    ),
                )
                conn.commit()
                return True
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to write audit_log event %s", event_type)
        return False


def timed_call_start() -> float:
    return time.perf_counter()


def timed_call_latency_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))
