from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from audit_logger import log_event
from db import execute_query
from settings_store import get_setting_float

DEFAULT_DAILY_LIMIT_CAD = 50.0
DEFAULT_MONTHLY_LIMIT_CAD = 500.0


@dataclass
class SpendLimitExceededError(Exception):
    period: str
    current_spend_cad: float
    limit_cad: float
    model_used: Optional[str] = None

    def message(self) -> str:
        return (
            f"AI spend limit reached for {self.period}: "
            f"${self.current_spend_cad:.2f} / ${self.limit_cad:.2f} CAD"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "spend_limit",
            "period": self.period,
            "current_spend_cad": round(self.current_spend_cad, 4),
            "limit_cad": round(self.limit_cad, 4),
            "model_used": self.model_used,
            "message": self.message(),
        }


def _period_start(period_type: str, today: Optional[date] = None) -> date:
    today = today or date.today()
    if period_type == "monthly":
        return today.replace(day=1)
    return today


def _get_period_rows(period_type: str, period_start: date) -> list[dict]:
    return execute_query(
        """
        SELECT model_used,
               total_calls,
               prompt_tokens,
               completion_tokens,
               tokens_used,
               spend_cad
        FROM cost_tracking
        WHERE period_type = %s AND period_start = %s
        ORDER BY spend_cad DESC, model_used
        """,
        (period_type, period_start),
    ) or []


def get_cost_snapshot(today: Optional[date] = None) -> dict[str, Any]:
    today = today or date.today()
    daily_start = _period_start("daily", today)
    monthly_start = _period_start("monthly", today)

    daily_rows = _get_period_rows("daily", daily_start)
    monthly_rows = _get_period_rows("monthly", monthly_start)

    daily_limit = get_setting_float("daily_spend_limit_cad", DEFAULT_DAILY_LIMIT_CAD)
    monthly_limit = get_setting_float("monthly_spend_limit_cad", DEFAULT_MONTHLY_LIMIT_CAD)

    def _period_payload(rows: list[dict], limit_cad: float) -> dict[str, Any]:
        spend_cad = round(sum(float(row.get("spend_cad") or 0.0) for row in rows), 4)
        total_calls = sum(int(row.get("total_calls") or 0) for row in rows)
        total_tokens = sum(int(row.get("tokens_used") or 0) for row in rows)
        return {
            "spend_cad": spend_cad,
            "limit_cad": round(limit_cad, 4),
            "remaining_cad": round(max(limit_cad - spend_cad, 0.0), 4),
            "total_calls": total_calls,
            "tokens_used": total_tokens,
            "by_model": [
                {
                    "model_used": row["model_used"],
                    "total_calls": int(row.get("total_calls") or 0),
                    "prompt_tokens": int(row.get("prompt_tokens") or 0),
                    "completion_tokens": int(row.get("completion_tokens") or 0),
                    "tokens_used": int(row.get("tokens_used") or 0),
                    "spend_cad": round(float(row.get("spend_cad") or 0.0), 4),
                }
                for row in rows
            ],
        }

    daily = _period_payload(daily_rows, daily_limit)
    monthly = _period_payload(monthly_rows, monthly_limit)

    return {
        "daily": daily,
        "monthly": monthly,
        "limits": {
            "daily_spend_limit_cad": daily["limit_cad"],
            "monthly_spend_limit_cad": monthly["limit_cad"],
        },
    }


def assert_within_spend_limits(model_used: Optional[str] = None) -> dict[str, Any]:
    snapshot = get_cost_snapshot()
    daily = snapshot["daily"]
    monthly = snapshot["monthly"]

    if daily["limit_cad"] > 0 and daily["spend_cad"] >= daily["limit_cad"]:
        raise SpendLimitExceededError(
            period="daily",
            current_spend_cad=daily["spend_cad"],
            limit_cad=daily["limit_cad"],
            model_used=model_used,
        )

    if monthly["limit_cad"] > 0 and monthly["spend_cad"] >= monthly["limit_cad"]:
        raise SpendLimitExceededError(
            period="monthly",
            current_spend_cad=monthly["spend_cad"],
            limit_cad=monthly["limit_cad"],
            model_used=model_used,
        )

    return snapshot


def record_cost_usage(
    model_used: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_cad: Optional[float],
) -> None:
    if not model_used or cost_cad is None:
        return

    tokens_used = int(prompt_tokens or 0) + int(completion_tokens or 0)
    today = date.today()

    for period_type in ("daily", "monthly"):
        period_start = _period_start(period_type, today)
        execute_query(
            """
            INSERT INTO cost_tracking (
                period_type,
                period_start,
                model_used,
                total_calls,
                prompt_tokens,
                completion_tokens,
                tokens_used,
                spend_cad,
                created_at,
                updated_at
            ) VALUES (%s, %s, %s, 1, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (period_type, period_start, model_used) DO UPDATE
            SET total_calls = cost_tracking.total_calls + 1,
                prompt_tokens = cost_tracking.prompt_tokens + EXCLUDED.prompt_tokens,
                completion_tokens = cost_tracking.completion_tokens + EXCLUDED.completion_tokens,
                tokens_used = cost_tracking.tokens_used + EXCLUDED.tokens_used,
                spend_cad = cost_tracking.spend_cad + EXCLUDED.spend_cad,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                period_type,
                period_start,
                model_used,
                int(prompt_tokens or 0),
                int(completion_tokens or 0),
                tokens_used,
                float(cost_cad),
            ),
            fetch=False,
        )


def build_spend_limit_message(meta: Optional[dict[str, Any]]) -> str:
    if not meta or meta.get("type") != "spend_limit":
        return "AI spend limit reached."
    return (
        f"AI operations paused: {meta.get('period', 'period')} spend is "
        f"${float(meta.get('current_spend_cad') or 0.0):.2f} / "
        f"${float(meta.get('limit_cad') or 0.0):.2f} CAD."
    )


def log_spend_limit_block(
    actor: Optional[str],
    entity_type: str,
    entity_id: Optional[str],
    model_used: Optional[str],
    meta: dict[str, Any],
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload = dict(meta)
    if extra_payload:
        payload.update(extra_payload)
    log_event(
        event_type="OPENAI_SPEND_LIMIT_BLOCK",
        entity_type=entity_type,
        entity_id=entity_id,
        actor=actor,
        model_used=model_used,
        payload=payload,
    )
