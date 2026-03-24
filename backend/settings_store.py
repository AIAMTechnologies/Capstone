import json
from typing import Any, Optional

from db import execute_query


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    rows = execute_query("SELECT value_text, value_boolean FROM settings WHERE key = %s", (key,))
    if not rows:
        return default
    row = rows[0]
    if row.get("value_text") is not None:
        return row.get("value_text")
    if row.get("value_boolean") is not None:
        return "true" if row.get("value_boolean") else "false"
    return default


def get_setting_float(key: str, default: float) -> float:
    value = get_setting(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_setting_bool(key: str, default: bool) -> bool:
    rows = execute_query("SELECT value_text, value_boolean FROM settings WHERE key = %s", (key,))
    if not rows:
        return default
    row = rows[0]
    if row.get("value_boolean") is not None:
        return bool(row.get("value_boolean"))
    value = row.get("value_text")
    if value in (None, ""):
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_setting_json(key: str, default: Any) -> Any:
    value = get_setting(key)
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def set_setting(
    key: str,
    value: Any,
    updated_by: Optional[str] = None,
    value_boolean: Optional[bool] = None,
) -> None:
    if isinstance(value, (dict, list)):
        value_text = json.dumps(value)
    elif value is None:
        value_text = None
    else:
        value_text = str(value)

    execute_query(
        """
        INSERT INTO settings (key, value_text, value_boolean, updated_by, created_at, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (key) DO UPDATE
        SET value_text = EXCLUDED.value_text,
            value_boolean = EXCLUDED.value_boolean,
            updated_by = EXCLUDED.updated_by,
            updated_at = CURRENT_TIMESTAMP
        """,
        (key, value_text, value_boolean, updated_by),
        fetch=False,
    )
