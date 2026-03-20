import json
import os
import logging
import html
import threading
import asyncio
import base64
from datetime import datetime, timedelta
from copy import deepcopy
from typing import Optional, List, Tuple
from urllib.parse import urlencode, unquote

import requests as http_requests
from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import AdminUser, get_current_user
from audit_logger import calculate_cost_cad, log_event, timed_call_latency_ms, timed_call_start
from cost_control import SpendLimitExceededError, assert_within_spend_limits, log_spend_limit_block, record_cost_usage
from db import execute_query, get_db_connection

logger = logging.getLogger("lead_allocation")

router = APIRouter(prefix="/api/email-intel", tags=["Email Intelligence"])

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
MS_LOGIN_BASE = "https://login.microsoftonline.com"
OAUTH_SCOPES = "Mail.Read Mail.Read.Shared Group.Read.All Group-Conversation.Read.All offline_access User.Read"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENCRYPTION_KEY = os.getenv("EMAIL_ENCRYPTION_KEY")
OPENAI_EMAIL_MODEL_CANDIDATES = [
    model.strip()
    for model in os.getenv("OPENAI_EMAIL_MODELS", "gpt-4.1-nano,gpt-4o-mini").split(",")
    if model.strip()
]
OPENAI_REASONING_MODEL_CANDIDATES = [
    model.strip()
    for model in os.getenv("OPENAI_REASONING_MODELS", "gpt-4.1-mini,gpt-4o-mini").split(",")
    if model.strip()
]
ANTHROPIC_REASONING_MODEL = os.getenv("ANTHROPIC_REASONING_MODEL", "claude-sonnet-4-20250514")

# Env-based MS config (auto-seeds DB on first access)
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI", "http://localhost:8000/api/email-intel/oauth/callback-redirect")


MAX_EMAIL_BODY_CHARS = 600  # Keep short for speed — subject + preview is usually enough
AI_BATCH_SIZE = 8  # Process this many AI calls concurrently
GRAPH_SYNC_PAGE_SIZE = 1000
try:
    EMAIL_SYNC_HISTORY_LIMIT = max(1, int(os.getenv("EMAIL_SYNC_HISTORY_LIMIT", "10000")))
except ValueError:
    EMAIL_SYNC_HISTORY_LIMIT = 10000

MAILBOX_TYPE_CONNECTED = "connected"
MAILBOX_TYPE_SHARED = "shared"
MAILBOX_TYPE_GROUP = "group"
VALID_TARGET_MAILBOX_TYPES = {
    MAILBOX_TYPE_CONNECTED,
    MAILBOX_TYPE_SHARED,
    MAILBOX_TYPE_GROUP,
}


# ---------------------------------------------------------------------------
# Token Encryption Helpers
# ---------------------------------------------------------------------------

def get_fernet():
    if not ENCRYPTION_KEY:
        raise HTTPException(status_code=500, detail="Email encryption key not configured")
    return Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)


def encrypt_value(value: str) -> str:
    if not value:
        return value
    return get_fernet().encrypt(value.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    if not encrypted:
        return encrypted
    return get_fernet().decrypt(encrypted.encode()).decode()


def _decode_jwt_payload(token: str) -> dict:
    if not token:
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
    except Exception:
        return {}


def _infer_user_email_from_token(token: str) -> str:
    payload = _decode_jwt_payload(token)
    return (
        (payload.get("preferred_username") or "").strip()
        or (payload.get("upn") or "").strip()
        or (payload.get("email") or "").strip()
        or (payload.get("unique_name") or "").strip()
    ).lower()


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

def audit_log(action: str, details: str = "", user: str = ""):
    """Log every email-related action to DB for security audit trail."""
    try:
        execute_query(
            "INSERT INTO email_audit_log (action, details, performed_by, performed_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)",
            (action, details, user), fetch=False
        )
    except Exception as exc:
        logger.warning("Email audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

_last_sync_time = None
_sync_state_lock = threading.Lock()
_sync_state = {
    "is_running": False,
    "started_at": None,
    "finished_at": None,
    "current_phase": None,
    "last_error": None,
    "last_result": None,
    "synced": 0,
    "matched": 0,
    "flagged_for_review": 0,
}


def _serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _snapshot_sync_state() -> dict:
    with _sync_state_lock:
        state = deepcopy(_sync_state)
    for key in ("started_at", "finished_at"):
        state[key] = _serialize_datetime(state.get(key))
    return state


def _update_sync_state(**kwargs) -> dict:
    with _sync_state_lock:
        _sync_state.update(kwargs)
        state = deepcopy(_sync_state)
    for key in ("started_at", "finished_at"):
        state[key] = _serialize_datetime(state.get(key))
    return state


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OAuthConfigIn(BaseModel):
    ms_tenant_id: Optional[str] = None
    ms_client_id: Optional[str] = None
    ms_client_secret: Optional[str] = None
    ms_redirect_uri: Optional[str] = None
    shared_mailbox_email: Optional[str] = None
    target_mailbox_type: Optional[str] = None
    sync_enabled: Optional[bool] = None
    sync_interval_minutes: Optional[int] = None


class OAuthCallbackIn(BaseModel):
    code: str


# ---------------------------------------------------------------------------
# Helpers -- DB convenience wrappers
# ---------------------------------------------------------------------------

def _ensure_sync_config_columns():
    execute_query(
        "ALTER TABLE email_sync_config ADD COLUMN IF NOT EXISTS shared_mailbox_email VARCHAR(255)",
        fetch=False,
    )
    execute_query(
        "ALTER TABLE email_sync_config ADD COLUMN IF NOT EXISTS target_mailbox_type VARCHAR(32)",
        fetch=False,
    )
    execute_query(
        "ALTER TABLE email_sync_config ADD COLUMN IF NOT EXISTS target_group_id VARCHAR(255)",
        fetch=False,
    )


def _get_sync_config() -> Optional[dict]:
    _ensure_sync_config_columns()
    rows = execute_query("SELECT * FROM email_sync_config ORDER BY id DESC LIMIT 1")
    if rows:
        return rows[0]
    # Auto-seed from env vars if DB is empty
    if MS_TENANT_ID and MS_CLIENT_ID and MS_CLIENT_SECRET:
        encrypted_secret = encrypt_value(MS_CLIENT_SECRET)
        execute_query(
            "INSERT INTO email_sync_config (ms_tenant_id, ms_client_id, ms_client_secret, ms_redirect_uri) VALUES (%s, %s, %s, %s)",
            (MS_TENANT_ID, MS_CLIENT_ID, encrypted_secret, MS_REDIRECT_URI),
            fetch=False,
        )
        rows = execute_query("SELECT * FROM email_sync_config ORDER BY id DESC LIMIT 1")
        return rows[0] if rows else None
    return None


def _get_sync_config_decrypted() -> Optional[dict]:
    """Get sync config with sensitive fields decrypted."""
    config = _get_sync_config()
    if not config:
        return None
    for field in ("access_token", "refresh_token", "ms_client_secret"):
        if config.get(field):
            try:
                config[field] = decrypt_value(config[field])
            except Exception as e:
                # Value may not be encrypted yet (legacy data)
                print(f"[EMAIL_INTEL] WARNING: Failed to decrypt field '{field}': {e}")
                pass
    return config


def _upsert_sync_config(**kwargs):
    """Insert or update the single sync-config row. Encrypts sensitive fields."""
    _ensure_sync_config_columns()
    # Encrypt sensitive values before storing
    for field in ("access_token", "refresh_token", "ms_client_secret"):
        if field in kwargs and kwargs[field]:
            kwargs[field] = encrypt_value(kwargs[field])

    existing = _get_sync_config()
    if existing:
        set_parts = []
        values = []
        for k, v in kwargs.items():
            set_parts.append(f"{k} = %s")
            values.append(v)
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        values.append(existing["id"])
        execute_query(
            f"UPDATE email_sync_config SET {', '.join(set_parts)} WHERE id = %s",
            tuple(values),
            fetch=False,
        )
    else:
        cols = list(kwargs.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        execute_query(
            f"INSERT INTO email_sync_config ({', '.join(cols)}) VALUES ({placeholders})",
            tuple(kwargs.values()),
            fetch=False,
        )


def _get_target_mailbox_email(config: Optional[dict]) -> str:
    if not config:
        return ""
    mailbox_type = _get_target_mailbox_type(config)
    if mailbox_type == MAILBOX_TYPE_CONNECTED:
        return ((config.get("user_email") or "").strip()).lower()
    return (
        (config.get("shared_mailbox_email") or "").strip()
        or (config.get("user_email") or "").strip()
    ).lower()


def _get_target_mailbox_type(config: Optional[dict]) -> str:
    raw = ((config or {}).get("target_mailbox_type") or "").strip().lower()
    if raw in VALID_TARGET_MAILBOX_TYPES:
        return raw
    if (config or {}).get("shared_mailbox_email"):
        return MAILBOX_TYPE_SHARED
    return MAILBOX_TYPE_CONNECTED


def _graph_mailbox_base(config: Optional[dict]) -> str:
    if _get_target_mailbox_type(config) == MAILBOX_TYPE_GROUP:
        return f"{GRAPH_BASE}/me"
    mailbox_email = _get_target_mailbox_email(config)
    if not mailbox_email:
        return f"{GRAPH_BASE}/me"
    user_email = ((config or {}).get("user_email") or "").strip().lower()
    if mailbox_email == user_email or not (config or {}).get("shared_mailbox_email"):
        return f"{GRAPH_BASE}/me"
    return f"{GRAPH_BASE}/users/{mailbox_email}"


def _get_matching_mailbox_email(config: Optional[dict]) -> str:
    if not config:
        return ""
    if _get_target_mailbox_type(config) == MAILBOX_TYPE_GROUP:
        return ((config.get("user_email") or "").strip() or _get_target_mailbox_email(config)).lower()
    return _get_target_mailbox_email(config)


def _extract_graph_recipient_email(recipient: Optional[dict]) -> str:
    if not recipient:
        return ""
    email_address = recipient.get("emailAddress") or recipient
    return ((email_address or {}).get("address") or "").strip().lower()


def _extract_graph_recipient_name(recipient: Optional[dict]) -> str:
    if not recipient:
        return ""
    email_address = recipient.get("emailAddress") or recipient
    return ((email_address or {}).get("name") or "").strip()


def _flatten_graph_recipients(*recipient_groups: Optional[list]) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()
    for group in recipient_groups:
        for recipient in group or []:
            email = _extract_graph_recipient_email(recipient)
            if email and email not in seen:
                flattened.append(email)
                seen.add(email)
    return flattened


def _resolve_group_target(token: str, config: Optional[dict]) -> dict:
    target_mailbox = _get_target_mailbox_email(config)
    if not target_mailbox:
        raise RuntimeError("Group mailbox mode requires a target group email.")

    cached_group_id = ((config or {}).get("target_group_id") or "").strip()
    if cached_group_id:
        try:
            group = _graph_get(
                token,
                f"{GRAPH_BASE}/groups/{cached_group_id}?$select=id,mail,displayName",
            )
            group_mail = (group.get("mail") or "").strip().lower()
            if not group_mail or group_mail == target_mailbox:
                return group
        except RuntimeError as exc:
            if "Authorization_RequestDenied" in str(exc):
                raise RuntimeError(
                    "Microsoft 365 Group sync requires delegated Group.Read.All and "
                    "Group-Conversation.Read.All permissions, admin consent, and a fresh Outlook reconnect."
                ) from exc

    escaped_target_mailbox = target_mailbox.replace("'", "''")
    query = urlencode(
        {
            "$filter": f"mail eq '{escaped_target_mailbox}'",
            "$select": "id,mail,displayName",
        }
    )
    try:
        data = _graph_get(token, f"{GRAPH_BASE}/groups?{query}")
    except RuntimeError as exc:
        if "Authorization_RequestDenied" in str(exc):
            raise RuntimeError(
                "Microsoft 365 Group sync requires delegated Group.Read.All and "
                "Group-Conversation.Read.All permissions, admin consent, and a fresh Outlook reconnect."
            ) from exc
        raise

    groups = data.get("value") or []
    if not groups:
        raise RuntimeError(f"No Microsoft 365 Group with the email address {target_mailbox} was found in Graph.")

    group = groups[0]
    if group.get("id") and group.get("id") != cached_group_id:
        _upsert_sync_config(target_group_id=group["id"])
    return group


def _is_group_internal_sender(sender_email: str, config: Optional[dict]) -> bool:
    sender = (sender_email or "").lower().strip()
    if not sender:
        return False

    aliases = {
        ((config or {}).get("user_email") or "").strip().lower(),
        _get_target_mailbox_email(config),
        _get_matching_mailbox_email(config),
    }
    aliases.discard("")
    if sender in aliases:
        return True

    internal_seed = _get_matching_mailbox_email(config) or _get_target_mailbox_email(config)
    internal_domain = internal_seed.split("@")[-1] if internal_seed else ""
    return bool(internal_domain and sender.endswith(f"@{internal_domain}"))


def _group_post_message_id(group_id: str, post_id: str) -> str:
    return f"group-post:{group_id}:{post_id}"


def _fetch_group_thread_posts(token: str, group_id: str, thread_id: str) -> list[dict]:
    query = urlencode(
        {
            "$top": GRAPH_SYNC_PAGE_SIZE,
            "$select": "id,conversationId,conversationThreadId,receivedDateTime,createdDateTime,body,from,sender",
        }
    )
    data = _graph_get(token, f"{GRAPH_BASE}/groups/{group_id}/threads/{thread_id}/posts?{query}")
    return data.get("value") or []


# ---------------------------------------------------------------------------
# AI helpers — OpenAI gpt-4o-mini for bulk email analysis (cheap),
#              Anthropic Claude for high-value lead context summaries (rare)
# ---------------------------------------------------------------------------

import re as _re


def _extract_json(text: str) -> dict:
    """Extract JSON from AI response, handling markdown code fences."""
    text = text.strip()
    fence_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_match = _re.search(r'\{.*\}', text, _re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass
        raise


def _get_openai_client():
    """Get OpenAI client for cheap bulk email analysis."""
    try:
        from openai import OpenAI
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set, AI analysis unavailable")
            return None
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logger.error("openai package not installed")
        return None


def _get_anthropic_client():
    """Get Anthropic client for high-value reasoning tasks only."""
    try:
        import anthropic
        if not ANTHROPIC_API_KEY:
            return None
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        return None


_DEFAULT_AI_RESULT = {
    "summary": "",
    "sentiment": "neutral",
    "action_items": [],
    "is_deal_related": False,
    "deal_stage": "inquiry",
}

# --- Cost tracking ---
# Pricing per 1M tokens (USD)
_MODEL_PRICING = {
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1":       {"input": 2.00, "output": 8.00},
    "gpt-4o":        {"input": 2.50, "output": 10.00},
    "gpt-5-nano":    {"input": 0.10, "output": 0.40},
    "gpt-5-mini":    {"input": 0.30, "output": 1.20},
}
_cost_tracker_lock = threading.Lock()
_cost_tracker = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "calls": 0,
    "by_model": {},
}


def _normalize_email_text(value: str, max_chars: Optional[int] = None) -> str:
    if not value:
        return ""
    normalized = html.unescape(value)
    normalized = _re.sub(r"<[^>]+>", " ", normalized)
    normalized = _re.sub(r"\s+", " ", normalized).strip()
    if max_chars is not None:
        return normalized[:max_chars]
    return normalized


def _normalize_phone(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


_FORM_FIELD_LABELS = {
    "name": ["name"],
    "first_name": ["first name", "first_name"],
    "last_name": ["last name", "last_name"],
    "email": ["email", "email address"],
    "phone": ["phone", "phone number", "primary phone", "cell phone", "work phone"],
    "address": ["address"],
    "city": ["city"],
    "province": ["province", "state"],
    "company_name": ["company name", "company name (if applicable)", "company"],
    "project_type": ["type of project", "project type"],
    "project_location_city_province": ["project location (city, province)", "location (city, province)"],
    "project_location_city": ["project location (city)", "location (city)"],
    "project_location_province": ["project location (province)", "location (province)"],
    "submission_host": ["submission host"],
    "submission_date": ["submission date"],
    "submission_ip": ["submission ip"],
    "submission_source": ["submission source"],
    "submission_data": ["submission data"],
    "comments": ["comments / questions", "comments", "message"],
}

_ALL_FORM_FIELD_VARIANTS = [
    variant
    for variants in _FORM_FIELD_LABELS.values()
    for variant in variants
]


def _form_label_pattern(label: str) -> str:
    parts = [_re.escape(part) for part in _re.split(r"[\s_]+", label.strip()) if part]
    return r"[\s_]*".join(parts)


def _clean_extracted_field_value(value: str) -> str:
    cleaned = _normalize_email_text(value or "")
    cleaned = _re.sub(r"^[\s,;:-]+|[\s,;:-]+$", "", cleaned)
    return cleaned


def _extract_labeled_value(text: str, labels: List[str]) -> Optional[str]:
    if not text:
        return None
    label_pattern = "|".join(_form_label_pattern(label) for label in labels)
    next_pattern = "|".join(_form_label_pattern(label) for label in _ALL_FORM_FIELD_VARIANTS)
    match = _re.search(
        rf"(?is)\b(?:{label_pattern})\s*:\s*(.+?)(?=\s+\b(?:{next_pattern})\s*:|$)",
        text,
    )
    if not match:
        return None
    value = _clean_extracted_field_value(match.group(1))
    return value or None


def _extract_form_fields(text: str) -> dict:
    fields = {}
    for key, labels in _FORM_FIELD_LABELS.items():
        value = _extract_labeled_value(text, labels)
        if value:
            fields[key] = value
    return fields


def _extract_host_from_urlish(value: str) -> str:
    if not value:
        return ""
    match = _re.search(r"(?i)(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})(?:/|$)", value.strip())
    return (match.group(1).lower() if match else value.strip().lower()).strip()


def _extract_first_urlish(value: str) -> str:
    if not value:
        return ""
    match = _re.search(r"(?i)((?:https?://)?(?:www\.)?[^\s]+)", value.strip())
    return (match.group(1).strip() if match else value.strip())


def _extract_email_text_from_graph_message(message: dict) -> tuple[str, str]:
    body_content = ((message.get("body") or {}).get("content") or "").strip()
    body_text = _normalize_email_text(body_content)
    graph_preview = _normalize_email_text(message.get("bodyPreview", "") or "", 1200)
    full_preview = _normalize_email_text(body_text, 1200)
    body_preview = full_preview if len(full_preview) > len(graph_preview) else graph_preview
    return body_preview, body_text[:4000]


def _fetch_graph_message_content(token: str, ms_message_id: str, config: Optional[dict] = None) -> tuple[str, str]:
    if not token or not ms_message_id:
        return "", ""
    if ms_message_id.startswith("group-post:"):
        return "", ""
    bases = []
    primary_base = _graph_mailbox_base(config)
    bases.append(primary_base)
    if primary_base != f"{GRAPH_BASE}/me":
        bases.append(f"{GRAPH_BASE}/me")

    for base in bases:
        resp = http_requests.get(
            f"{base}/messages/{ms_message_id}?$select=body,bodyPreview",
            headers={
                "Authorization": f"Bearer {token}",
                "Prefer": 'outlook.body-content-type="text"',
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return _extract_email_text_from_graph_message(resp.json())
        logger.warning("Graph body fetch failed for %s via %s: %s", ms_message_id, base, resp.status_code)
    return "", ""


def _hydrate_email_record_for_parsing(email_record: dict, token: str = "", config: Optional[dict] = None) -> dict:
    enriched = dict(email_record)
    current_preview = _normalize_email_text(enriched.get("body_preview") or "", 1200)
    current_body = _normalize_email_text(enriched.get("body_text") or "", 4000)
    sender_email = (enriched.get("sender_email") or "").lower().strip()
    subject = (enriched.get("subject") or "").lower()
    needs_refresh = (
        not current_body
        and enriched.get("ms_message_id")
        and (
            sender_email.startswith("lead@")
            or "form submission" in subject
            or "request a quote" in subject
            or current_preview.lower().find("first_name:") >= 0
        )
    )
    if needs_refresh and token:
        fetched_preview, fetched_body = _fetch_graph_message_content(token, enriched["ms_message_id"], config)
        if len(fetched_body) > len(current_body) or len(fetched_preview) > len(current_preview):
            enriched["body_preview"] = fetched_preview or current_preview
            enriched["body_text"] = fetched_body or current_body
            if enriched.get("id"):
                execute_query(
                    "UPDATE email_messages SET body_preview = %s, body_text = %s WHERE id = %s",
                    (enriched["body_preview"], enriched["body_text"], enriched["id"]),
                    fetch=False,
                )
            return enriched
    enriched["body_preview"] = current_preview
    enriched["body_text"] = current_body
    return enriched


def _openai_json_completion(
    system_prompt: str,
    user_prompt: str,
    model_candidates: List[str],
    max_tokens: int = 200,
) -> Tuple[Optional[dict], Optional[str]]:
    """Try each model candidate in order until one succeeds."""
    client = _get_openai_client()
    if client is None:
        return None, None

    for model in model_candidates:
        try:
            assert_within_spend_limits(model)
            started_at = timed_call_start()
            request_kwargs = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if model.startswith(("gpt-5", "o3", "o4")):
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens
                request_kwargs["temperature"] = 0.1

            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or "{}"
            latency_ms = timed_call_latency_ms(started_at)

            # Track token usage & cost
            usage = response.usage
            inp = (usage.prompt_tokens or 0) if usage else 0
            out = (usage.completion_tokens or 0) if usage else 0
            cost_cad = calculate_cost_cad(model, inp, out)
            if usage:
                pricing = _MODEL_PRICING.get(model, {"input": 0.15, "output": 0.60})
                cost = (inp * pricing["input"] / 1_000_000) + (out * pricing["output"] / 1_000_000)
                with _cost_tracker_lock:
                    _cost_tracker["total_input_tokens"] += inp
                    _cost_tracker["total_output_tokens"] += out
                    _cost_tracker["total_cost_usd"] += cost
                    _cost_tracker["calls"] += 1
                    if model not in _cost_tracker["by_model"]:
                        _cost_tracker["by_model"][model] = {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
                    _cost_tracker["by_model"][model]["input"] += inp
                    _cost_tracker["by_model"][model]["output"] += out
                    _cost_tracker["by_model"][model]["cost"] += cost
                    _cost_tracker["by_model"][model]["calls"] += 1
            log_event(
                event_type="OPENAI_API_CALL",
                entity_type="email_sync",
                actor="system",
                model_used=model,
                tokens_used=inp + out,
                cost_cad=cost_cad,
                latency_ms=latency_ms,
                payload={"operation": "email_intel_json_completion", "prompt_tokens": inp, "completion_tokens": out},
            )
            record_cost_usage(model, inp, out, cost_cad)

            return json.loads(content), model
        except SpendLimitExceededError as exc:
            meta = exc.to_payload()
            log_spend_limit_block(
                actor="system",
                entity_type="email_sync",
                entity_id=None,
                model_used=model,
                meta=meta,
                extra_payload={"operation": "email_intel_json_completion"},
            )
            logger.warning(exc.message())
            return None, None
        except Exception as exc:
            logger.warning("OpenAI JSON call failed for %s: %s", model, exc)

    return None, None


def _anthropic_json_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 200,
) -> Tuple[Optional[dict], Optional[str]]:
    client = _get_anthropic_client()
    if client is None:
        return None, None
    try:
        response = client.messages.create(
            model=ANTHROPIC_REASONING_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _extract_json(response.content[0].text), ANTHROPIC_REASONING_MODEL
    except Exception as exc:
        logger.warning("Anthropic JSON call failed: %s", exc)
        return None, None


def _reasoning_json_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
) -> Tuple[Optional[dict], Optional[str]]:
    """Try OpenAI reasoning models first, fall back to Anthropic."""
    result, model = _openai_json_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_candidates=OPENAI_REASONING_MODEL_CANDIDATES,
        max_tokens=max_tokens,
    )
    if result is not None:
        return result, model
    return _anthropic_json_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )


async def analyze_email_with_ai(
    email_body: str, email_subject: str, lead_data: dict = None
) -> dict:
    """Use cheapest available OpenAI model for email analysis."""
    lead_context = ""
    if lead_data:
        lead_context = (
            f"\nLead: {lead_data.get('first_name', '')} {lead_data.get('last_name', '')} "
            f"({lead_data.get('email', '')}) Status: {lead_data.get('status', '')}"
        )

    result, model = _openai_json_completion(
        system_prompt=(
            "Analyse the sales email. Return JSON: "
            '{"summary":"1 sentence","sentiment":"positive|neutral|negative",'
            '"action_items":["..."],"is_deal_related":true/false,'
            '"deal_stage":"inquiry|quoting|negotiation|closing|closed"}'
        ),
        user_prompt=(
            f"Subject: {email_subject}\n"
            f"Body:\n{_normalize_email_text(email_body, MAX_EMAIL_BODY_CHARS)}"
            f"{lead_context}"
        ),
        model_candidates=OPENAI_EMAIL_MODEL_CANDIDATES,
    )
    if result is None:
        return dict(_DEFAULT_AI_RESULT)
    logger.info("Email analysis completed with model %s", model)
    return {**dict(_DEFAULT_AI_RESULT), **result}


# ---------------------------------------------------------------------------
# Lead matching — match the OTHER party in the email (not the mailbox owner)
# ---------------------------------------------------------------------------

def _load_leads_cache(mailbox_email: str = "") -> dict:
    """
    Pre-load ALL leads into in-memory lookup dicts for fast matching.
    Called once per sync, NOT per email.
    Returns dict with lookup tables.
    """
    mailbox_email = mailbox_email.lower().strip()
    exclude_clause = ""
    exclude_params: tuple = ()
    if mailbox_email:
        exclude_clause = " WHERE LOWER(email) != %s"
        exclude_params = (mailbox_email,)

    rows = execute_query(
        f"SELECT id, email, dealer_email, first_name, last_name, phone, created_at "
        f"FROM leads{exclude_clause} ORDER BY created_at DESC",
        exclude_params,
    ) or []

    # Build lookup dicts
    by_email: dict[str, int] = {}          # lowercase email -> lead id
    by_dealer_email: dict[str, int] = {}   # lowercase dealer_email -> lead id
    by_name: dict[str, int] = {}           # "first last" -> lead id
    phones: list[tuple[int, str]] = []     # [(lead_id, normalized_phone), ...]

    for r in rows:
        lid = r["id"]
        email = (r.get("email") or "").lower().strip()
        dealer_email = (r.get("dealer_email") or "").lower().strip()
        first = (r.get("first_name") or "").lower().strip()
        last = (r.get("last_name") or "").lower().strip()
        phone = _normalize_phone((r.get("phone") or "").strip())

        # First match wins (rows are ordered by created_at DESC = newest first)
        if email and email not in by_email:
            by_email[email] = lid
        if dealer_email and dealer_email not in by_dealer_email:
            by_dealer_email[dealer_email] = lid
        if first and last:
            name_key = f"{first} {last}"
            if name_key not in by_name:
                by_name[name_key] = lid
        if (
            phone
            and len(phone) >= 10
            and not phone.startswith("0000")
            and len(set(phone)) > 2
        ):
            phones.append((lid, phone))

    print(f"[EMAIL_INTEL] Leads cache loaded: {len(rows)} leads, {len(by_email)} emails, {len(by_dealer_email)} dealer emails, {len(by_name)} names, {len(phones)} phones", flush=True)
    return {
        "by_email": by_email,
        "by_dealer_email": by_dealer_email,
        "by_name": by_name,
        "phones": phones,
    }


def _extract_candidate_lead_fields(email_record: dict, mailbox_email: str) -> Optional[dict]:
    """Extract customer lead fields from a likely form-submission email."""
    body = _normalize_email_text(
        f"{email_record.get('body_text') or ''} {email_record.get('body_preview') or ''}",
        8000,
    )
    subject = email_record.get("subject", "")
    parsed_fields = _extract_form_fields(body)
    sender_email = (email_record.get("sender_email") or "").lower()

    # Only auto-create from form submissions or lead assignment emails
    is_form = (
        "form submission" in body.lower()
        or "form submission" in subject.lower()
        or "squarespace" in sender_email
        or _re.search(r'\b(lead|Lead)\b.*-.*\d{2}/\d{2}/\d{4}', subject)
        or "contact form submission" in subject.lower()
        or sender_email.startswith("lead@")
        or "submission_host" in parsed_fields
        or "submission_source" in parsed_fields
    )
    if not is_form:
        return None

    # Extract customer info — try multiple formats
    customer_email = None
    customer_first = None
    customer_last = None
    customer_phone = None
    customer_city = None
    customer_province = None
    customer_company = None
    customer_project_type = None
    submission_host = parsed_fields.get("submission_host", "")
    submission_source = parsed_fields.get("submission_source", "")
    comments = parsed_fields.get("comments", "")

    if parsed_fields.get("first_name"):
        customer_first = parsed_fields["first_name"].split()[0]
    if parsed_fields.get("last_name"):
        last_parts = parsed_fields["last_name"].split()
        customer_last = last_parts[-1] if last_parts else None

    # Formats that only provide a single full-name label.
    if not customer_first and parsed_fields.get("name"):
        name_parts = parsed_fields["name"].split()
        if len(name_parts) >= 2:
            customer_first = name_parts[0]
            customer_last = " ".join(name_parts[1:])
    if not customer_first:
        name_m = _re.search(r'(?i)\bname\s*:\s*([A-Z][A-Za-zÀ-ÿ\'\-]+)\s+([A-Z][A-Za-zÀ-ÿ\'\-\s]+)', body)
        if name_m:
            customer_first = name_m.group(1).strip()
            customer_last = name_m.group(2).strip()

    # Extract email
    if parsed_fields.get("email"):
        em_m = _re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', parsed_fields["email"])
        if em_m:
            customer_email = em_m.group(1).lower()
    if not customer_email:
        em_m = _re.search(r'Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', body, _re.I)
        if em_m:
            customer_email = em_m.group(1).lower()

    # Extract phone
    if parsed_fields.get("phone"):
        ph_m = _re.search(r'([+\d(][\d\s().-]{8,})', parsed_fields["phone"])
        if ph_m:
            customer_phone = ph_m.group(1).strip()
    if not customer_phone:
        ph_m = _re.search(r'Phone:\s*([(\d][\d\s().-]{8,})', body, _re.I)
        if ph_m:
            customer_phone = ph_m.group(1).strip()

    # Extract city/province
    if parsed_fields.get("project_location_city_province"):
        parts = [part.strip() for part in parsed_fields["project_location_city_province"].split(",") if part.strip()]
        if parts:
            customer_city = parts[0]
        if len(parts) > 1:
            customer_province = parts[1]
    if not customer_city and parsed_fields.get("project_location_city"):
        customer_city = parsed_fields["project_location_city"].strip()
    if not customer_city and parsed_fields.get("city"):
        customer_city = parsed_fields["city"].strip()
    if not customer_province and parsed_fields.get("project_location_province"):
        customer_province = parsed_fields["project_location_province"].strip()
    if not customer_province and parsed_fields.get("province"):
        customer_province = parsed_fields["province"].strip()

    # Extract project type
    if parsed_fields.get("project_type"):
        customer_project_type = parsed_fields["project_type"].strip()
    if not customer_project_type:
        proj_m = _re.search(r'Type of Project:\s*([^\n]+?)(?=\s+\b(?:Project|Comments|Email|Phone)\b\s*:|$)', body, _re.I)
        if proj_m:
            customer_project_type = proj_m.group(1).strip()

    # Extract company
    if parsed_fields.get("company_name"):
        val = parsed_fields["company_name"].strip()
        if val and val.lower() not in ("n/a", "na", "none", ""):
            customer_company = val

    if not submission_source:
        submission_source_m = _re.search(r'(?i)submission\s+source\s*:\s*([^\s]+)', body)
        if submission_source_m:
            submission_source = submission_source_m.group(1).strip()
    if not submission_host:
        submission_host_m = _re.search(r'(?i)submission\s+host\s*:\s*([^\s]+)', body)
        if submission_host_m:
            submission_host = submission_host_m.group(1).strip()
    if submission_source and not submission_host:
        submission_host = _extract_host_from_urlish(submission_source)
    elif submission_host:
        submission_host = _extract_host_from_urlish(submission_host)
    submission_source = _extract_first_urlish(submission_source)

    if customer_province:
        customer_province = customer_province.replace(".", "").strip()
        if len(customer_province) <= 3:
            customer_province = customer_province.upper()

    # Must have at least a name AND email to create a lead
    if not customer_email or not customer_first:
        return None

    # Extract dealer info from recipients (who Colin is forwarding to)
    recipients_raw = email_record.get("recipient_emails", "[]")
    try:
        recipients = json.loads(recipients_raw) if isinstance(recipients_raw, str) else recipients_raw
    except json.JSONDecodeError:
        recipients = []
    dealer_email = None
    for r in recipients:
        if r and r.lower().strip() != mailbox_email and "windowfilmcanada" not in r.lower():
            dealer_email = r.lower().strip()
            break

    return {
        "full_name": f"{customer_first or ''} {customer_last or ''}".strip(),
        "first_name": customer_first,
        "last_name": customer_last,
        "email": customer_email,
        "phone": customer_phone or "",
        "city": customer_city or "",
        "province": customer_province or "",
        "company_name": customer_company or "",
        "project_type": customer_project_type or "",
        "dealer_email": dealer_email or "",
        "lead_source": "Email Auto-Created",
        "landing_page": submission_host or "",
        "landing_page_url": submission_source or "",
        "comments": comments or "",
    }


def _insert_email_candidate_lead(candidate: dict, email_record: dict, leads_cache: dict, created_by: str = "") -> Optional[int]:
    """Create a lead from extracted email fields and refresh the in-memory cache."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO leads (
                    name, first_name, last_name, email, phone,
                    city, province, company_name, project_type,
                    dealer_email, lead_source, source, status,
                    email_match_count, last_email_activity, form_submit_date,
                    landing_page, landing_page_url, comments
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    candidate["full_name"],
                    candidate["first_name"],
                    candidate["last_name"],
                    candidate["email"],
                    candidate["phone"],
                    candidate["city"],
                    candidate["province"],
                    candidate["company_name"],
                    candidate["project_type"],
                    candidate["dealer_email"],
                    candidate["lead_source"],
                    candidate["lead_source"],
                    "active",
                    0,
                    email_record.get("received_at"),
                    email_record.get("received_at"),
                    candidate.get("landing_page", ""),
                    candidate.get("landing_page_url", ""),
                    candidate.get("comments", ""),
                ),
            )
            row = cursor.fetchone()
            if not row:
                conn.rollback()
                return None

            new_id = row["id"]
            cursor.execute(
                """INSERT INTO lead_logs (lead_id, log_type, message, created_by, created_at)
                   VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                (
                    new_id,
                    "email_intel",
                    f"Lead created from email candidate: {email_record.get('subject', '')[:200]}",
                    created_by or "email_intel",
                ),
            )
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if candidate["email"]:
        leads_cache["by_email"][candidate["email"]] = new_id
    if candidate["first_name"] and candidate["last_name"]:
        name_key = f"{candidate['first_name'].lower()} {candidate['last_name'].lower()}"
        leads_cache["by_name"][name_key] = new_id

    return new_id


def _enrich_existing_lead_from_candidate(lead_id: int, candidate: dict):
    """Fill in missing lead metadata from an email-derived candidate payload."""
    execute_query(
        """
        UPDATE leads
        SET phone = CASE WHEN COALESCE(NULLIF(TRIM(phone), ''), '') = '' THEN %s ELSE phone END,
            city = CASE WHEN COALESCE(NULLIF(TRIM(city), ''), '') = '' THEN %s ELSE city END,
            province = CASE WHEN COALESCE(NULLIF(TRIM(province), ''), '') = '' THEN %s ELSE province END,
            company_name = CASE WHEN COALESCE(NULLIF(TRIM(company_name), ''), '') = '' THEN %s ELSE company_name END,
            project_type = CASE WHEN COALESCE(NULLIF(TRIM(project_type), ''), '') = '' THEN %s ELSE project_type END,
            dealer_email = CASE WHEN COALESCE(NULLIF(TRIM(dealer_email), ''), '') = '' THEN %s ELSE dealer_email END,
            landing_page = CASE WHEN COALESCE(NULLIF(TRIM(landing_page), ''), '') = '' THEN %s ELSE landing_page END,
            landing_page_url = CASE WHEN COALESCE(NULLIF(TRIM(landing_page_url), ''), '') = '' THEN %s ELSE landing_page_url END,
            comments = CASE WHEN COALESCE(NULLIF(TRIM(comments), ''), '') = '' THEN %s ELSE comments END
        WHERE id = %s
        """,
        (
            candidate.get("phone", ""),
            candidate.get("city", ""),
            candidate.get("province", ""),
            candidate.get("company_name", ""),
            candidate.get("project_type", ""),
            candidate.get("dealer_email", ""),
            candidate.get("landing_page", ""),
            candidate.get("landing_page_url", ""),
            candidate.get("comments", ""),
            lead_id,
        ),
        fetch=False,
    )


def _refresh_lead_email_rollup(lead_id: int):
    """Recompute email rollup fields from matched email rows."""
    stats_rows = execute_query(
        """SELECT COUNT(*) AS cnt, MAX(received_at) AS last_email_at
           FROM email_messages
           WHERE matched_lead_id = %s""",
        (lead_id,),
    )
    latest_sentiment_rows = execute_query(
        """SELECT ai_sentiment
           FROM email_messages
           WHERE matched_lead_id = %s
             AND ai_sentiment IS NOT NULL
           ORDER BY received_at DESC NULLS LAST, id DESC
           LIMIT 1""",
        (lead_id,),
    )
    stats = stats_rows[0] if stats_rows else {"cnt": 0, "last_email_at": None}
    latest_sentiment = latest_sentiment_rows[0]["ai_sentiment"] if latest_sentiment_rows else None
    execute_query(
        """UPDATE leads
           SET email_match_count = %s,
               last_email_activity = %s,
               email_sentiment = COALESCE(%s, email_sentiment)
           WHERE id = %s""",
        (stats.get("cnt", 0), stats.get("last_email_at"), latest_sentiment, lead_id),
        fetch=False,
    )


def _attach_email_candidate_to_lead(email_row: dict, lead_id: int, method: str, confidence: float) -> int:
    """Attach a candidate email and any unmatched thread siblings to a lead."""
    conversation_id = (email_row.get("conversation_id") or "").strip()
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if conversation_id:
                cursor.execute(
                    """UPDATE email_messages
                       SET matched_lead_id = %s,
                           match_confidence = CASE WHEN id = %s THEN %s ELSE 0.80 END,
                           match_method = CASE WHEN id = %s THEN %s ELSE 'thread' END
                       WHERE matched_lead_id IS NULL
                         AND (id = %s OR conversation_id = %s)
                       RETURNING id""",
                    (lead_id, email_row["id"], confidence, email_row["id"], method, email_row["id"], conversation_id),
                )
            else:
                cursor.execute(
                    """UPDATE email_messages
                       SET matched_lead_id = %s,
                           match_confidence = %s,
                           match_method = %s
                       WHERE matched_lead_id IS NULL
                         AND id = %s
                       RETURNING id""",
                    (lead_id, confidence, method, email_row["id"]),
                )
            updated = cursor.fetchall()
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    matched_count = len(updated or [])
    _refresh_lead_email_rollup(lead_id)
    return matched_count


def _try_auto_create_lead(email_record: dict, mailbox_email: str, leads_cache: dict) -> tuple:
    """
    If an email contains customer info from a form submission (Name, Email, Phone)
    and no matching lead exists, auto-create the lead.

    This handles the case where Colin forwards form submissions to dealers
    before the lead is created in Lasso/the CRM.

    Returns: (lead_id, confidence, method) or (None, 0, None)
    """
    candidate = _extract_candidate_lead_fields(email_record, mailbox_email)
    if not candidate:
        return (None, 0, None)

    # Don't create if this email already exists in leads cache
    customer_email = candidate["email"]
    if customer_email and customer_email in leads_cache["by_email"]:
        # Already exists — return the match
        return (leads_cache["by_email"][customer_email], 0.90, "body_email")

    # Don't create if this name already exists
    if candidate["first_name"] and candidate["last_name"]:
        name_key = f"{candidate['first_name'].lower()} {candidate['last_name'].lower()}"
        if name_key in leads_cache["by_name"]:
            return (leads_cache["by_name"][name_key], 0.85, "body_name")

    # Create the lead
    new_id = _insert_email_candidate_lead(candidate, email_record, leads_cache)
    if new_id:
        print(
            f"[EMAIL_INTEL] AUTO-CREATED Lead #{new_id}: {candidate['full_name']} ({candidate['email']}) "
            f"from '{email_record.get('subject', '')[:50]}'",
            flush=True,
        )
        return (new_id, 0.95, "auto_created")

    return (None, 0, None)


def match_email_to_lead(email_record: dict, mailbox_email: str = "", leads_cache: dict | None = None):
    """
    Match the OTHER party in the email to a lead.
    The connected mailbox owner and all their lead records are EXCLUDED.

    For inbound emails: match the SENDER to a lead.
    For outbound emails: match the RECIPIENT(S) to a lead.

    Uses pre-loaded leads_cache for O(1) lookups — NO database calls.

    Returns: (lead_id, confidence, method) or (None, 0, None)
    """
    if not leads_cache:
        return (None, 0, None)

    mailbox_email = mailbox_email.lower().strip()
    mailbox_domain = mailbox_email.split("@")[-1] if mailbox_email else ""

    sender = (email_record.get("sender_email") or "").lower().strip()
    sender_name = (email_record.get("sender_name") or "").lower().strip()
    direction = email_record.get("direction", "inbound")
    body = _normalize_email_text(
        f"{email_record.get('body_text') or ''} {email_record.get('body_preview') or ''}"
    )
    recipient_raw = email_record.get("recipient_emails") or "[]"
    try:
        recipients = json.loads(recipient_raw) if isinstance(recipient_raw, str) else recipient_raw
    except json.JSONDecodeError:
        recipients = []

    # Determine which emails belong to the "other party" (not the mailbox owner)
    if direction == "outbound":
        match_emails = [r.lower().strip() for r in recipients if r and r.lower().strip() != mailbox_email]
    else:
        if sender and sender != mailbox_email:
            match_emails = [sender]
        else:
            match_emails = [r.lower().strip() for r in recipients if r and r.lower().strip() != mailbox_email]

    # Filter out same-domain emails (internal company emails aren't customer leads)
    if mailbox_domain:
        external_emails = [e for e in match_emails if not e.endswith(f"@{mailbox_domain}")]
        if external_emails:
            match_emails = external_emails

    by_email = leads_cache["by_email"]
    by_dealer_email = leads_cache["by_dealer_email"]
    by_name = leads_cache["by_name"]
    phones = leads_cache["phones"]

    # --- 1. Exact email match ---
    for em in match_emails:
        if em in by_email:
            return (by_email[em], 1.0, "email")
    for em in match_emails:
        if em in by_dealer_email:
            return (by_dealer_email[em], 0.95, "dealer_email")

    # --- 2. Name match (for inbound from external sender) ---
    if direction == "inbound" and sender_name and sender != mailbox_email:
        parts = sender_name.split()
        if len(parts) >= 2:
            name_key = f"{parts[0]} {parts[-1]}"
            if name_key in by_name:
                return (by_name[name_key], 0.85, "name")

    # --- 3. Email-in-body match (lead assignment emails contain customer email in body) ---
    body_emails = _re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', body)
    body_emails = [
        e.lower() for e in body_emails
        if e.lower() != mailbox_email
        and not e.lower().endswith(f"@{mailbox_domain}" if mailbox_domain else "@impossible")
        and "noreply" not in e.lower()
        and "no-reply" not in e.lower()
        and "squarespace" not in e.lower()
    ]
    for em in body_emails:
        if em in by_email:
            return (by_email[em], 0.90, "body_email")

    # --- 4. Name-in-body match (form submissions contain "Name: First Last") ---
    name_match = _re.search(r'Name:\s*([A-Z][a-z]+)\s+([A-Z][a-zA-Z\'\-]+)', body)
    if name_match:
        body_first = name_match.group(1).lower()
        body_last = name_match.group(2).lower()
        name_key = f"{body_first} {body_last}"
        if name_key in by_name:
            return (by_name[name_key], 0.85, "body_name")

    # --- 5. Phone match (strict: 10+ digit real phone numbers only) ---
    if body:
        normalized_body = _normalize_phone(body)
        for lead_id, phone in phones:
            if phone in normalized_body:
                return (lead_id, 0.75, "phone")

    # No match found
    return (None, 0, None)


# ---------------------------------------------------------------------------
# Closure candidate detection
# ---------------------------------------------------------------------------

async def check_closure_candidates():
    """
    After sync, check for leads that should be flagged for closure.

    Criteria:
    1. Lead has matched emails
    2. Last email indicates deal completion (positive sentiment + closing stage)
    3. Lead hasn't been updated in X days (default 7)
    4. Lead is currently 'active' status
    5. AI confirms the deal appears complete

    Inserts into closure_review_queue -- NEVER auto-closes.
    Returns number of newly flagged leads.
    """
    flagged = 0
    candidates = execute_query("""
        SELECT l.id AS lead_id, l.first_name, l.last_name, l.email, l.status,
               l.last_email_activity, l.email_match_count,
               MAX(em.received_at) AS last_email_at,
               COUNT(em.id) AS email_count
        FROM leads l
        JOIN email_messages em ON em.matched_lead_id = l.id
        WHERE l.status = 'active'
          AND l.ai_closure_flagged = FALSE
          AND l.last_email_activity < CURRENT_TIMESTAMP - INTERVAL '7 days'
        GROUP BY l.id
        HAVING COUNT(em.id) >= 1
    """)

    if not candidates:
        return 0

    for cand in candidates:
        # Look at latest email sentiment / stage
        latest = execute_query(
            "SELECT ai_sentiment, ai_summary, ai_ready_to_close, ai_close_reasoning "
            "FROM email_messages WHERE matched_lead_id = %s "
            "ORDER BY received_at DESC LIMIT 3",
            (cand["lead_id"],),
        )
        if not latest:
            continue

        sentiments = [e.get("ai_sentiment") for e in latest]
        any_ready = any(e.get("ai_ready_to_close") for e in latest)

        if not any_ready and "positive" not in sentiments:
            continue

        reasoning = "Positive email sentiment with extended inactivity."
        summaries = "\n".join(
            f"- {e.get('ai_summary', 'No summary')}" for e in latest
        )
        ai_result, model_used = _reasoning_json_completion(
            system_prompt=(
                "You are a sales-ops assistant. Based on recent email summaries "
                "for a lead, determine if the deal appears complete and the lead "
                "can be closed. Return JSON: "
                '{"should_flag": true/false, "reasoning": "..."}'
            ),
            user_prompt=(
                f"Lead: {cand.get('first_name','')} {cand.get('last_name','')}\n"
                f"Status: {cand.get('status','')}\n"
                f"Days since last email: "
                f"{(datetime.utcnow() - cand['last_email_at']).days if cand.get('last_email_at') else '?'}\n"
                f"Recent email summaries:\n{summaries}"
            ),
        )
        if ai_result is not None:
            if not ai_result.get("should_flag", False):
                continue
            reasoning = ai_result.get("reasoning", reasoning)
            logger.info("Closure review reasoning completed with model %s", model_used)

        days_inactive = 0
        if cand.get("last_email_at"):
            days_inactive = (datetime.utcnow() - cand["last_email_at"]).days

        # Check not already in queue
        existing = execute_query(
            "SELECT id FROM closure_review_queue WHERE lead_id = %s AND status = 'pending'",
            (cand["lead_id"],),
        )
        if existing:
            continue

        execute_query(
            "INSERT INTO closure_review_queue "
            "(lead_id, ai_reasoning, days_inactive, last_email_at, email_count) "
            "VALUES (%s, %s, %s, %s, %s)",
            (cand["lead_id"], reasoning, days_inactive, cand.get("last_email_at"), cand.get("email_count", 0)),
            fetch=False,
        )
        execute_query(
            "UPDATE leads SET ai_closure_flagged = TRUE WHERE id = %s",
            (cand["lead_id"],),
            fetch=False,
        )
        flagged += 1

    return flagged


# ---------------------------------------------------------------------------
# Token refresh helper
# ---------------------------------------------------------------------------

def _refresh_access_token(config: dict) -> str:
    """Refresh the MS Graph access token if expired. Returns valid access token."""
    expires = config.get("token_expires_at")
    if expires and datetime.utcnow() < expires:
        return config["access_token"]

    resp = http_requests.post(
        f"{MS_LOGIN_BASE}/{config['ms_tenant_id']}/oauth2/v2.0/token",
        data={
            "client_id": config["ms_client_id"],
            "client_secret": config["ms_client_secret"],
            "refresh_token": config["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Token refresh failed: {resp.text}")

    data = resp.json()
    new_expires = datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600))
    _upsert_sync_config(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", config["refresh_token"]),
        token_expires_at=new_expires,
    )
    return data["access_token"]


# ---------------------------------------------------------------------------
# Graph API helper
# ---------------------------------------------------------------------------

def _graph_get(token: str, url: str) -> dict:
    resp = http_requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code != 200:
        error_code = ""
        error_message = ""
        try:
            payload = resp.json()
            error = payload.get("error") or {}
            error_code = (error.get("code") or "").strip()
            error_message = (error.get("message") or "").strip()
        except Exception:
            payload = None

        logger.error(f"Graph API error {resp.status_code}: {resp.text[:300]}")

        mailbox_hint = ""
        marker = "/users/"
        if marker in url:
            mailbox_hint = unquote(url.split(marker, 1)[1].split("/", 1)[0]).strip()

        if error_code == "ErrorGroupIsUsedInNonGroupURI":
            target = mailbox_hint or "the configured target mailbox"
            raise RuntimeError(
                f"{target} is being treated by Microsoft Graph as a Microsoft 365 Group, not a shared/user mailbox. "
                "This sync path uses /users/{mailbox}/messages, so it cannot read that address directly. "
                "Use Colin's mailbox as the sync target, or add Group mailbox support with the required Graph Group permissions."
            )

        detail = f"Graph API error {resp.status_code}"
        if error_code:
            detail += f" ({error_code})"
        if error_message:
            detail += f": {error_message}"
        raise RuntimeError(detail)
    return resp.json()


# ===================================================================
# ENDPOINTS
# ===================================================================


# ---------------------------------------------------------------------------
# OAuth & Config
# ---------------------------------------------------------------------------

@router.get("/config")
async def get_config(current_user: AdminUser = Depends(get_current_user)):
    """Return current email sync config (without secrets)."""
    config = _get_sync_config()
    if not config:
        return {"configured": False}

    is_connected = bool(config.get("access_token")) and bool(config.get("refresh_token"))

    return {
        "id": config.get("id"),
        "configured": True,
        "ms_tenant_id": config.get("ms_tenant_id"),
        "ms_client_id": config.get("ms_client_id"),
        "ms_redirect_uri": config.get("ms_redirect_uri"),
        "shared_mailbox_email": config.get("shared_mailbox_email"),
        "target_mailbox_type": _get_target_mailbox_type(config),
        "target_mailbox_email": _get_target_mailbox_email(config) or config.get("user_email"),
        "target_group_id": config.get("target_group_id"),
        "sync_enabled": config.get("sync_enabled", False),
        "sync_interval_minutes": config.get("sync_interval_minutes", 15),
        "last_sync_at": config.get("last_sync_at"),
        "user_email": config.get("user_email"),
        "is_connected": is_connected,
    }


@router.post("/config")
async def save_config(
    body: OAuthConfigIn,
    current_user: AdminUser = Depends(get_current_user),
):
    """Save / update MS Graph OAuth config."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return {"success": True}

    if "target_mailbox_type" in updates:
        mailbox_type = (updates.get("target_mailbox_type") or "").strip().lower()
        if mailbox_type not in VALID_TARGET_MAILBOX_TYPES:
            raise HTTPException(status_code=400, detail="Invalid target_mailbox_type")
        updates["target_mailbox_type"] = mailbox_type

    if "shared_mailbox_email" in updates:
        normalized_mailbox = (updates.get("shared_mailbox_email") or "").strip().lower()
        updates["shared_mailbox_email"] = normalized_mailbox or None

    existing = _get_sync_config()
    if not existing:
        required_fields = ("ms_tenant_id", "ms_client_id", "ms_client_secret", "ms_redirect_uri")
        missing = [field for field in required_fields if not updates.get(field)]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required config fields: {', '.join(missing)}")

    effective_type = updates.get("target_mailbox_type") or _get_target_mailbox_type(existing)
    effective_shared = updates.get("shared_mailbox_email")
    if effective_shared is None and existing:
        effective_shared = (existing.get("shared_mailbox_email") or "").strip().lower() or None
    if effective_type in {MAILBOX_TYPE_SHARED, MAILBOX_TYPE_GROUP} and not effective_shared:
        raise HTTPException(status_code=400, detail="A shared/group mailbox email is required for this mailbox type.")

    if existing:
        previous_mailbox = (existing.get("shared_mailbox_email") or "").strip().lower()
        new_mailbox = (updates.get("shared_mailbox_email") or previous_mailbox).strip().lower()
        previous_type = _get_target_mailbox_type(existing)
        new_type = updates.get("target_mailbox_type") or previous_type
        if new_mailbox != previous_mailbox or new_type != previous_type:
            updates["target_group_id"] = None

    _upsert_sync_config(**updates)
    audit_log("CONFIG_UPDATED", "OAuth config saved/updated", current_user.username)
    return {"success": True}


@router.get("/oauth/authorize")
async def oauth_authorize(current_user: AdminUser = Depends(get_current_user)):
    """Build and return the Microsoft OAuth2 authorisation URL."""
    config = _get_sync_config()
    if not config or not config.get("ms_tenant_id"):
        raise HTTPException(status_code=400, detail="OAuth config not set. Save config first.")

    params = {
        "client_id": config["ms_client_id"],
        "response_type": "code",
        "redirect_uri": config["ms_redirect_uri"],
        "response_mode": "query",
        "scope": OAUTH_SCOPES,
        "prompt": "select_account",
    }
    if config.get("user_email"):
        params["login_hint"] = config["user_email"]
    auth_url = (
        f"{MS_LOGIN_BASE}/{config['ms_tenant_id']}/oauth2/v2.0/authorize?"
        + urlencode(params)
    )
    audit_log("OAUTH_AUTHORIZE", "OAuth authorize URL generated", current_user.username)
    return {"auth_url": auth_url}


@router.post("/oauth/callback")
async def oauth_callback(
    body: OAuthCallbackIn,
    current_user: AdminUser = Depends(get_current_user),
):
    """Exchange authorisation code for tokens (called from frontend)."""
    result = _exchange_code_for_tokens(body.code)
    audit_log("OAUTH_CONNECT", f"OAuth connected for {result['email']}", current_user.username)
    return result


@router.get("/oauth/callback-redirect")
async def oauth_callback_redirect(code: str = None, error: str = None, error_description: str = None):
    """
    Microsoft redirects here after user grants consent.
    This is a GET endpoint — no auth required since it's a browser redirect.
    Exchanges the code for tokens, then redirects back to the frontend.
    """
    from fastapi.responses import RedirectResponse

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3001") + "/admin/email-intel"

    if error:
        logger.error(f"OAuth error: {error} - {error_description}")
        return RedirectResponse(url=f"{frontend_url}?oauth_error={error}")

    if not code:
        return RedirectResponse(url=f"{frontend_url}?oauth_error=no_code")

    try:
        result = _exchange_code_for_tokens(code)
        audit_log("OAUTH_CONNECT", f"OAuth connected for {result['email']}", "oauth-redirect")
        return RedirectResponse(url=f"{frontend_url}?oauth_success=true&email={result['email']}")
    except Exception as e:
        logger.error(f"OAuth token exchange failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"{frontend_url}?oauth_error={quote(str(e)[:200])}")


def _exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access/refresh tokens."""
    config = _get_sync_config_decrypted()
    if not config:
        raise HTTPException(status_code=400, detail="OAuth config not set.")

    token_url = f"{MS_LOGIN_BASE}/{config['ms_tenant_id']}/oauth2/v2.0/token"
    payload = {
        "client_id": config["ms_client_id"],
        "client_secret": config["ms_client_secret"],
        "code": code,
        "redirect_uri": config["ms_redirect_uri"],
        "grant_type": "authorization_code",
        "scope": OAUTH_SCOPES,
    }

    logger.info("Exchanging Microsoft OAuth authorization code for Graph access token")
    from urllib.parse import urlencode
    encoded_body = urlencode(payload)

    resp = http_requests.post(
        token_url,
        data=encoded_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise Exception(f"Token exchange failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    expires_at = datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600))
    token_email = _infer_user_email_from_token(data.get("access_token", ""))

    # Fetch user profile to get email
    profile_resp = http_requests.get(
        f"{GRAPH_BASE}/me",
        headers={"Authorization": f"Bearer {data['access_token']}"},
        timeout=15,
    )
    if profile_resp.status_code == 200:
        profile = profile_resp.json()
        user_email = (
            (profile.get("mail") or "").strip()
            or (profile.get("userPrincipalName") or "").strip()
            or token_email
        ).lower()
    else:
        actual_identity = token_email or "the selected Microsoft account"
        logger.error("Graph /me lookup failed during OAuth connect: %s", profile_resp.text[:300])
        raise Exception(
            "Connected account is not a usable Microsoft 365 mailbox user context for Graph. "
            f"Signed in as {actual_identity}. Reconnect as Colin's Microsoft 365 work account "
            "(cmacleod@windowfilmcanada.ca), not a guest or personal account."
        )

    if not user_email:
        raise Exception(
            "OAuth completed but Microsoft Graph did not return a mailbox identity. "
            "Reconnect as Colin's Microsoft 365 work account (cmacleod@windowfilmcanada.ca)."
        )

    _upsert_sync_config(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        token_expires_at=expires_at,
        sync_enabled=True,
        user_email=user_email,
    )
    return {"success": True, "email": user_email}


# ---------------------------------------------------------------------------
# Emergency Disconnect (Kill Switch)
# ---------------------------------------------------------------------------

@router.post("/emergency-disconnect")
async def emergency_disconnect(current_user: AdminUser = Depends(get_current_user)):
    """Instant kill switch -- wipes all tokens, disables sync, logs action."""
    execute_query("UPDATE email_sync_config SET access_token = NULL, refresh_token = NULL, sync_enabled = FALSE, token_expires_at = NULL", fetch=False)
    audit_log("EMERGENCY_DISCONNECT", "All email tokens wiped and sync disabled", current_user.username)
    return {"success": True, "message": "All email access revoked immediately"}


# ---------------------------------------------------------------------------
# Email Sync
# ---------------------------------------------------------------------------

@router.post("/sync")
async def sync_emails(current_user: AdminUser = Depends(get_current_user)):
    """Trigger an email sync from MS Graph in the background."""
    global _last_sync_time
    current_state = _snapshot_sync_state()
    if current_state["is_running"]:
        return {
            "started": False,
            "sync_in_progress": True,
            "message": "Email sync is already running.",
            "result": current_state.get("last_result"),
        }

    now = datetime.utcnow()
    if _last_sync_time and (now - _last_sync_time).total_seconds() < 30:
        raise HTTPException(status_code=429, detail="Sync rate limited. Wait 30 seconds between syncs.")

    config = _get_sync_config_decrypted()
    if not config or not config.get("access_token"):
        raise HTTPException(status_code=400, detail="OAuth not configured or not authorised.")

    _last_sync_time = now
    _update_sync_state(
        is_running=True,
        started_at=now,
        finished_at=None,
        current_phase="Starting sync",
        last_error=None,
        last_result=None,
        synced=0,
        matched=0,
        flagged_for_review=0,
    )

    def _run_sync_in_background(started_by: str):
        asyncio.run(_perform_sync_job(started_by))

    threading.Thread(
        target=_run_sync_in_background,
        args=(current_user.username,),
        daemon=True,
    ).start()

    audit_log("EMAIL_SYNC_STARTED", "Email sync started in background", current_user.username)

    return {
        "started": True,
        "sync_in_progress": True,
        "message": "Email sync started in background.",
    }


def _propagate_thread_matches() -> int:
    """
    For every conversation that has at least one matched email,
    propagate the lead match to all unmatched emails in the same conversation.
    This connects dealer replies, follow-ups, and forwarded chains to the lead.
    """
    propagated = 0

    # Find conversations where at least one email is matched
    matched_threads = execute_query("""
        SELECT DISTINCT conversation_id, matched_lead_id
        FROM email_messages
        WHERE conversation_id IS NOT NULL AND conversation_id != ''
          AND matched_lead_id IS NOT NULL
    """)
    if not matched_threads:
        return 0

    for thread in matched_threads:
        conv_id = thread["conversation_id"]
        lead_id = thread["matched_lead_id"]

        # Update all unmatched emails in this conversation
        result = execute_query(
            """UPDATE email_messages
               SET matched_lead_id = %s, match_confidence = 0.80, match_method = 'thread'
               WHERE conversation_id = %s
                 AND (matched_lead_id IS NULL)
               RETURNING id""",
            (lead_id, conv_id),
        )
        count = len(result) if result else 0
        if count > 0:
            propagated += count
            # Update lead email counters
            execute_query(
                "UPDATE leads SET email_match_count = email_match_count + %s WHERE id = %s",
                (count, lead_id),
                fetch=False,
            )

    return propagated


async def _perform_sync_job(started_by: str):
    try:
        print("[EMAIL_INTEL] Sync job starting...", flush=True)
        config = _get_sync_config_decrypted()
        if not config or not config.get("access_token"):
            raise RuntimeError("OAuth not configured or not authorised.")

        print("[EMAIL_INTEL] Config loaded, refreshing token...", flush=True)
        _update_sync_state(current_phase="Refreshing access token")
        token = _refresh_access_token(config)
        mailbox_type = _get_target_mailbox_type(config)
        mailbox_email = _get_target_mailbox_email(config)
        matching_mailbox_email = _get_matching_mailbox_email(config)
        print(
            f"[EMAIL_INTEL] Token ready, starting {mailbox_type} sync for {mailbox_email or 'connected mailbox'}...",
            flush=True,
        )

        synced = 0
        matched = 0
        history_limit = EMAIL_SYNC_HISTORY_LIMIT

        if mailbox_type == MAILBOX_TYPE_GROUP:
            group = _resolve_group_target(token, config)
            group_label = group.get("displayName") or group.get("mail") or mailbox_email or "configured group"
            _update_sync_state(current_phase=f"Syncing Microsoft 365 Group ({group_label})")
            synced_group, matched_group = await _sync_group_threads(
                token,
                group,
                config=config,
                max_messages=history_limit,
            )
            synced += synced_group
            matched += matched_group
            _update_sync_state(synced=synced, matched=matched)
        else:
            mailbox_base = _graph_mailbox_base(config)

            _update_sync_state(current_phase="Syncing inbox")
            url = (
                f"{mailbox_base}/messages?"
                f"$top={GRAPH_SYNC_PAGE_SIZE}&$orderby=receivedDateTime desc"
                "&$select=id,subject,from,toRecipients,body,bodyPreview,receivedDateTime,isRead,conversationId"
            )
            synced_inbound, matched_inbound = await _sync_messages(
                token,
                url,
                direction="inbound",
                mailbox_email=matching_mailbox_email,
                max_messages=history_limit,
            )
            synced += synced_inbound
            matched += matched_inbound
            _update_sync_state(synced=synced, matched=matched)

            _update_sync_state(current_phase="Syncing sent mail")
            url_sent = (
                f"{mailbox_base}/mailFolders/SentItems/messages?"
                f"$top={GRAPH_SYNC_PAGE_SIZE}&$orderby=receivedDateTime desc"
                "&$select=id,subject,from,toRecipients,body,bodyPreview,receivedDateTime,isRead,conversationId"
            )
            synced_outbound, matched_outbound = await _sync_messages(
                token,
                url_sent,
                direction="outbound",
                mailbox_email=matching_mailbox_email,
                max_messages=history_limit,
            )
            synced += synced_outbound
            matched += matched_outbound
            _update_sync_state(synced=synced, matched=matched)

        # --- Thread propagation: if any email in a conversation matched,
        #     propagate that match to ALL emails in the same conversation ---
        _update_sync_state(current_phase="Propagating thread matches")
        thread_matched = _propagate_thread_matches()
        matched += thread_matched
        _update_sync_state(matched=matched)
        print(f"[EMAIL_INTEL] Thread propagation: {thread_matched} additional matches", flush=True)

        _upsert_sync_config(last_sync_at=datetime.utcnow())

        _update_sync_state(current_phase="Evaluating closure candidates")
        flagged = await check_closure_candidates()

        result = {
            "synced": synced,
            "matched": matched,
            "flagged_for_review": flagged,
        }
        log_event(
            event_type="MS_GRAPH_EMAIL_SYNC",
            entity_type="email_sync",
            actor=started_by,
            payload={
                "synced": synced,
                "matched": matched,
                "unmatched_estimate": max(synced - matched, 0),
                "flagged_for_review": flagged,
                "mailbox_type": mailbox_type,
                "mailbox_email": mailbox_email,
            },
        )

        _update_sync_state(
            is_running=False,
            finished_at=datetime.utcnow(),
            current_phase="Completed",
            last_error=None,
            last_result=result,
            synced=synced,
            matched=matched,
            flagged_for_review=flagged,
        )
        audit_log("EMAIL_SYNC", f"Synced {synced} emails, matched {matched}, flagged {flagged}", started_by)
    except Exception as exc:
        import traceback
        error_message = str(exc).strip() or exc.__class__.__name__
        print(f"[EMAIL_INTEL] SYNC FAILED: {exc}")
        traceback.print_exc()
        logger.exception("Email sync job failed")
        log_event(
            event_type="MS_GRAPH_EMAIL_SYNC_FAILED",
            entity_type="email_sync",
            actor=started_by,
            payload={"error": error_message},
        )
        _update_sync_state(
            is_running=False,
            finished_at=datetime.utcnow(),
            current_phase="Failed",
            last_error=error_message,
        )
        audit_log("EMAIL_SYNC_FAILED", error_message, started_by)


async def _sync_messages(
    token: str,
    url: str,
    direction: str = "inbound",
    mailbox_email: str = "",
    max_messages: int = EMAIL_SYNC_HISTORY_LIMIT,
):
    """
    Paginate through Graph messages and sync them.
    Fast: parallel AI calls in batches, skips AI for unmatched emails.
    """
    synced = 0
    matched = 0
    scanned = 0

    mailbox_email = mailbox_email.lower().strip()

    # Pre-load existing message IDs to skip duplicates instantly
    existing_ids_rows = execute_query("SELECT ms_message_id FROM email_messages")
    existing_ids = {r["ms_message_id"] for r in (existing_ids_rows or [])}

    # Pre-load ALL leads into memory for fast matching (1 DB query instead of thousands)
    leads_cache = _load_leads_cache(mailbox_email)

    page_num = 0
    while url and scanned < max_messages:
        page_num += 1
        print(f"[EMAIL_INTEL] Fetching page {page_num} from Graph API...", flush=True)
        data = _graph_get(token, url)
        messages = data.get("value", [])
        remaining_budget = max_messages - scanned
        if remaining_budget <= 0:
            break
        if len(messages) > remaining_budget:
            messages = messages[:remaining_budget]
        scanned += len(messages)
        print(
            f"[EMAIL_INTEL] Page {page_num}: got {len(messages)} messages "
            f"(scanned {scanned}/{max_messages} for {direction})",
            flush=True,
        )
        if not messages:
            break

        # --- Phase 1: Parse all messages in this page & match leads (fast, no AI) ---
        print(f"[EMAIL_INTEL] Page {page_num}: Phase 1 — matching leads...", flush=True)
        page_records = []
        skipped = 0
        for msg_idx, msg in enumerate(messages):
            if msg_idx % 100 == 0 and msg_idx > 0:
                print(f"[EMAIL_INTEL]   ... processed {msg_idx}/{len(messages)} messages", flush=True)
            ms_id = msg.get("id")
            if not ms_id or ms_id in existing_ids:
                skipped += 1
                continue

            from_obj = msg.get("from", {}).get("emailAddress", {})
            sender_email = from_obj.get("address", "")
            sender_name = from_obj.get("name", "")
            recipients = [
                r.get("emailAddress", {}).get("address", "")
                for r in msg.get("toRecipients", [])
            ]
            # Get full body content (HTML), strip tags for text matching
            full_body_html = (msg.get("body", {}) or {}).get("content", "") or ""
            # Strip HTML tags to get plain text for matching
            full_body_text = _re.sub(r'<[^>]+>', ' ', full_body_html)
            full_body_text = _re.sub(r'\s+', ' ', full_body_text).strip()
            graph_body_preview = _normalize_email_text(msg.get("bodyPreview", "") or "", 1200)
            full_body_preview = _normalize_email_text(full_body_text, 1200)
            # Prefer the richer preview when Graph bodyPreview is shorter than the extracted body content.
            body_preview = full_body_preview if len(full_body_preview) > len(graph_body_preview) else graph_body_preview

            # Detect REAL direction: if sender is the mailbox owner, it's outbound
            actual_direction = "outbound" if sender_email.lower().strip() == mailbox_email else "inbound"
            received_at = msg.get("receivedDateTime")

            email_record = {
                "ms_message_id": ms_id,
                "conversation_id": msg.get("conversationId", ""),
                "subject": msg.get("subject", ""),
                "sender_email": sender_email,
                "sender_name": sender_name,
                "recipient_emails": json.dumps(recipients),
                "body_preview": body_preview,
                "body_text": _normalize_email_text(full_body_text, 4000),
                "received_at": received_at,
                "is_read": msg.get("isRead", False),
                "direction": actual_direction,
                "folder": "SentItems" if actual_direction == "outbound" else "Inbox",
            }

            # Match to lead (excludes mailbox owner's lead records) — uses in-memory cache
            lead_id, confidence, method = match_email_to_lead(email_record, mailbox_email, leads_cache)

            # --- Auto-create lead from form submission emails ---
            # If no match found AND the email contains customer info from a form submission,
            # create a new lead so we can track it
            if not lead_id:
                lead_id, confidence, method = _try_auto_create_lead(email_record, mailbox_email, leads_cache)

            email_record["matched_lead_id"] = lead_id
            email_record["match_confidence"] = confidence
            email_record["match_method"] = method
            page_records.append(email_record)

        # --- Phase 2: Run AI analysis in PARALLEL batches (only matched emails) ---
        ai_tasks = []
        ai_indices = []
        for i, rec in enumerate(page_records):
            if rec["matched_lead_id"]:
                lead_rows = execute_query("SELECT first_name, last_name, email, status FROM leads WHERE id = %s", (rec["matched_lead_id"],))
                lead_data = lead_rows[0] if lead_rows else None
                ai_tasks.append(analyze_email_with_ai(
                    email_body=rec["body_preview"],
                    email_subject=rec["subject"],
                    lead_data=lead_data,
                ))
                ai_indices.append(i)

        # Run all AI calls for this page concurrently
        print(f"[EMAIL_INTEL] Page {page_num}: {len(page_records)} new records, {len(ai_tasks)} need AI analysis", flush=True)
        if ai_tasks:
            ai_results = await asyncio.gather(*ai_tasks, return_exceptions=True)
            print(f"[EMAIL_INTEL] Page {page_num}: AI analysis complete", flush=True)
        else:
            ai_results = []

        # Apply AI results
        ai_map = {}
        for idx, ai_idx in enumerate(ai_indices):
            result = ai_results[idx]
            if isinstance(result, Exception):
                logger.error("AI batch call failed: %s", result)
                result = dict(_DEFAULT_AI_RESULT)
            ai_map[ai_idx] = result

        # --- Phase 3: Insert all records into DB ---
        for i, rec in enumerate(page_records):
            ai_result = ai_map.get(i, _DEFAULT_AI_RESULT)
            rec["ai_summary"] = ai_result.get("summary", "")
            rec["ai_sentiment"] = ai_result.get("sentiment", "neutral")
            rec["ai_action_items"] = json.dumps(ai_result.get("action_items", []))
            rec["ai_ready_to_close"] = ai_result.get("deal_stage") in ("closed", "closing")
            rec["ai_close_reasoning"] = f"Deal stage: {ai_result.get('deal_stage', 'unknown')}" if rec["matched_lead_id"] else ""
            rec["processed_at"] = datetime.utcnow()

            # Verify FK: if matched_lead_id set, confirm lead exists
            if rec["matched_lead_id"]:
                exists = execute_query("SELECT id FROM leads WHERE id = %s", (rec["matched_lead_id"],))
                if not exists:
                    print(f"[EMAIL_INTEL] WARNING: Lead #{rec['matched_lead_id']} missing, clearing match for '{rec.get('subject','')[:50]}'", flush=True)
                    rec["matched_lead_id"] = None
                    rec["match_confidence"] = 0
                    rec["match_method"] = None

            try:
                insert_rec = rec
                cols = list(insert_rec.keys())
                placeholders = ", ".join(["%s"] * len(cols))
                execute_query(
                    f"INSERT INTO email_messages ({', '.join(cols)}) VALUES ({placeholders})",
                    tuple(insert_rec[c] for c in cols),
                    fetch=False,
                )
                existing_ids.add(rec["ms_message_id"])
                synced += 1

                if rec["matched_lead_id"]:
                    matched += 1
                    execute_query(
                        "UPDATE leads SET email_match_count = email_match_count + 1, "
                        "last_email_activity = %s, email_sentiment = %s WHERE id = %s",
                        (rec["received_at"], rec["ai_sentiment"], rec["matched_lead_id"]),
                        fetch=False,
                    )
            except Exception as insert_err:
                print(f"[EMAIL_INTEL] WARNING: Failed to insert email '{rec.get('subject','')[:50]}': {insert_err}", flush=True)
                continue

        print(
            f"[EMAIL_INTEL] Page {page_num} complete: total synced={synced}, matched={matched}, "
            f"scanned={scanned}/{max_messages}",
            flush=True,
        )
        _update_sync_state(
            synced=synced,
            matched=matched,
            current_phase=f"Processing {direction} ({scanned}/{max_messages} scanned, {synced} saved)",
        )
        url = data.get("@odata.nextLink") if scanned < max_messages else None

    print(
        f"[EMAIL_INTEL] {direction} sync finished: synced={synced}, matched={matched}, "
        f"scanned={scanned}/{max_messages}",
        flush=True,
    )
    _update_sync_state(synced=synced, matched=matched)
    return synced, matched


async def _sync_group_threads(
    token: str,
    group: dict,
    config: Optional[dict] = None,
    max_messages: int = EMAIL_SYNC_HISTORY_LIMIT,
):
    """
    Sync a Microsoft 365 Group mailbox via group threads/posts.
    We translate posts into the same email_messages shape used by the rest of the app.
    """
    synced = 0
    matched = 0
    scanned = 0

    group_id = (group.get("id") or "").strip()
    if not group_id:
        raise RuntimeError("Microsoft 365 Group sync could not resolve a valid group id.")

    target_mailbox_email = _get_target_mailbox_email(config)
    matching_mailbox_email = _get_matching_mailbox_email(config) or target_mailbox_email

    existing_ids_rows = execute_query("SELECT ms_message_id FROM email_messages")
    existing_ids = {r["ms_message_id"] for r in (existing_ids_rows or [])}
    leads_cache = _load_leads_cache(matching_mailbox_email)

    query = urlencode(
        {
            "$top": 100,
            "$orderby": "lastDeliveredDateTime desc",
            "$select": "id,topic,preview,lastDeliveredDateTime,toRecipients,ccRecipients,uniqueSenders",
            "$expand": "posts($select=id,conversationId,conversationThreadId,receivedDateTime,createdDateTime,body,from,sender)",
        }
    )
    url = f"{GRAPH_BASE}/groups/{group_id}/threads?{query}"

    page_num = 0
    while url and scanned < max_messages:
        page_num += 1
        print(f"[EMAIL_INTEL] Fetching group thread page {page_num} from Graph API...", flush=True)
        data = _graph_get(token, url)
        threads = data.get("value", [])
        print(
            f"[EMAIL_INTEL] Group page {page_num}: got {len(threads)} threads "
            f"(scanned {scanned}/{max_messages} posts so far)",
            flush=True,
        )
        if not threads:
            break

        page_records = []
        skipped = 0
        for thread_idx, thread in enumerate(threads):
            if thread_idx % 50 == 0 and thread_idx > 0:
                print(
                    f"[EMAIL_INTEL]   ... processed {thread_idx}/{len(threads)} group threads",
                    flush=True,
                )

            recipients = _flatten_graph_recipients(
                thread.get("toRecipients"),
                thread.get("ccRecipients"),
            )
            posts = thread.get("posts") or []
            if not posts and thread.get("id"):
                posts = _fetch_group_thread_posts(token, group_id, thread["id"])

            for post in posts:
                if scanned >= max_messages:
                    break

                post_id = (post.get("id") or "").strip()
                if not post_id:
                    continue

                scanned += 1
                ms_id = _group_post_message_id(group_id, post_id)
                if ms_id in existing_ids:
                    skipped += 1
                    continue

                sender_obj = post.get("sender") or post.get("from") or {}
                sender_email = _extract_graph_recipient_email(sender_obj)
                sender_name = _extract_graph_recipient_name(sender_obj)
                body_preview, body_text = _extract_email_text_from_graph_message(
                    {
                        "body": post.get("body") or {},
                        "bodyPreview": thread.get("preview") or "",
                    }
                )
                actual_direction = (
                    "outbound" if _is_group_internal_sender(sender_email, config) else "inbound"
                )
                received_at = post.get("receivedDateTime") or post.get("createdDateTime")

                email_record = {
                    "ms_message_id": ms_id,
                    "conversation_id": post.get("conversationId") or f"group-thread:{group_id}:{thread.get('id', '')}",
                    "subject": thread.get("topic", "") or "",
                    "sender_email": sender_email,
                    "sender_name": sender_name,
                    "recipient_emails": json.dumps(recipients),
                    "body_preview": body_preview,
                    "body_text": body_text,
                    "received_at": received_at,
                    "is_read": True,
                    "direction": actual_direction,
                    "folder": "Group",
                }

                lead_id, confidence, method = match_email_to_lead(
                    email_record,
                    matching_mailbox_email,
                    leads_cache,
                )
                if not lead_id:
                    lead_id, confidence, method = _try_auto_create_lead(
                        email_record,
                        matching_mailbox_email,
                        leads_cache,
                    )

                email_record["matched_lead_id"] = lead_id
                email_record["match_confidence"] = confidence
                email_record["match_method"] = method
                page_records.append(email_record)

        ai_tasks = []
        ai_indices = []
        for i, rec in enumerate(page_records):
            if rec["matched_lead_id"]:
                lead_rows = execute_query(
                    "SELECT first_name, last_name, email, status FROM leads WHERE id = %s",
                    (rec["matched_lead_id"],),
                )
                lead_data = lead_rows[0] if lead_rows else None
                ai_tasks.append(
                    analyze_email_with_ai(
                        email_body=rec["body_preview"],
                        email_subject=rec["subject"],
                        lead_data=lead_data,
                    )
                )
                ai_indices.append(i)

        print(
            f"[EMAIL_INTEL] Group page {page_num}: {len(page_records)} new records, "
            f"{len(ai_tasks)} need AI analysis, {skipped} skipped",
            flush=True,
        )
        if ai_tasks:
            ai_results = await asyncio.gather(*ai_tasks, return_exceptions=True)
            print(f"[EMAIL_INTEL] Group page {page_num}: AI analysis complete", flush=True)
        else:
            ai_results = []

        ai_map = {}
        for idx, ai_idx in enumerate(ai_indices):
            result = ai_results[idx]
            if isinstance(result, Exception):
                logger.error("AI batch call failed: %s", result)
                result = dict(_DEFAULT_AI_RESULT)
            ai_map[ai_idx] = result

        for i, rec in enumerate(page_records):
            ai_result = ai_map.get(i, _DEFAULT_AI_RESULT)
            rec["ai_summary"] = ai_result.get("summary", "")
            rec["ai_sentiment"] = ai_result.get("sentiment", "neutral")
            rec["ai_action_items"] = json.dumps(ai_result.get("action_items", []))
            rec["ai_ready_to_close"] = ai_result.get("deal_stage") in ("closed", "closing")
            rec["ai_close_reasoning"] = (
                f"Deal stage: {ai_result.get('deal_stage', 'unknown')}"
                if rec["matched_lead_id"]
                else ""
            )
            rec["processed_at"] = datetime.utcnow()

            if rec["matched_lead_id"]:
                exists = execute_query("SELECT id FROM leads WHERE id = %s", (rec["matched_lead_id"],))
                if not exists:
                    print(
                        f"[EMAIL_INTEL] WARNING: Lead #{rec['matched_lead_id']} missing, "
                        f"clearing group match for '{rec.get('subject','')[:50]}'",
                        flush=True,
                    )
                    rec["matched_lead_id"] = None
                    rec["match_confidence"] = 0
                    rec["match_method"] = None

            try:
                insert_rec = rec
                cols = list(insert_rec.keys())
                placeholders = ", ".join(["%s"] * len(cols))
                execute_query(
                    f"INSERT INTO email_messages ({', '.join(cols)}) VALUES ({placeholders})",
                    tuple(insert_rec[c] for c in cols),
                    fetch=False,
                )
                existing_ids.add(rec["ms_message_id"])
                synced += 1

                if rec["matched_lead_id"]:
                    matched += 1
                    execute_query(
                        "UPDATE leads SET email_match_count = email_match_count + 1, "
                        "last_email_activity = %s, email_sentiment = %s WHERE id = %s",
                        (rec["received_at"], rec["ai_sentiment"], rec["matched_lead_id"]),
                        fetch=False,
                    )
            except Exception as insert_err:
                print(
                    f"[EMAIL_INTEL] WARNING: Failed to insert group post '{rec.get('subject','')[:50]}': "
                    f"{insert_err}",
                    flush=True,
                )
                continue

        print(
            f"[EMAIL_INTEL] Group page {page_num} complete: total synced={synced}, matched={matched}, "
            f"scanned={scanned}/{max_messages}",
            flush=True,
        )
        _update_sync_state(
            synced=synced,
            matched=matched,
            current_phase=f"Processing Microsoft 365 Group ({scanned}/{max_messages} scanned, {synced} saved)",
        )
        url = data.get("@odata.nextLink") if scanned < max_messages else None

    print(
        f"[EMAIL_INTEL] Group sync finished: synced={synced}, matched={matched}, "
        f"scanned={scanned}/{max_messages}",
        flush=True,
    )
    _update_sync_state(synced=synced, matched=matched)
    return synced, matched


@router.get("/status")
async def get_sync_status(current_user: AdminUser = Depends(get_current_user)):
    """Return sync status overview."""
    config = _get_sync_config()
    sync_state = _snapshot_sync_state()

    total_rows = execute_query("SELECT COUNT(*) AS cnt FROM email_messages")
    total = total_rows[0]["cnt"] if total_rows else 0

    matched_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM email_messages WHERE matched_lead_id IS NOT NULL"
    )
    matched = matched_rows[0]["cnt"] if matched_rows else 0

    pending_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM closure_review_queue WHERE status = 'pending'"
    )
    pending = pending_rows[0]["cnt"] if pending_rows else 0

    with _cost_tracker_lock:
        cost_snapshot = deepcopy(_cost_tracker)

    return {
        "sync_enabled": config.get("sync_enabled", False) if config else False,
        "last_sync": config.get("last_sync_at") if config else None,
        "total_emails": total,
        "matched_emails": matched,
        "pending_reviews": pending,
        "sync_in_progress": sync_state.get("is_running", False),
        "sync_started_at": sync_state.get("started_at"),
        "sync_finished_at": sync_state.get("finished_at"),
        "sync_message": sync_state.get("current_phase"),
        "last_sync_error": sync_state.get("last_error"),
        "last_sync_result": sync_state.get("last_result"),
        "current_sync_counts": {
            "synced": sync_state.get("synced", 0),
            "matched": sync_state.get("matched", 0),
            "flagged_for_review": sync_state.get("flagged_for_review", 0),
        },
        "ai_costs": cost_snapshot,
    }


@router.get("/ai-costs")
async def get_ai_costs(current_user: AdminUser = Depends(get_current_user)):
    """Return AI API token usage and cost breakdown (resets on server restart)."""
    with _cost_tracker_lock:
        return deepcopy(_cost_tracker)


# ---------------------------------------------------------------------------
# Lead Email Intel
# ---------------------------------------------------------------------------

@router.get("/lead/{lead_id}/emails")
async def get_lead_emails(
    lead_id: int,
    current_user: AdminUser = Depends(get_current_user),
):
    """Get all matched emails for a specific lead."""
    audit_log("EMAIL_READ", f"Viewed emails for lead {lead_id}", current_user.username)
    if lead_id == 0:
        # Return recent matched emails for the dashboard feed.
        emails = execute_query(
            "SELECT em.id, em.ms_message_id, em.subject, em.sender_email, em.sender_name, "
            "em.recipient_emails, em.body_preview, em.received_at, em.is_read, em.direction, "
            "em.matched_lead_id, em.match_confidence, em.match_method, em.ai_summary, em.ai_sentiment, "
            "em.ai_action_items, em.ai_ready_to_close, em.ai_close_reasoning, em.processed_at, "
            "l.first_name AS lead_first_name, l.last_name AS lead_last_name, l.status AS lead_status "
            "FROM email_messages em "
            "LEFT JOIN leads l ON l.id = em.matched_lead_id "
            "WHERE em.matched_lead_id IS NOT NULL "
            "ORDER BY em.received_at DESC LIMIT 50"
        )
    else:
        emails = execute_query(
            "SELECT em.id, em.ms_message_id, em.subject, em.sender_email, em.sender_name, "
            "em.recipient_emails, em.body_preview, em.received_at, em.is_read, em.direction, "
            "em.matched_lead_id, em.match_confidence, em.match_method, em.ai_summary, em.ai_sentiment, "
            "em.ai_action_items, em.ai_ready_to_close, em.ai_close_reasoning, em.processed_at, "
            "l.first_name AS lead_first_name, l.last_name AS lead_last_name, l.status AS lead_status "
            "FROM email_messages em "
            "LEFT JOIN leads l ON l.id = em.matched_lead_id "
            "WHERE em.matched_lead_id = %s "
            "ORDER BY em.received_at DESC",
            (lead_id,),
        )
    return emails or []


@router.get("/lead/{lead_id}/context")
async def get_lead_context(
    lead_id: int,
    current_user: AdminUser = Depends(get_current_user),
):
    """AI-generated context summary for a lead based on all matched emails."""
    audit_log("EMAIL_READ", f"Viewed email context for lead {lead_id}", current_user.username)
    lead_rows = execute_query("SELECT * FROM leads WHERE id = %s", (lead_id,))
    if not lead_rows:
        raise HTTPException(status_code=404, detail="Lead not found")
    lead = lead_rows[0]

    emails = execute_query(
        "SELECT subject, sender_email, sender_name, body_preview, received_at, "
        "direction, ai_summary, ai_sentiment "
        "FROM email_messages WHERE matched_lead_id = %s "
        "ORDER BY received_at ASC",
        (lead_id,),
    )
    if not emails:
        return {
            "context": "No email activity found for this lead.",
            "timeline": [],
            "key_insights": [],
            "recommended_action": "No email data available to generate recommendations.",
        }

    timeline = [
        {
            "date": str(e.get("received_at", "")),
            "direction": e.get("direction", ""),
            "subject": e.get("subject", ""),
            "summary": e.get("ai_summary", ""),
            "sentiment": e.get("ai_sentiment", ""),
        }
        for e in emails
    ]

    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        return {
            "context": "AI service unavailable.",
            "timeline": timeline,
            "key_insights": [],
            "recommended_action": "AI service is not configured.",
        }

    email_summaries = "\n".join(
        f"[{e.get('received_at','')}] ({e.get('direction','')}) "
        f"Subject: {e.get('subject','')} | Summary: {e.get('ai_summary','')}"
        for e in emails
    )

    try:
        result, model_used = _reasoning_json_completion(
            system_prompt=(
                "You are a sales intelligence assistant. Given email history for "
                "a lead, produce a JSON object with:\n"
                '- "context": a comprehensive narrative summary of all interactions\n'
                '- "key_insights": array of important observations\n'
                '- "recommended_action": what the sales team should do next\n'
                "Return ONLY valid JSON."
            ),
            user_prompt=(
                f"Lead: {lead.get('first_name','')} {lead.get('last_name','')}\n"
                f"Email: {lead.get('email','')}\n"
                f"Status: {lead.get('status','')}\n"
                f"Province: {lead.get('province','')}\n\n"
                f"Email history ({len(emails)} messages):\n{email_summaries}"
            ),
            max_tokens=700,
        )
        if result is None:
            raise RuntimeError("No AI model available for lead context generation")
        logger.info("Lead context generated with model %s", model_used)
        result["timeline"] = timeline
        return result
    except Exception as e:
        logger.error(f"AI context generation failed: {e}")
        return {
            "context": "Failed to generate AI context.",
            "timeline": timeline,
            "key_insights": [],
            "recommended_action": "Review email timeline manually.",
        }


# ---------------------------------------------------------------------------
# Active matched lead review & new lead candidates
# ---------------------------------------------------------------------------

@router.get("/active-match-review")
async def get_active_match_review(current_user: AdminUser = Depends(get_current_user)):
    """Return active/follow-up leads with matched email activity and cleanup flags."""
    audit_log("EMAIL_READ", "Viewed active matched lead review", current_user.username)
    rows = execute_query(
        """
        WITH match_rollup AS (
            SELECT
                l.id AS lead_id,
                COALESCE(
                    NULLIF(TRIM(CONCAT(COALESCE(l.first_name, ''), ' ', COALESCE(l.last_name, ''))), ''),
                    NULLIF(TRIM(COALESCE(l.name, '')), ''),
                    'Lead'
                ) AS lead_name,
                COALESCE(NULLIF(TRIM(l.email), ''), '') AS lead_email,
                COALESCE(NULLIF(TRIM(l.phone), ''), '') AS lead_phone,
                l.status AS lead_status,
                COALESCE(NULLIF(TRIM(l.source), ''), '') AS lead_source,
                COALESCE(NULLIF(TRIM(l.landing_page), ''), '') AS landing_page,
                COALESCE(NULLIF(TRIM(l.landing_page_url), ''), '') AS landing_page_url,
                l.assigned_dealer_id,
                d.name AS assigned_dealer_name,
                COALESCE(l.email_match_count, 0) AS email_match_count,
                l.last_email_activity,
                l.created_at,
                COUNT(em.id) AS matched_email_count,
                COUNT(*) FILTER (WHERE em.match_method IN ('name', 'phone')) AS weak_match_count,
                COUNT(*) FILTER (
                    WHERE em.match_method IN ('email', 'dealer_email', 'body_email', 'body_name', 'auto_created', 'thread')
                ) AS strong_match_count,
                MAX(em.received_at) AS latest_email_at,
                MAX(em.match_confidence) AS max_match_confidence,
                STRING_AGG(
                    DISTINCT COALESCE(em.match_method, '<null>'),
                    ', '
                    ORDER BY COALESCE(em.match_method, '<null>')
                ) AS match_methods,
                STRING_AGG(
                    DISTINCT COALESCE(NULLIF(BTRIM(em.sender_email), ''), '<unknown>'),
                    ' | '
                    ORDER BY COALESCE(NULLIF(BTRIM(em.sender_email), ''), '<unknown>')
                ) AS contact_emails
            FROM leads l
            JOIN email_messages em ON em.matched_lead_id = l.id
            LEFT JOIN dealers d ON d.id = l.assigned_dealer_id
            WHERE l.status IN ('active', 'follow_up')
            GROUP BY
                l.id,
                lead_name,
                lead_email,
                lead_phone,
                l.status,
                lead_source,
                landing_page,
                landing_page_url,
                l.assigned_dealer_id,
                d.name,
                l.email_match_count,
                l.last_email_activity,
                l.created_at
        ),
        latest_email AS (
            SELECT DISTINCT ON (em.matched_lead_id)
                em.matched_lead_id AS lead_id,
                em.id AS latest_email_id,
                em.received_at AS latest_email_at,
                COALESCE(NULLIF(BTRIM(em.sender_email), ''), '<unknown>') AS latest_sender_email,
                COALESCE(NULLIF(BTRIM(em.sender_name), ''), '<unknown>') AS latest_sender_name,
                COALESCE(em.subject, '') AS latest_subject,
                COALESCE(em.match_method, '<null>') AS latest_match_method,
                em.match_confidence AS latest_match_confidence,
                COALESCE(em.ai_summary, '') AS latest_ai_summary,
                COALESCE(em.ai_sentiment, 'neutral') AS latest_ai_sentiment,
                COALESCE(em.ai_ready_to_close, FALSE) AS latest_ai_ready_to_close
            FROM email_messages em
            JOIN leads l ON l.id = em.matched_lead_id
            WHERE l.status IN ('active', 'follow_up')
            ORDER BY em.matched_lead_id, em.received_at DESC, em.id DESC
        ),
        dup_email AS (
            SELECT
                LOWER(BTRIM(email)) AS email_key,
                COUNT(*) FILTER (WHERE status IN ('active', 'follow_up')) AS active_duplicate_email_count
            FROM leads
            WHERE email IS NOT NULL AND BTRIM(email) <> ''
            GROUP BY 1
        )
        SELECT
            mr.*,
            le.latest_email_id,
            le.latest_sender_email,
            le.latest_sender_name,
            le.latest_subject,
            le.latest_match_method,
            le.latest_match_confidence,
            le.latest_ai_summary,
            le.latest_ai_sentiment,
            le.latest_ai_ready_to_close,
            ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - mr.created_at)) / 86400.0, 1) AS lead_age_days,
            ROUND(
                EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - COALESCE(mr.last_email_activity, mr.latest_email_at)))
                / 86400.0,
                1
            ) AS days_since_last_email,
            COALESCE(de.active_duplicate_email_count, 0) AS active_duplicate_email_count,
            (mr.assigned_dealer_id IS NULL) AS missing_dealer,
            (
                mr.weak_match_count > 0
                OR COALESCE(de.active_duplicate_email_count, 0) > 1
                OR COALESCE(mr.max_match_confidence, 0) < 0.90
            ) AS needs_match_review,
            CASE
                WHEN mr.assigned_dealer_id IS NULL THEN 'high'
                WHEN mr.weak_match_count > 0 OR COALESCE(de.active_duplicate_email_count, 0) > 1 THEN 'medium'
                ELSE 'low'
            END AS review_priority,
            CASE
                WHEN mr.assigned_dealer_id IS NULL AND COALESCE(mr.landing_page, '') <> '' THEN
                    'Matched email activity exists, the lead is missing a dealer assignment, and the lead came from ' || mr.landing_page || '.'
                WHEN mr.assigned_dealer_id IS NULL THEN 'Matched email activity exists, but the lead is still missing a dealer assignment.'
                WHEN COALESCE(de.active_duplicate_email_count, 0) > 1 THEN 'Multiple open leads share this customer email, so email activity may be split across duplicates.'
                WHEN mr.weak_match_count > 0 THEN 'This lead includes at least one weak email match from a name- or phone-based fallback.'
                ELSE 'Matched email activity is available for manual review.'
            END AS review_reason
        FROM match_rollup mr
        LEFT JOIN latest_email le ON le.lead_id = mr.lead_id
        LEFT JOIN dup_email de ON de.email_key = LOWER(BTRIM(mr.lead_email))
        ORDER BY
            (mr.assigned_dealer_id IS NULL) DESC,
            (
                mr.weak_match_count > 0
                OR COALESCE(de.active_duplicate_email_count, 0) > 1
                OR COALESCE(mr.max_match_confidence, 0) < 0.90
            ) DESC,
            COALESCE(le.latest_email_at, mr.latest_email_at) DESC,
            mr.lead_id DESC
        """
    )
    return rows or []


@router.get("/new-lead-candidates")
async def get_new_lead_candidates(current_user: AdminUser = Depends(get_current_user)):
    """Return unmatched inbound emails that look likely to be net-new leads."""
    audit_log("EMAIL_READ", "Viewed new lead candidates", current_user.username)
    rows = execute_query(
        """
        WITH candidates AS (
            SELECT
                em.id,
                em.received_at,
                COALESCE(NULLIF(BTRIM(em.sender_email), ''), '<unknown>') AS sender_email,
                COALESCE(NULLIF(BTRIM(em.sender_name), ''), '<unknown>') AS sender_name,
                COALESCE(em.subject, '') AS subject,
                COALESCE(em.body_preview, '') AS body_preview,
                LOWER(COALESCE(em.subject, '')) AS subject_lc,
                LOWER(COALESCE(em.body_preview, '')) AS preview_lc,
                CASE
                    WHEN LOWER(COALESCE(em.sender_email, '')) LIKE 'lead@%' THEN 'dealer-site forwarded lead'
                    WHEN LOWER(COALESCE(em.subject, '')) LIKE '%contact form submission%' THEN 'contact form submission'
                    WHEN LOWER(COALESCE(em.subject, '')) LIKE '%form submission%' THEN 'website form submission'
                    WHEN LOWER(COALESCE(em.subject, '')) LIKE '%request a quote%' THEN 'quote request'
                    WHEN LOWER(COALESCE(em.subject, '')) LIKE '%consultation%' THEN 'consultation request'
                    WHEN LOWER(COALESCE(em.subject, '')) LIKE '%3m lead%' THEN 'lead assignment email'
                    WHEN LOWER(COALESCE(em.body_preview, '')) LIKE '%submission data%' THEN 'submission data in body'
                    ELSE 'manual review'
                END AS candidate_reason,
                (
                    CASE WHEN LOWER(COALESCE(em.sender_email, '')) LIKE 'lead@%' THEN 5 ELSE 0 END
                    + CASE
                        WHEN LOWER(COALESCE(em.subject, '')) LIKE '%contact form submission%' THEN 4
                        WHEN LOWER(COALESCE(em.subject, '')) LIKE '%form submission%' THEN 4
                        ELSE 0
                    END
                    + CASE
                        WHEN LOWER(COALESCE(em.subject, '')) LIKE '%request a quote%' THEN 3
                        WHEN LOWER(COALESCE(em.subject, '')) LIKE '%consultation%' THEN 3
                        WHEN LOWER(COALESCE(em.subject, '')) LIKE '%3m lead%' THEN 3
                        ELSE 0
                    END
                    + CASE WHEN LOWER(COALESCE(em.body_preview, '')) LIKE '%submission data%' THEN 2 ELSE 0 END
                    + CASE WHEN LOWER(COALESCE(em.body_preview, '')) LIKE '%first_name:%' THEN 2 ELSE 0 END
                    + CASE WHEN LOWER(COALESCE(em.body_preview, '')) LIKE '%email:%' THEN 1 ELSE 0 END
                    + CASE WHEN LOWER(COALESCE(em.body_preview, '')) LIKE '%phone:%' THEN 1 ELSE 0 END
                ) AS candidate_score
            FROM email_messages em
            WHERE em.matched_lead_id IS NULL
              AND em.direction = 'inbound'
              AND (
                  LOWER(COALESCE(em.sender_email, '')) LIKE 'lead@%'
                  OR LOWER(COALESCE(em.subject, '')) LIKE '%form submission%'
                  OR LOWER(COALESCE(em.subject, '')) LIKE '%request a quote%'
                  OR LOWER(COALESCE(em.subject, '')) LIKE '%consultation%'
                  OR LOWER(COALESCE(em.subject, '')) LIKE '%3m lead%'
                  OR LOWER(COALESCE(em.body_preview, '')) LIKE '%submission data%'
                  OR LOWER(COALESCE(em.body_preview, '')) LIKE '%first_name:%'
              )
        )
        SELECT
            c.id,
            c.received_at,
            c.sender_email,
            c.sender_name,
            c.subject,
            c.body_preview,
            c.candidate_reason,
            c.candidate_score,
            COALESCE(existing.sender_lead_count, 0) AS existing_sender_lead_count
        FROM candidates c
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS sender_lead_count
            FROM leads l
            WHERE l.email IS NOT NULL
              AND BTRIM(l.email) <> ''
              AND LOWER(BTRIM(l.email)) = LOWER(BTRIM(c.sender_email))
        ) existing ON TRUE
        WHERE c.candidate_score >= 3
        ORDER BY c.candidate_score DESC, c.received_at DESC, c.id DESC
        LIMIT 100
        """
    )
    return rows or []


@router.post("/new-lead-candidates/{email_id}/create")
async def create_lead_from_candidate(email_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Create or match a lead from a high-signal unmatched inbound email."""
    rows = execute_query(
        """
        SELECT id, ms_message_id, conversation_id, subject, sender_email, sender_name, recipient_emails,
               body_preview, body_text, received_at, matched_lead_id, direction
        FROM email_messages
        WHERE id = %s
        """,
        (email_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Candidate email not found")

    email_row = rows[0]
    if email_row.get("matched_lead_id"):
        raise HTTPException(status_code=400, detail="This email is already matched to a lead")
    if email_row.get("direction") != "inbound":
        raise HTTPException(status_code=400, detail="Only inbound candidate emails can be turned into leads")

    config = _get_sync_config_decrypted()
    mailbox_email = _get_target_mailbox_email(config)
    token = _refresh_access_token(config) if config and config.get("access_token") else ""
    email_row = _hydrate_email_record_for_parsing(email_row, token, config)
    leads_cache = _load_leads_cache(mailbox_email)
    candidate = _extract_candidate_lead_fields(email_row, mailbox_email)
    if not candidate:
        raise HTTPException(status_code=400, detail="Could not extract enough lead details from this email")

    action = "created"
    match_method = "manual_candidate_create"
    match_confidence = 0.97

    lead_id = None
    email_key = candidate["email"]
    if email_key and email_key in leads_cache["by_email"]:
        lead_id = leads_cache["by_email"][email_key]
        action = "matched_existing"
        match_method = "body_email"
        match_confidence = 0.90
    elif candidate["first_name"] and candidate["last_name"]:
        name_key = f"{candidate['first_name'].lower()} {candidate['last_name'].lower()}"
        if name_key in leads_cache["by_name"]:
            lead_id = leads_cache["by_name"][name_key]
            action = "matched_existing"
            match_method = "body_name"
            match_confidence = 0.85

    if not lead_id:
        lead_id = _insert_email_candidate_lead(candidate, email_row, leads_cache, current_user.username)
        if not lead_id:
            raise HTTPException(status_code=500, detail="Failed to create lead from candidate email")
    else:
        _enrich_existing_lead_from_candidate(lead_id, candidate)

    matched_email_count = _attach_email_candidate_to_lead(email_row, lead_id, match_method, match_confidence)
    audit_log(
        "EMAIL_CREATE_LEAD",
        f"{action} lead {lead_id} from email candidate {email_id} ({candidate['email']})",
        current_user.username,
    )
    return {
        "success": True,
        "action": action,
        "lead_id": lead_id,
        "matched_email_count": matched_email_count,
    }


# ---------------------------------------------------------------------------
# Closure Review Queue
# ---------------------------------------------------------------------------

@router.get("/review-queue")
async def get_review_queue(current_user: AdminUser = Depends(get_current_user)):
    """Get all pending closure reviews with lead details."""
    rows = execute_query("""
        SELECT crq.id AS id, crq.lead_id, crq.flagged_at,
               crq.ai_reasoning, crq.days_inactive, crq.last_email_at,
               crq.email_count, crq.status,
               TRIM(CONCAT(COALESCE(l.first_name, ''), ' ', COALESCE(l.last_name, ''))) AS lead_name,
               l.email AS lead_email,
               COALESCE(d.name, l.final_dealer_selection, 'Unassigned') AS dealer_name,
               l.phone, l.province, l.status AS lead_status,
               l.email_match_count, l.last_email_activity, l.email_sentiment
        FROM closure_review_queue crq
        JOIN leads l ON l.id = crq.lead_id
        LEFT JOIN dealers d ON d.id = l.assigned_dealer_id
        WHERE crq.status = 'pending'
        ORDER BY crq.flagged_at DESC
    """)
    return rows or []


@router.post("/review-queue/{review_id}/approve")
async def approve_review(
    review_id: int,
    current_user: AdminUser = Depends(get_current_user),
):
    """Approve closure: mark lead as converted, update review status."""
    review = execute_query(
        "SELECT * FROM closure_review_queue WHERE id = %s", (review_id,)
    )
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    review = review[0]

    if review["status"] != "pending":
        raise HTTPException(status_code=400, detail="Review already processed")

    # Update review
    execute_query(
        "UPDATE closure_review_queue SET status = 'approved', "
        "reviewed_by = %s, reviewed_at = CURRENT_TIMESTAMP WHERE id = %s",
        (current_user.username, review_id),
        fetch=False,
    )

    # Update lead status
    execute_query(
        "UPDATE leads SET status = 'converted' WHERE id = %s",
        (review["lead_id"],),
        fetch=False,
    )

    # Add lead log entry
    execute_query(
        "INSERT INTO lead_logs (lead_id, log_type, message, created_by, created_at) "
        "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)",
        (
            review["lead_id"],
            "status_change",
            json.dumps({
                "old_status": "active",
                "new_status": "converted",
                "source": "email_intel_closure_review",
                "ai_reasoning": review.get("ai_reasoning", ""),
            }),
            current_user.username,
        ),
        fetch=False,
    )

    audit_log("REVIEW_APPROVED", f"Review {review_id} approved, lead {review['lead_id']} converted", current_user.username)
    return {"success": True, "lead_id": review["lead_id"], "new_status": "converted"}


@router.post("/review-queue/{review_id}/dismiss")
async def dismiss_review(
    review_id: int,
    current_user: AdminUser = Depends(get_current_user),
):
    """Dismiss the closure flag."""
    review = execute_query(
        "SELECT * FROM closure_review_queue WHERE id = %s", (review_id,)
    )
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    review = review[0]

    if review["status"] != "pending":
        raise HTTPException(status_code=400, detail="Review already processed")

    # Update review
    execute_query(
        "UPDATE closure_review_queue SET status = 'dismissed', "
        "reviewed_by = %s, reviewed_at = CURRENT_TIMESTAMP WHERE id = %s",
        (current_user.username, review_id),
        fetch=False,
    )

    # Reset closure flag on lead
    execute_query(
        "UPDATE leads SET ai_closure_flagged = FALSE WHERE id = %s",
        (review["lead_id"],),
        fetch=False,
    )

    audit_log("REVIEW_DISMISSED", f"Review {review_id} dismissed for lead {review['lead_id']}", current_user.username)
    return {"success": True, "lead_id": review["lead_id"]}
