import json
import os
import logging
import html
import threading
import asyncio
from datetime import datetime, timedelta
from copy import deepcopy
from typing import Optional, List, Tuple
from urllib.parse import urlencode

import requests as http_requests
from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import AdminUser, get_current_user
from db import execute_query

logger = logging.getLogger("lead_allocation")

router = APIRouter(prefix="/api/email-intel", tags=["Email Intelligence"])

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
MS_LOGIN_BASE = "https://login.microsoftonline.com"
OAUTH_SCOPES = "Mail.Read offline_access User.Read"

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
    sync_enabled: Optional[bool] = None
    sync_interval_minutes: Optional[int] = None


class OAuthCallbackIn(BaseModel):
    code: str


# ---------------------------------------------------------------------------
# Helpers -- DB convenience wrappers
# ---------------------------------------------------------------------------

def _get_sync_config() -> Optional[dict]:
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

            # Track token usage & cost
            usage = response.usage
            if usage:
                inp = usage.prompt_tokens or 0
                out = usage.completion_tokens or 0
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

            return json.loads(content), model
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


def _try_auto_create_lead(email_record: dict, mailbox_email: str, leads_cache: dict) -> tuple:
    """
    If an email contains customer info from a form submission (Name, Email, Phone)
    and no matching lead exists, auto-create the lead.

    This handles the case where Colin forwards form submissions to dealers
    before the lead is created in Lasso/the CRM.

    Returns: (lead_id, confidence, method) or (None, 0, None)
    """
    body = f"{email_record.get('body_text') or ''} {email_record.get('body_preview') or ''}"
    subject = email_record.get("subject", "")

    # Only auto-create from form submissions or lead assignment emails
    is_form = (
        "form submission" in body.lower()
        or "form submission" in subject.lower()
        or "squarespace" in email_record.get("sender_email", "").lower()
        or _re.search(r'\b(lead|Lead)\b.*-.*\d{2}/\d{2}/\d{4}', subject)
        or "contact form submission" in subject.lower()
        or email_record.get("sender_email", "").startswith("lead@")
    )
    if not is_form:
        return (None, 0, None)

    # Extract customer info — try multiple formats
    customer_email = None
    customer_first = None
    customer_last = None
    customer_phone = None
    customer_city = None
    customer_province = None
    customer_company = None
    customer_project_type = None

    # Format 1: "Name: First Last" + "Email: xxx@yyy.com"
    name_m = _re.search(r'Name:\s*([A-Z][a-z]+)\s+([A-Z][a-zA-Z\'\-]+)', body)
    if name_m:
        customer_first = name_m.group(1)
        customer_last = name_m.group(2)

    # Format 2: "First Name: X" + "Last Name: Y"
    if not customer_first:
        fn_m = _re.search(r'First\s*Name:\s*(\S+)', body)
        ln_m = _re.search(r'Last\s*Name:\s*(\S+)', body)
        if fn_m and ln_m:
            customer_first = fn_m.group(1)
            customer_last = ln_m.group(1)

    # Extract email
    em_m = _re.search(r'Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', body)
    if em_m:
        customer_email = em_m.group(1).lower()

    # Extract phone
    ph_m = _re.search(r'Phone:\s*([(\d][\d\s().-]{8,})', body)
    if ph_m:
        customer_phone = ph_m.group(1).strip()

    # Extract city/province
    city_m = _re.search(r'(?:City|Location)[^:]*:\s*([^,\n]+)', body)
    if city_m:
        customer_city = city_m.group(1).strip()
    prov_m = _re.search(r'Province:\s*([A-Za-z\s]+?)(?:\s*(?:Project|Phone|Email|$))', body)
    if prov_m:
        customer_province = prov_m.group(1).strip()

    # Extract project type
    proj_m = _re.search(r'Type of Project:\s*(\w+)', body)
    if proj_m:
        customer_project_type = proj_m.group(1).strip()

    # Extract company
    comp_m = _re.search(r'Company\s*Name[^:]*:\s*([^\n]+?)(?:\s*Type|\s*$)', body)
    if comp_m:
        val = comp_m.group(1).strip()
        if val and val.lower() not in ("n/a", "na", "none", ""):
            customer_company = val

    # Must have at least a name AND email to create a lead
    if not customer_email or not customer_first:
        return (None, 0, None)

    # Don't create if this email already exists in leads cache
    if customer_email and customer_email in leads_cache["by_email"]:
        # Already exists — return the match
        return (leads_cache["by_email"][customer_email], 0.90, "body_email")

    # Don't create if this name already exists
    if customer_first and customer_last:
        name_key = f"{customer_first.lower()} {customer_last.lower()}"
        if name_key in leads_cache["by_name"]:
            return (leads_cache["by_name"][name_key], 0.85, "body_name")

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

    # Create the lead
    full_name = f"{customer_first or ''} {customer_last or ''}".strip()
    result = execute_query(
        """INSERT INTO leads (
            name, first_name, last_name, email, phone,
            city, province, company_name, project_type,
            dealer_email, source, status, email_match_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id""",
        (
            full_name, customer_first, customer_last, customer_email or "",
            customer_phone or "", customer_city or "", customer_province or "",
            customer_company or "", customer_project_type or "",
            dealer_email or "", "Email Auto-Created", "active", 1,
        ),
    )
    if result:
        new_id = result[0]["id"]
        # Update the in-memory cache so subsequent emails can match
        if customer_email:
            leads_cache["by_email"][customer_email] = new_id
        if customer_first and customer_last:
            name_key = f"{customer_first.lower()} {customer_last.lower()}"
            leads_cache["by_name"][name_key] = new_id
        print(f"[EMAIL_INTEL] AUTO-CREATED Lead #{new_id}: {full_name} ({customer_email}) from '{email_record.get('subject', '')[:50]}'", flush=True)
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
            "scope": OAUTH_SCOPES,
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
        logger.error(f"Graph API error {resp.status_code}: {resp.text[:300]}")
        raise HTTPException(status_code=502, detail=f"Graph API error: {resp.status_code}")
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

    existing = _get_sync_config()
    if not existing:
        required_fields = ("ms_tenant_id", "ms_client_id", "ms_client_secret", "ms_redirect_uri")
        missing = [field for field in required_fields if not updates.get(field)]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required config fields: {', '.join(missing)}")

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
    }
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
        print(f"[EMAIL_INTEL] OAuth token exchange FAILED: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"{frontend_url}?oauth_error={quote(str(e)[:200])}")


def _exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access/refresh tokens."""
    config = _get_sync_config_decrypted()
    if not config:
        raise HTTPException(status_code=400, detail="OAuth config not set.")

    secret = config["ms_client_secret"]
    # Debug: verify decrypted secret is intact (log length + first/last 4 chars only)
    print(f"[EMAIL_INTEL] Exchanging code:")
    print(f"  tenant   = {config['ms_tenant_id']}")
    print(f"  client   = {config['ms_client_id']}")
    print(f"  redirect = {config['ms_redirect_uri']}")
    print(f"  secret   = {secret[:4]}...{secret[-4:]} (len={len(secret)})")
    print(f"  scope    = {OAUTH_SCOPES}")
    print(f"  code     = {code[:10]}... (len={len(code)})")

    token_url = f"{MS_LOGIN_BASE}/{config['ms_tenant_id']}/oauth2/v2.0/token"
    payload = {
        "client_id": config["ms_client_id"],
        "client_secret": secret,
        "code": code,
        "redirect_uri": config["ms_redirect_uri"],
        "grant_type": "authorization_code",
        "scope": OAUTH_SCOPES,
    }

    # Use explicit Content-Type header and encode body manually to avoid any encoding issues
    from urllib.parse import urlencode
    encoded_body = urlencode(payload)
    print(f"[EMAIL_INTEL] POST {token_url}")
    print(f"[EMAIL_INTEL] Body (first 200): {encoded_body[:200]}")

    resp = http_requests.post(
        token_url,
        data=encoded_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"[EMAIL_INTEL] Token exchange failed ({resp.status_code}): {resp.text[:500]}")
        raise Exception(f"Token exchange failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    expires_at = datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600))

    # Fetch user profile to get email
    profile = http_requests.get(
        f"{GRAPH_BASE}/me",
        headers={"Authorization": f"Bearer {data['access_token']}"},
        timeout=15,
    ).json()
    user_email = profile.get("mail") or profile.get("userPrincipalName", "")

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
        print(f"[EMAIL_INTEL] Token ready, starting inbox sync...", flush=True)

        synced = 0
        matched = 0

        _update_sync_state(current_phase="Syncing inbox")
        url = (
            f"{GRAPH_BASE}/me/messages?"
            "$top=1000&$orderby=receivedDateTime desc"
            "&$select=id,subject,from,toRecipients,body,bodyPreview,receivedDateTime,isRead,conversationId"
        )
        synced_inbound, matched_inbound = await _sync_messages(token, url, direction="inbound")
        synced += synced_inbound
        matched += matched_inbound
        _update_sync_state(synced=synced, matched=matched)

        _update_sync_state(current_phase="Syncing sent mail")
        url_sent = (
            f"{GRAPH_BASE}/me/mailFolders/SentItems/messages?"
            "$top=1000&$orderby=receivedDateTime desc"
            "&$select=id,subject,from,toRecipients,body,bodyPreview,receivedDateTime,isRead,conversationId"
        )
        synced_outbound, matched_outbound = await _sync_messages(token, url_sent, direction="outbound")
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
        print(f"[EMAIL_INTEL] SYNC FAILED: {exc}")
        traceback.print_exc()
        logger.exception("Email sync job failed")
        _update_sync_state(
            is_running=False,
            finished_at=datetime.utcnow(),
            current_phase="Failed",
            last_error=str(exc),
        )
        audit_log("EMAIL_SYNC_FAILED", str(exc), started_by)


async def _sync_messages(token: str, url: str, direction: str = "inbound"):
    """
    Paginate through Graph messages and sync them.
    Fast: parallel AI calls in batches, skips AI for unmatched emails.
    """
    synced = 0
    matched = 0

    # Cache mailbox email once
    config = _get_sync_config()
    mailbox_email = (config.get("user_email") or "").lower().strip() if config else ""

    # Pre-load existing message IDs to skip duplicates instantly
    existing_ids_rows = execute_query("SELECT ms_message_id FROM email_messages")
    existing_ids = {r["ms_message_id"] for r in (existing_ids_rows or [])}

    # Pre-load ALL leads into memory for fast matching (1 DB query instead of thousands)
    leads_cache = _load_leads_cache(mailbox_email)

    page_num = 0
    while url:
        page_num += 1
        print(f"[EMAIL_INTEL] Fetching page {page_num} from Graph API...", flush=True)
        data = _graph_get(token, url)
        messages = data.get("value", [])
        print(f"[EMAIL_INTEL] Page {page_num}: got {len(messages)} messages", flush=True)
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
            # Use full body for matching, but store truncated preview for display
            body_preview = _normalize_email_text(msg.get("bodyPreview", "") or full_body_text[:500], 500)

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
                "body_text": full_body_text[:2000],  # Full body for matching (up to 2000 chars)
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
                # Exclude body_text from DB insert (used for matching only, not stored beyond body_preview)
                insert_rec = {k: v for k, v in rec.items() if k != "body_text"}
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

        print(f"[EMAIL_INTEL] Page {page_num} complete: total synced={synced}, matched={matched}", flush=True)
        _update_sync_state(synced=synced, matched=matched, current_phase=f"Processing {direction} ({synced} done)")
        url = data.get("@odata.nextLink")

    print(f"[EMAIL_INTEL] {direction} sync finished: synced={synced}, matched={matched}", flush=True)
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
               COALESCE(d.name, l.final_installer_selection, 'Unassigned') AS dealer_name,
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
