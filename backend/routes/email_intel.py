import json
import os
import logging
import hashlib
import html
from datetime import datetime, timedelta
from math import sqrt
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
    for model in os.getenv("OPENAI_EMAIL_MODELS", "gpt-5-nano,gpt-4o-mini").split(",")
    if model.strip()
]
OPENAI_REASONING_MODEL_CANDIDATES = [
    model.strip()
    for model in os.getenv("OPENAI_REASONING_MODELS", "gpt-5-mini,gpt-4o-mini").split(",")
    if model.strip()
]
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANTHROPIC_REASONING_MODEL = os.getenv("ANTHROPIC_REASONING_MODEL", "claude-sonnet-4-20250514")

# Env-based MS config (auto-seeds DB on first access)
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI", "http://localhost:8000/api/email-intel/oauth/callback-redirect")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_EMAIL_BODY_CHARS = _env_int("EMAIL_INTEL_MAX_BODY_CHARS", 1200)
EMBEDDING_CANDIDATE_LIMIT = _env_int("EMAIL_INTEL_EMBEDDING_CANDIDATE_LIMIT", 8)
EMBEDDING_SCORE_THRESHOLD = _env_float("EMAIL_INTEL_EMBEDDING_SCORE_THRESHOLD", 0.45)
EMBEDDING_GAP_THRESHOLD = _env_float("EMAIL_INTEL_EMBEDDING_GAP_THRESHOLD", 0.03)


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

_embedding_cache: dict[str, List[float]] = {}

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
            candidate = obj_match.group(0)
            try:
                return json.loads(candidate)
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


def _name_tokens(value: str) -> List[str]:
    return [
        token
        for token in _re.findall(r"[a-zA-Z]+", (value or "").lower())
        if len(token) >= 2 and token not in {"re", "fw", "fwd", "mr", "mrs", "ms"}
    ]


def _openai_json_completion(
    system_prompt: str,
    user_prompt: str,
    model_candidates: List[str],
    max_tokens: int,
) -> Tuple[Optional[dict], Optional[str]]:
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
            if model.startswith("gpt-5"):
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens
                request_kwargs["temperature"] = 0.1

            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or "{}"
            return _extract_json(content), model
        except Exception as exc:
            logger.warning("OpenAI JSON call failed for %s: %s", model, exc)

    return None, None


def _anthropic_json_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
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
        logger.warning("Anthropic JSON call failed for %s: %s", ANTHROPIC_REASONING_MODEL, exc)
        return None, None


def _reasoning_json_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
) -> Tuple[Optional[dict], Optional[str]]:
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


def _get_embedding(text: str, cache_key: Optional[str] = None) -> Optional[List[float]]:
    if not text.strip():
        return None
    if cache_key and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    client = _get_openai_client()
    if client is None:
        return None

    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text[:4000],
        )
        embedding = response.data[0].embedding
        if cache_key:
            _embedding_cache[cache_key] = embedding
        return embedding
    except Exception as exc:
        logger.warning("OpenAI embedding call failed for %s: %s", OPENAI_EMBEDDING_MODEL, exc)
        return None


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    denominator = sqrt(sum(a * a for a in vec_a)) * sqrt(sum(b * b for b in vec_b))
    if not denominator:
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b)) / denominator


async def analyze_email_with_ai(
    email_body: str, email_subject: str, lead_data: dict = None
) -> dict:
    """
    Use the cheapest configured OpenAI model for bulk parsing, with fallback
    to a known working model if the preferred one is unavailable.
    """
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
        max_tokens=200,
    )
    if result is None:
        logger.error("AI email analysis failed for all configured OpenAI models")
        return dict(_DEFAULT_AI_RESULT)

    logger.info("Email analysis completed with model %s", model)
    return {
        **dict(_DEFAULT_AI_RESULT),
        **result,
    }


# ---------------------------------------------------------------------------
# Lead matching
# ---------------------------------------------------------------------------

def _build_embedding_candidates(match_emails: List[str], sender_name: str) -> List[dict]:
    candidates: dict[int, dict] = {}
    email_domains = {
        email.split("@", 1)[1]
        for email in match_emails
        if "@" in email and "." in email.split("@", 1)[1]
    }

    for domain in email_domains:
        rows = execute_query(
            "SELECT id, first_name, last_name, email, dealer_email, company_name, province, status "
            "FROM leads "
            "WHERE status NOT IN ('archived', 'dead') "
            "AND ("
            "  (email IS NOT NULL AND POSITION('@' IN email) > 0 AND LOWER(SPLIT_PART(email, '@', 2)) = %s) "
            "  OR "
            "  (dealer_email IS NOT NULL AND POSITION('@' IN dealer_email) > 0 AND LOWER(SPLIT_PART(dealer_email, '@', 2)) = %s)"
            ") "
            "ORDER BY created_at DESC LIMIT %s",
            (domain, domain, EMBEDDING_CANDIDATE_LIMIT),
        )
        for row in rows or []:
            candidates[row["id"]] = row

    name_parts = _name_tokens(sender_name)
    if len(name_parts) >= 2:
        rows = execute_query(
            "SELECT id, first_name, last_name, email, dealer_email, company_name, province, status "
            "FROM leads "
            "WHERE status NOT IN ('archived', 'dead') "
            "AND LOWER(first_name) = %s AND LOWER(last_name) = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (name_parts[0], name_parts[-1], EMBEDDING_CANDIDATE_LIMIT),
        )
        for row in rows or []:
            candidates[row["id"]] = row
    elif len(name_parts) == 1:
        rows = execute_query(
            "SELECT id, first_name, last_name, email, dealer_email, company_name, province, status "
            "FROM leads "
            "WHERE status NOT IN ('archived', 'dead') "
            "AND (LOWER(first_name) = %s OR LOWER(last_name) = %s) "
            "ORDER BY created_at DESC LIMIT %s",
            (name_parts[0], name_parts[0], EMBEDDING_CANDIDATE_LIMIT),
        )
        for row in rows or []:
            candidates[row["id"]] = row

    return list(candidates.values())[:EMBEDDING_CANDIDATE_LIMIT]


def _match_with_embeddings(email_record: dict, candidates: List[dict]) -> Tuple[Optional[int], float, Optional[str]]:
    if len(candidates) < 2:
        return (None, 0.0, None)

    try:
        recipients = json.loads(email_record.get("recipient_emails") or "[]")
    except json.JSONDecodeError:
        recipients = []

    email_profile = (
        f"Subject: {email_record.get('subject', '')}\n"
        f"Sender: {email_record.get('sender_name', '')} <{email_record.get('sender_email', '')}>\n"
        f"Recipients: {', '.join(recipients)}\n"
        f"Preview: {email_record.get('body_preview', '')}\n"
        f"Body: {_normalize_email_text(email_record.get('body_text', ''), MAX_EMAIL_BODY_CHARS)}"
    )
    email_embedding = _get_embedding(
        email_profile,
        cache_key=f"email:{hashlib.sha1(email_profile.encode('utf-8')).hexdigest()}",
    )
    if email_embedding is None:
        return (None, 0.0, None)

    scores: List[Tuple[float, int]] = []
    for candidate in candidates:
        lead_profile = (
            f"Lead name: {(candidate.get('first_name') or '').strip()} {(candidate.get('last_name') or '').strip()}\n"
            f"Primary email: {candidate.get('email') or ''}\n"
            f"Dealer email: {candidate.get('dealer_email') or ''}\n"
            f"Company: {candidate.get('company_name') or ''}\n"
            f"Province: {candidate.get('province') or ''}\n"
            f"Status: {candidate.get('status') or ''}"
        )
        lead_embedding = _get_embedding(
            lead_profile,
            cache_key=f"lead:{candidate['id']}",
        )
        if lead_embedding is None:
            continue
        scores.append((_cosine_similarity(email_embedding, lead_embedding), candidate["id"]))

    if not scores:
        return (None, 0.0, None)

    scores.sort(reverse=True)
    best_score, best_lead_id = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0

    if best_score < EMBEDDING_SCORE_THRESHOLD:
        return (None, 0.0, None)
    if len(scores) > 1 and (best_score - second_score) < EMBEDDING_GAP_THRESHOLD:
        return (None, 0.0, None)

    confidence = min(0.9, max(0.55, round(best_score, 2)))
    return (best_lead_id, confidence, "embedding")


def match_email_to_lead(email_record: dict):
    """
    Match the OTHER party in the email to a lead.
    The connected mailbox owner (e.g. cmacleod@windowfilmcanada.ca) is excluded —
    we want to find which customer/dealer lead this conversation is about.

    For inbound emails: match the SENDER to a lead.
    For outbound emails: match the RECIPIENT(S) to a lead.

    Match priority:
    1. Exact email address match
    2. Name match (sender name for inbound)
    3. Phone match (phone number in email body)

    Returns: (lead_id, confidence, method) or (None, 0, None)
    """
    # Get connected mailbox email to exclude from matching
    config = _get_sync_config()
    mailbox_email = (config.get("user_email") or "").lower().strip() if config else ""
    # Also exclude common company domain emails
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
        # Outbound: match recipients (excluding the mailbox owner)
        match_emails = [r.lower().strip() for r in recipients if r and r.lower().strip() != mailbox_email]
    else:
        # Inbound: match sender (if it's not the mailbox owner)
        if sender and sender != mailbox_email:
            match_emails = [sender]
        else:
            # Sender is mailbox owner somehow — try recipients
            match_emails = [r.lower().strip() for r in recipients if r and r.lower().strip() != mailbox_email]

    # Filter out same-domain emails (internal company emails aren't leads)
    if mailbox_domain:
        external_emails = [e for e in match_emails if not e.endswith(f"@{mailbox_domain}")]
        # Only use external filter if it leaves us with candidates
        if external_emails:
            match_emails = external_emails

    # --- 1. Exact email match ---
    if match_emails:
        placeholders = ", ".join(["%s"] * len(match_emails))
        rows = execute_query(
            f"SELECT id, first_name, last_name, email FROM leads WHERE LOWER(email) IN ({placeholders}) ORDER BY created_at DESC LIMIT 1",
            tuple(match_emails),
        )
        if rows:
            return (rows[0]["id"], 1.0, "email")

        # Also check dealer_email field
        rows = execute_query(
            f"SELECT id FROM leads WHERE LOWER(dealer_email) IN ({placeholders}) ORDER BY created_at DESC LIMIT 1",
            tuple(match_emails),
        )
        if rows:
            return (rows[0]["id"], 0.95, "dealer_email")

    # --- 2. Name match (for inbound from external sender) ---
    if direction == "inbound" and sender_name and sender != mailbox_email:
        parts = sender_name.split()
        if len(parts) >= 2:
            rows = execute_query(
                "SELECT id FROM leads WHERE LOWER(first_name) = %s AND LOWER(last_name) = %s ORDER BY created_at DESC LIMIT 1",
                (parts[0], parts[-1]),
            )
            if rows:
                return (rows[0]["id"], 0.85, "name")

    # --- 3. Phone match ---
    if body and match_emails:
        leads_with_phone = execute_query(
            "SELECT id, phone FROM leads WHERE phone IS NOT NULL AND phone != '' ORDER BY created_at DESC LIMIT 500"
        )
        normalized_body = _normalize_phone(body)
        for lead in leads_with_phone or []:
            phone = _normalize_phone((lead.get("phone") or "").strip())
            if phone and len(phone) >= 7 and phone in normalized_body:
                return (lead["id"], 0.75, "phone")

    candidates = _build_embedding_candidates(match_emails, sender_name)
    lead_id, confidence, method = _match_with_embeddings(email_record, candidates)
    if lead_id:
        return (lead_id, confidence, method)

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
            max_tokens=200,
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
    """Manually trigger an email sync from MS Graph."""
    global _last_sync_time
    now = datetime.utcnow()
    if _last_sync_time and (now - _last_sync_time).total_seconds() < 300:
        raise HTTPException(status_code=429, detail="Sync rate limited. Wait 5 minutes between syncs.")
    _last_sync_time = now

    config = _get_sync_config_decrypted()
    if not config or not config.get("access_token"):
        raise HTTPException(status_code=400, detail="OAuth not configured or not authorised.")

    token = _refresh_access_token(config)

    synced = 0
    matched = 0

    # --- Fetch inbound messages (with pagination) ---
    url = (
        f"{GRAPH_BASE}/me/messages?"
        "$top=50&$orderby=receivedDateTime desc"
        "&$select=id,subject,from,toRecipients,bodyPreview,body,receivedDateTime,isRead"
    )
    synced_inbound, matched_inbound = await _sync_messages(token, url, direction="inbound")
    synced += synced_inbound
    matched += matched_inbound

    # --- Fetch outbound (Sent Items) ---
    url_sent = (
        f"{GRAPH_BASE}/me/mailFolders/SentItems/messages?"
        "$top=50&$orderby=receivedDateTime desc"
        "&$select=id,subject,from,toRecipients,bodyPreview,body,receivedDateTime,isRead"
    )
    synced_outbound, matched_outbound = await _sync_messages(token, url_sent, direction="outbound")
    synced += synced_outbound
    matched += matched_outbound

    # Update last_sync_at
    _upsert_sync_config(last_sync_at=datetime.utcnow())

    # Check closure candidates
    flagged = await check_closure_candidates()

    audit_log("EMAIL_SYNC", f"Synced {synced} emails, matched {matched}, flagged {flagged}", current_user.username)

    return {"synced": synced, "matched": matched, "flagged_for_review": flagged}


async def _sync_messages(token: str, url: str, direction: str = "inbound"):
    """Paginate through Graph messages and sync them."""
    synced = 0
    matched = 0

    while url:
        data = _graph_get(token, url)
        messages = data.get("value", [])
        if not messages:
            break

        for msg in messages:
            ms_id = msg.get("id")
            if not ms_id:
                continue

            # Skip already-synced
            existing = execute_query(
                "SELECT id FROM email_messages WHERE ms_message_id = %s",
                (ms_id,),
            )
            if existing:
                continue

            from_obj = msg.get("from", {}).get("emailAddress", {})
            sender_email = from_obj.get("address", "")
            sender_name = from_obj.get("name", "")
            recipients = [
                r.get("emailAddress", {}).get("address", "")
                for r in msg.get("toRecipients", [])
            ]
            # Full body used transiently for AI analysis, NOT stored permanently
            body_text = _normalize_email_text(msg.get("body", {}).get("content") or "", 10000)
            body_preview = _normalize_email_text(msg.get("bodyPreview", "") or "", 200)
            received_at = msg.get("receivedDateTime")

            email_record = {
                "ms_message_id": ms_id,
                "subject": msg.get("subject", ""),
                "sender_email": sender_email,
                "sender_name": sender_name,
                "recipient_emails": json.dumps(recipients),
                "body_preview": body_preview,
                "body_text": body_text,  # kept transiently for matching/AI
                "received_at": received_at,
                "is_read": msg.get("isRead", False),
                "direction": direction,
                "folder": "SentItems" if direction == "outbound" else "Inbox",
            }

            # Match to lead
            lead_id, confidence, method = match_email_to_lead(email_record)
            email_record["matched_lead_id"] = lead_id
            email_record["match_confidence"] = confidence
            email_record["match_method"] = method

            # AI analysis (uses full body transiently)
            lead_data = None
            if lead_id:
                lead_rows = execute_query("SELECT * FROM leads WHERE id = %s", (lead_id,))
                lead_data = lead_rows[0] if lead_rows else None

            ai_result = await analyze_email_with_ai(
                email_body=body_text,
                email_subject=msg.get("subject", ""),
                lead_data=lead_data,
            )
            email_record["ai_summary"] = ai_result.get("summary", "")
            email_record["ai_sentiment"] = ai_result.get("sentiment", "neutral")
            email_record["ai_action_items"] = json.dumps(ai_result.get("action_items", []))
            email_record["ai_ready_to_close"] = (
                ai_result.get("deal_stage") == "closed"
                or ai_result.get("deal_stage") == "closing"
            )
            email_record["ai_close_reasoning"] = (
                f"Deal stage: {ai_result.get('deal_stage', 'unknown')}"
            )
            email_record["processed_at"] = datetime.utcnow()

            # Discard full body before DB insert -- only store body_preview
            del email_record["body_text"]

            # Insert
            cols = list(email_record.keys())
            placeholders = ", ".join(["%s"] * len(cols))
            execute_query(
                f"INSERT INTO email_messages ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(email_record[c] for c in cols),
                fetch=False,
            )
            synced += 1

            # Update lead email intel columns
            if lead_id:
                matched += 1
                execute_query(
                    "UPDATE leads SET email_match_count = email_match_count + 1, "
                    "last_email_activity = %s, email_sentiment = %s WHERE id = %s",
                    (received_at, ai_result.get("sentiment", "neutral"), lead_id),
                    fetch=False,
                )

        # Pagination
        url = data.get("@odata.nextLink")

    return synced, matched


@router.get("/status")
async def get_sync_status(current_user: AdminUser = Depends(get_current_user)):
    """Return sync status overview."""
    config = _get_sync_config()

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

    return {
        "sync_enabled": config.get("sync_enabled", False) if config else False,
        "last_sync": config.get("last_sync_at") if config else None,
        "total_emails": total,
        "matched_emails": matched,
        "pending_reviews": pending,
    }


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
        # Return all recent emails (most recent 50) with lead name joined
        emails = execute_query(
            "SELECT em.id, em.ms_message_id, em.subject, em.sender_email, em.sender_name, "
            "em.recipient_emails, em.body_preview, em.received_at, em.is_read, em.direction, "
            "em.matched_lead_id, em.match_confidence, em.match_method, em.ai_summary, em.ai_sentiment, "
            "em.ai_action_items, em.ai_ready_to_close, em.ai_close_reasoning, em.processed_at, "
            "l.first_name AS lead_first_name, l.last_name AS lead_last_name, l.status AS lead_status "
            "FROM email_messages em "
            "LEFT JOIN leads l ON l.id = em.matched_lead_id "
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
