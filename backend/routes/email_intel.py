import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from urllib.parse import urlencode

import requests as http_requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import AdminUser, get_current_user
from db import execute_query

logger = logging.getLogger("lead_allocation")

router = APIRouter(prefix="/api/email-intel", tags=["Email Intelligence"])

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
MS_LOGIN_BASE = "https://login.microsoftonline.com"
OAUTH_SCOPES = "Mail.Read Mail.ReadWrite offline_access User.Read"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OAuthConfigIn(BaseModel):
    ms_tenant_id: str
    ms_client_id: str
    ms_client_secret: str
    ms_redirect_uri: str


class OAuthCallbackIn(BaseModel):
    code: str


# ---------------------------------------------------------------------------
# Helpers – DB convenience wrappers
# ---------------------------------------------------------------------------

def _get_sync_config() -> Optional[dict]:
    rows = execute_query("SELECT * FROM email_sync_config ORDER BY id DESC LIMIT 1")
    return rows[0] if rows else None


def _upsert_sync_config(**kwargs):
    """Insert or update the single sync-config row."""
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
# AI helpers (Anthropic / Claude)
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, AI analysis unavailable")
            return None
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.error("anthropic package not installed")
        return None


async def analyze_email_with_ai(
    email_body: str, email_subject: str, lead_data: dict = None
) -> dict:
    """
    Call Claude API (claude-sonnet-4-20250514) to analyse email content.

    Returns dict with keys: summary, sentiment, action_items,
    is_deal_related, deal_stage.
    """
    client = _get_anthropic_client()
    if client is None:
        return {
            "summary": "",
            "sentiment": "neutral",
            "action_items": [],
            "is_deal_related": False,
            "deal_stage": "inquiry",
        }

    lead_context = ""
    if lead_data:
        lead_context = (
            f"\n\nRelated lead context:\n"
            f"Name: {lead_data.get('first_name', '')} {lead_data.get('last_name', '')}\n"
            f"Email: {lead_data.get('email', '')}\n"
            f"Phone: {lead_data.get('phone', '')}\n"
            f"Status: {lead_data.get('status', '')}\n"
            f"Province: {lead_data.get('province', '')}\n"
        )

    system_prompt = (
        "You are an expert sales-email analyst. Analyse the email and return "
        "a JSON object with exactly these keys:\n"
        '- "summary": a brief 1-2 sentence summary\n'
        '- "sentiment": one of "positive", "neutral", "negative"\n'
        '- "action_items": an array of short action-item strings\n'
        '- "is_deal_related": boolean\n'
        '- "deal_stage": one of "inquiry", "quoting", "negotiation", "closing", "closed"\n'
        "Return ONLY valid JSON, no markdown."
    )

    user_prompt = (
        f"Subject: {email_subject}\n\n"
        f"Body:\n{email_body[:3000]}"
        f"{lead_context}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        content = response.content[0].text
        return json.loads(content)
    except json.JSONDecodeError:
        logger.error("AI returned non-JSON for email analysis")
        return {
            "summary": "",
            "sentiment": "neutral",
            "action_items": [],
            "is_deal_related": False,
            "deal_stage": "inquiry",
        }
    except Exception as e:
        logger.error(f"AI email analysis failed: {e}")
        return {
            "summary": "",
            "sentiment": "neutral",
            "action_items": [],
            "is_deal_related": False,
            "deal_stage": "inquiry",
        }


# ---------------------------------------------------------------------------
# Lead matching
# ---------------------------------------------------------------------------

def match_email_to_lead(email_record: dict):
    """
    Match priority:
    1. Exact email address match (sender/recipient email = lead.email or lead.dealer_email)
    2. Name match (sender_name contains lead.first_name + lead.last_name)
    3. Phone match (email body contains lead.phone)
    4. AI match (use Claude to analyse email content against unmatched leads)

    Returns: (lead_id, confidence, method) or (None, 0, None)
    """
    sender = (email_record.get("sender_email") or "").lower().strip()
    sender_name = (email_record.get("sender_name") or "").lower().strip()
    body = (email_record.get("body_text") or "") + " " + (email_record.get("body_preview") or "")
    recipient_raw = email_record.get("recipient_emails") or "[]"
    try:
        recipients = json.loads(recipient_raw) if isinstance(recipient_raw, str) else recipient_raw
    except json.JSONDecodeError:
        recipients = []
    all_emails = [sender] + [r.lower().strip() for r in recipients if r]

    # --- 1. Exact email match ---
    if all_emails:
        placeholders = ", ".join(["%s"] * len(all_emails))
        rows = execute_query(
            f"SELECT id FROM leads WHERE LOWER(email) IN ({placeholders}) LIMIT 1",
            tuple(all_emails),
        )
        if rows:
            return (rows[0]["id"], 1.0, "email")

    # --- 2. Name match ---
    if sender_name:
        parts = sender_name.split()
        if len(parts) >= 2:
            rows = execute_query(
                "SELECT id FROM leads WHERE LOWER(first_name) = %s AND LOWER(last_name) = %s LIMIT 1",
                (parts[0], parts[-1]),
            )
            if rows:
                return (rows[0]["id"], 0.85, "name")

    # --- 3. Phone match ---
    if body:
        leads_with_phone = execute_query(
            "SELECT id, phone FROM leads WHERE phone IS NOT NULL AND phone != ''"
        )
        for lead in leads_with_phone or []:
            phone = (lead.get("phone") or "").strip()
            if phone and len(phone) >= 7 and phone in body:
                return (lead["id"], 0.75, "phone")

    # --- 4. AI match (lightweight – compare subject against recent active leads) ---
    client = _get_anthropic_client()
    if client and (email_record.get("subject") or body):
        active_leads = execute_query(
            "SELECT id, first_name, last_name, email, phone, province "
            "FROM leads WHERE status = 'active' ORDER BY created_at DESC LIMIT 20"
        )
        if active_leads:
            leads_text = "\n".join(
                f"ID={l['id']} Name={l.get('first_name','')} {l.get('last_name','')} "
                f"Email={l.get('email','')} Phone={l.get('phone','')} Province={l.get('province','')}"
                for l in active_leads
            )
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=256,
                    system=(
                        "You match emails to leads. Return JSON: "
                        '{"lead_id": <int or null>, "confidence": <0-1>}. '
                        "Return ONLY valid JSON."
                    ),
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Email subject: {email_record.get('subject','')}\n"
                                f"Sender: {sender} ({sender_name})\n"
                                f"Preview: {(email_record.get('body_preview') or '')[:500]}\n\n"
                                f"Leads:\n{leads_text}"
                            ),
                        }
                    ],
                )
                result = json.loads(resp.content[0].text)
                lid = result.get("lead_id")
                conf = result.get("confidence", 0)
                if lid and conf >= 0.6:
                    return (int(lid), float(conf), "ai")
            except Exception as e:
                logger.warning(f"AI lead matching failed: {e}")

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

        # Confirm with AI
        client = _get_anthropic_client()
        reasoning = "Positive email sentiment with extended inactivity."
        if client:
            summaries = "\n".join(
                f"- {e.get('ai_summary', 'No summary')}" for e in latest
            )
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    system=(
                        "You are a sales-ops assistant. Based on recent email "
                        "summaries for a lead, determine if the deal appears "
                        "complete and the lead can be closed. Return JSON: "
                        '{"should_flag": true/false, "reasoning": "..."}'
                    ),
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Lead: {cand.get('first_name','')} {cand.get('last_name','')}\n"
                                f"Status: {cand.get('status','')}\n"
                                f"Days since last email: "
                                f"{(datetime.utcnow() - cand['last_email_at']).days if cand.get('last_email_at') else '?'}\n"
                                f"Recent email summaries:\n{summaries}"
                            ),
                        }
                    ],
                )
                ai_result = json.loads(resp.content[0].text)
                if not ai_result.get("should_flag", False):
                    continue
                reasoning = ai_result.get("reasoning", reasoning)
            except Exception as e:
                logger.warning(f"AI closure confirmation failed: {e}")

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
    return {
        "configured": True,
        "ms_tenant_id": config.get("ms_tenant_id"),
        "ms_client_id": config.get("ms_client_id"),
        "ms_redirect_uri": config.get("ms_redirect_uri"),
        "sync_enabled": config.get("sync_enabled", False),
        "sync_interval_minutes": config.get("sync_interval_minutes", 15),
        "last_sync_at": config.get("last_sync_at"),
        "user_email": config.get("user_email"),
    }


@router.post("/config")
async def save_config(
    body: OAuthConfigIn,
    current_user: AdminUser = Depends(get_current_user),
):
    """Save / update MS Graph OAuth config."""
    _upsert_sync_config(
        ms_tenant_id=body.ms_tenant_id,
        ms_client_id=body.ms_client_id,
        ms_client_secret=body.ms_client_secret,
        ms_redirect_uri=body.ms_redirect_uri,
    )
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
    return {"auth_url": auth_url}


@router.post("/oauth/callback")
async def oauth_callback(
    body: OAuthCallbackIn,
    current_user: AdminUser = Depends(get_current_user),
):
    """Exchange authorisation code for tokens."""
    config = _get_sync_config()
    if not config:
        raise HTTPException(status_code=400, detail="OAuth config not set.")

    resp = http_requests.post(
        f"{MS_LOGIN_BASE}/{config['ms_tenant_id']}/oauth2/v2.0/token",
        data={
            "client_id": config["ms_client_id"],
            "client_secret": config["ms_client_secret"],
            "code": body.code,
            "redirect_uri": config["ms_redirect_uri"],
            "grant_type": "authorization_code",
            "scope": OAUTH_SCOPES,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {resp.text}")

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
# Email Sync
# ---------------------------------------------------------------------------

@router.post("/sync")
async def sync_emails(current_user: AdminUser = Depends(get_current_user)):
    """Manually trigger an email sync from MS Graph."""
    config = _get_sync_config()
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
            body_text = (msg.get("body", {}).get("content") or "")[:10000]
            body_preview = msg.get("bodyPreview", "")
            received_at = msg.get("receivedDateTime")

            email_record = {
                "ms_message_id": ms_id,
                "subject": msg.get("subject", ""),
                "sender_email": sender_email,
                "sender_name": sender_name,
                "recipient_emails": json.dumps(recipients),
                "body_preview": body_preview,
                "body_text": body_text,
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

            # AI analysis
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
    emails = execute_query(
        "SELECT id, ms_message_id, subject, sender_email, sender_name, "
        "recipient_emails, body_preview, received_at, is_read, direction, "
        "match_confidence, match_method, ai_summary, ai_sentiment, "
        "ai_action_items, ai_ready_to_close, ai_close_reasoning, processed_at "
        "FROM email_messages WHERE matched_lead_id = %s "
        "ORDER BY received_at DESC",
        (lead_id,),
    )
    return emails or []


@router.get("/lead/{lead_id}/context")
async def get_lead_context(
    lead_id: int,
    current_user: AdminUser = Depends(get_current_user),
):
    """AI-generated context summary for a lead based on all matched emails."""
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

    client = _get_anthropic_client()
    if not client:
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
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "You are a sales intelligence assistant. Given email history for "
                "a lead, produce a JSON object with:\n"
                '- "context": a comprehensive narrative summary of all interactions\n'
                '- "key_insights": array of important observations\n'
                '- "recommended_action": what the sales team should do next\n'
                "Return ONLY valid JSON."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Lead: {lead.get('first_name','')} {lead.get('last_name','')}\n"
                        f"Email: {lead.get('email','')}\n"
                        f"Status: {lead.get('status','')}\n"
                        f"Province: {lead.get('province','')}\n\n"
                        f"Email history ({len(emails)} messages):\n{email_summaries}"
                    ),
                }
            ],
        )
        result = json.loads(resp.content[0].text)
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
        SELECT crq.id AS review_id, crq.lead_id, crq.flagged_at,
               crq.ai_reasoning, crq.days_inactive, crq.last_email_at,
               crq.email_count, crq.status,
               l.first_name, l.last_name, l.email AS lead_email,
               l.phone, l.province, l.status AS lead_status,
               l.email_match_count, l.last_email_activity, l.email_sentiment
        FROM closure_review_queue crq
        JOIN leads l ON l.id = crq.lead_id
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
        "INSERT INTO lead_logs (lead_id, action, details, performed_by) "
        "VALUES (%s, %s, %s, %s)",
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

    return {"success": True, "lead_id": review["lead_id"]}
