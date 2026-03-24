import hashlib

from fastapi import APIRouter, Depends
from access_control import build_ai_pause_message
from auth import AdminUser, get_current_user
from db import execute_query
from ai_service import ai_client
from cost_control import build_spend_limit_message

router = APIRouter(prefix="/api/ai", tags=["AI Insights"])


def _format_age(days: int) -> str:
    if days >= 730:
        return f"about {round(days / 365)} years"
    if days >= 365:
        return "over 1 year"
    if days >= 60:
        return f"about {round(days / 30)} months"
    return f"{days} days"


def _fallback_insights(
    aging_count: int,
    oldest_unassigned_days: int,
    active_count: int,
    stale_assigned_count: int,
    province_stats: list[dict],
) -> list[dict]:
    insights: list[dict] = []

    if aging_count > 0:
        insights.append({
            "type": "warning",
            "title": f"{aging_count} unassigned leads are aging",
            "body": (
                f"{aging_count} leads are sitting in the unassigned queue. "
                f"The oldest has been waiting {_format_age(oldest_unassigned_days)}."
            ),
            "action": "Work the unassigned queue and clear the oldest leads today.",
        })

    if active_count > 0 and stale_assigned_count > 0:
        stale_pct = round((stale_assigned_count / active_count) * 100)
        insights.append({
            "type": "alert",
            "title": f"{stale_assigned_count} assigned leads look stalled",
            "body": (
                f"{stale_assigned_count} assigned leads have had no Lasso interaction in the last 7 days, "
                f"about {stale_pct}% of the active pipeline."
            ),
            "action": "Have the team review stale assigned leads and push next actions or reassign them.",
        })

    if province_stats:
        best = max(province_stats, key=lambda r: r.get("total") or 0)
        if best.get("province") and (best.get("total") or 0) >= 5:
            insights.append({
                "type": "trend",
                "title": f"{best['province']} has the most active leads",
                "body": (
                    f"{best['province']} has {best['total']} active leads in the current Lasso snapshot."
                ),
                "action": "Ensure adequate dealer coverage in that province.",
            })

    return insights[:4]


@router.get("/insights")
async def get_insights(current_user: AdminUser = Depends(get_current_user)):
    """AI-generated actionable insights from the Lasso live snapshot."""

    # Unassigned leads (all are effectively "new" in the Lasso queue)
    aging_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM dashboard_unassigned_leads "
        "WHERE record_date < NOW() - INTERVAL '48 hours'"
    )
    aging_count = int(aging_rows[0]["cnt"]) if aging_rows else 0

    oldest_rows = execute_query(
        "SELECT COALESCE(MAX(EXTRACT(DAY FROM NOW() - record_date)), 0) AS days "
        "FROM dashboard_unassigned_leads"
    )
    oldest_unassigned_days = int(oldest_rows[0]["days"]) if oldest_rows else 0

    # Active (assigned) leads
    active_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM dashboard_active_leads WHERE dealer_id IS NOT NULL"
    )
    active_count = int(active_rows[0]["cnt"]) if active_rows else 0

    # Stale assigned: no last_interaction in 7 days
    stale_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM dashboard_active_leads "
        "WHERE dealer_id IS NOT NULL "
        "AND (last_interaction IS NULL OR last_interaction < NOW() - INTERVAL '7 days')"
    )
    stale_assigned_count = int(stale_rows[0]["cnt"]) if stale_rows else 0

    # Province breakdown from active leads
    province_stats = execute_query(
        "SELECT CASE "
        "  WHEN UPPER(TRIM(province)) IN ('ONTARIO','ON') THEN 'ON' "
        "  WHEN UPPER(TRIM(province)) IN ('BRITISH COLUMBIA','BC') THEN 'BC' "
        "  WHEN UPPER(TRIM(province)) IN ('ALBERTA','AB') THEN 'AB' "
        "  WHEN UPPER(TRIM(province)) IN ('QUEBEC','QC') THEN 'QC' "
        "  WHEN UPPER(TRIM(province)) IN ('MANITOBA','MB') THEN 'MB' "
        "  WHEN UPPER(TRIM(province)) IN ('SASKATCHEWAN','SK') THEN 'SK' "
        "  WHEN UPPER(TRIM(province)) IN ('NOVA SCOTIA','NS') THEN 'NS' "
        "  WHEN UPPER(TRIM(province)) IN ('NEW BRUNSWICK','NB') THEN 'NB' "
        "  WHEN UPPER(TRIM(province)) IN ('NEWFOUNDLAND AND LABRADOR','NEWFOUNDLAND','NL') THEN 'NL' "
        "  WHEN UPPER(TRIM(province)) IN ('PRINCE EDWARD ISLAND','PEI','PE') THEN 'PE' "
        "  WHEN UPPER(TRIM(province)) IN ('NORTHWEST TERRITORIES','NT') THEN 'NT' "
        "  WHEN UPPER(TRIM(province)) IN ('NUNAVUT','NU') THEN 'NU' "
        "  WHEN UPPER(TRIM(province)) IN ('YUKON','YT') THEN 'YT' "
        "  ELSE UPPER(TRIM(province)) "
        "END AS province, COUNT(*) AS total "
        "FROM dashboard_active_leads "
        "WHERE province IS NOT NULL "
        "GROUP BY 1 ORDER BY total DESC LIMIT 5"
    ) or []

    summary = (
        f"Unassigned leads older than 48h: {aging_count}. "
        f"Oldest unassigned lead age: {oldest_unassigned_days} days. "
        f"Active assigned leads: {active_count}. "
        f"Stale assigned leads (no interaction 7+ days): {stale_assigned_count}. "
    )
    if province_stats:
        summary += "Province breakdown (active): " + ", ".join(
            f"{p['province']}: {p['total']}" for p in province_stats
        )

    cache_key = hashlib.sha1(summary.encode("utf-8")).hexdigest()

    result = ai_client.call_json(
        system=(
            "You are the revenue operations copilot for Window Film Canada. "
            "Generate 3-4 sharp manager-facing insights using the exact numbers provided. "
            "Prioritize backlog risk, regional pipeline, stalled assigned leads, and unassigned queue aging. "
            "Do not repeat the same theme twice, avoid generic phrasing, and make each action specific. "
            "Return JSON: {\"insights\": [{\"type\": \"warning|trend|alert\", \"title\": \"short title\", "
            "\"body\": \"1-2 sentence insight with actual numbers\", \"action\": \"specific action text\"}]}"
        ),
        user=f"Current Lasso snapshot summary:\n{summary}",
        cache_key=f"insights_{cache_key}",
        max_tokens=250,
        temperature=0.2,
        request_timeout=6.0,
        retries=1,
        task_type="reasoning",
    )

    fallback = _fallback_insights(
        aging_count=aging_count,
        oldest_unassigned_days=oldest_unassigned_days,
        active_count=active_count,
        stale_assigned_count=stale_assigned_count,
        province_stats=province_stats,
    )

    if result and result.get("insights"):
        ai_insights = result["insights"][:4]
        if len(ai_insights) >= 3:
            return {"insights": ai_insights}
        return {"insights": (ai_insights + fallback)[:4]}

    if ai_client.last_error_meta.get("type") == "agent_paused":
        return {
            "insights": [
                {
                    "type": "warning",
                    "title": "AI operations paused",
                    "body": build_ai_pause_message(ai_client.last_error_meta),
                    "action": "Resume AI operations in Tools when ready.",
                },
                *(fallback[:3] if fallback else []),
            ][:4]
        }

    if ai_client.last_error_meta.get("type") == "spend_limit":
        return {
            "insights": [
                {
                    "type": "warning",
                    "title": "AI spend limit reached",
                    "body": build_spend_limit_message(ai_client.last_error_meta),
                    "action": "Review AI spend limits on the dashboard before retrying.",
                },
                *(fallback[:3] if fallback else []),
            ][:4]
        }

    return {
        "insights": fallback or [{
            "type": "trend",
            "title": "Pipeline looks healthy",
            "body": "No urgent issues detected in the current Lasso snapshot.",
            "action": "Review dashboard for details.",
        }]
    }


@router.get("/churn-risks")
async def get_churn_risks(current_user: AdminUser = Depends(get_current_user)):
    """Identify active Lasso leads at risk based on inactivity."""
    at_risk = execute_query(
        """
        SELECT
            lasso_lead_id AS lead_id,
            name,
            city,
            province,
            current_status AS status,
            dealer_id AS assigned_dealer_id,
            dealer_name,
            ROUND(
                EXTRACT(EPOCH FROM (NOW() - COALESCE(last_interaction, date_assigned, synced_at)))
                / 86400.0
            )::int AS days_inactive
        FROM dashboard_active_leads
        WHERE last_interaction IS NULL
           OR last_interaction < NOW() - INTERVAL '7 days'
        ORDER BY days_inactive DESC
        LIMIT 20
        """
    ) or []

    results = []
    for lead in at_risk:
        days = lead.get("days_inactive") or 0
        risk_level = "high" if days > 14 else "medium" if days > 7 else "low"
        reason = f"No Lasso interaction in {days} days."
        if not lead.get("assigned_dealer_id"):
            reason += " Lead is unassigned."
            risk_level = "high"
        action = (
            "Reassign to a more responsive dealer"
            if risk_level == "high"
            else "Follow up with assigned dealer"
        )
        results.append({
            "lead_id": lead["lead_id"],
            "name": lead.get("name", ""),
            "risk_level": risk_level,
            "days_inactive": days,
            "reason": reason,
            "recommended_action": action,
            "dealer_name": lead.get("dealer_name"),
        })

    return {"at_risk_leads": results}
