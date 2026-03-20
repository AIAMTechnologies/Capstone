import hashlib

from fastapi import APIRouter, Depends
from access_control import build_ai_pause_message
from auth import AdminUser, get_current_user
from db import execute_query
from ai_service import ai_client
from cost_control import build_spend_limit_message

router = APIRouter(prefix="/api/ai", tags=["AI Insights"])


def _conversion_rate(row: dict) -> float:
    total = row.get('total') or 0
    converted = row.get('converted') or 0
    if not total:
        return 0.0
    return converted / total


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
    follow_up_count: int,
    stale_assigned_count: int,
    province_stats: list[dict],
) -> list[dict]:
    insights: list[dict] = []

    if aging_count > 0:
        insights.append({
            "type": "warning",
            "title": f"{aging_count} unassigned leads are aging",
            "body": (
                f"{aging_count} open leads have been sitting unassigned for more than 48 hours. "
                f"The oldest has been waiting {_format_age(oldest_unassigned_days)}."
            ),
            "action": "Work the unassigned queue first and clear the oldest leads today.",
        })

    if active_count > 0 and stale_assigned_count > 0:
        stale_pct = round((stale_assigned_count / active_count) * 100)
        insights.append({
            "type": "alert",
            "title": f"{stale_assigned_count} assigned leads look stalled",
            "body": (
                f"{stale_assigned_count} assigned leads have no logged activity in the last 7 days, "
                f"which is about {stale_pct}% of the active pipeline."
            ),
            "action": "Have the team review stale assigned leads and push next actions or reassign them.",
        })

    if follow_up_count > 0:
        follow_up_pct = round((follow_up_count / max(active_count, 1)) * 100)
        insights.append({
            "type": "trend",
            "title": f"{follow_up_count} leads are sitting in follow-up",
            "body": (
                f"{follow_up_count} leads are already marked follow-up, representing about "
                f"{follow_up_pct}% of the active pipeline."
            ),
            "action": "Audit follow-up leads for overdue quotes, callbacks, and next-touch dates.",
        })

    provinces_with_volume = [row for row in province_stats if (row.get('total') or 0) >= 100]
    if provinces_with_volume:
        best = max(provinces_with_volume, key=_conversion_rate)
        worst = min(provinces_with_volume, key=_conversion_rate)

        if best.get('province'):
            insights.append({
                "type": "trend",
                "title": f"{best['province']} is converting best",
                "body": (
                    f"{best['province']} is converting at {_conversion_rate(best) * 100:.1f}% "
                    f"from {best['total']} leads."
                ),
                "action": "Review what is working in that region and reuse the same playbook elsewhere.",
            })

        if worst.get('province') and worst.get('province') != best.get('province'):
            insights.append({
                "type": "warning",
                "title": f"{worst['province']} needs attention",
                "body": (
                    f"{worst['province']} is converting at only {_conversion_rate(worst) * 100:.1f}% "
                    f"from {worst['total']} leads."
                ),
                "action": "Check dealer responsiveness, quote speed, and close rates in that province.",
            })

    return insights[:4]


@router.get("/insights")
async def get_insights(current_user: AdminUser = Depends(get_current_user)):
    """AI-generated actionable insights from current lead/dealer data."""
    # Gather aggregations
    aging = execute_query(
        "SELECT COUNT(*) as cnt FROM leads WHERE assigned_dealer_id IS NULL AND status = 'active' AND created_at < CURRENT_TIMESTAMP - INTERVAL '48 hours'"
    )
    aging_count = aging[0]['cnt'] if aging else 0
    oldest_unassigned = execute_query(
        "SELECT COALESCE(MAX(EXTRACT(DAY FROM CURRENT_TIMESTAMP - created_at)), 0) AS days "
        "FROM leads WHERE assigned_dealer_id IS NULL AND status = 'active'"
    )
    oldest_unassigned_days = int(oldest_unassigned[0]['days']) if oldest_unassigned else 0

    pipeline_counts = execute_query("""
        SELECT
            COUNT(*) FILTER (WHERE assigned_dealer_id IS NOT NULL AND status IN ('active', 'follow_up')) AS active_count,
            COUNT(*) FILTER (WHERE status = 'follow_up') AS follow_up_count
        FROM leads
        WHERE status NOT IN ('converted', 'dead', 'archived')
    """)
    active_count = int(pipeline_counts[0]['active_count']) if pipeline_counts else 0
    follow_up_count = int(pipeline_counts[0]['follow_up_count']) if pipeline_counts else 0

    stale_assigned = execute_query("""
        SELECT COUNT(*) AS cnt
        FROM leads l
        WHERE l.assigned_dealer_id IS NOT NULL
        AND l.status IN ('active', 'follow_up')
        AND GREATEST(
            COALESCE((SELECT MAX(created_at) FROM lead_logs WHERE lead_id = l.id), l.created_at),
            COALESCE(l.updated_at, l.created_at)
        ) < CURRENT_TIMESTAMP - INTERVAL '7 days'
    """)
    stale_assigned_count = int(stale_assigned[0]['cnt']) if stale_assigned else 0

    province_stats = execute_query("""
        SELECT province, COUNT(*) as total,
            COUNT(CASE WHEN status = 'converted' THEN 1 END) as converted
        FROM leads WHERE province IS NOT NULL
        GROUP BY province ORDER BY total DESC LIMIT 5
    """)

    summary = (
        f"Aging unassigned leads: {aging_count}. "
        f"Oldest unassigned lead age: {oldest_unassigned_days} days. "
        f"Active assigned leads: {active_count}. "
        f"Follow-up leads: {follow_up_count}. "
        f"Stale assigned leads: {stale_assigned_count}. "
    )
    summary += "Province breakdown: " + ", ".join(
        f"{p['province']}: {p['total']} total/{p['converted']} converted" for p in (province_stats or [])
    )

    cache_key = hashlib.sha1(summary.encode('utf-8')).hexdigest()

    result = ai_client.call_json(
        system=(
            "You are the revenue operations copilot for Window Film Canada. "
            "Generate 3-4 sharp manager-facing insights using the exact numbers provided. "
            "Prioritize backlog risk, regional conversion performance, stalled pipeline, and follow-up load. "
            "Do not repeat the same theme twice, avoid generic phrasing, and make each action specific. "
            "Return JSON: {\"insights\": [{\"type\": \"warning|trend|alert\", \"title\": \"short title\", "
            "\"body\": \"1-2 sentence insight with actual numbers\", \"action\": \"specific action text\"}]}"
        ),
        user=f"Current data summary:\n{summary}",
        cache_key=f"insights_{cache_key}",
        max_tokens=250,
        temperature=0.2,
        request_timeout=6.0,
        retries=1,
    )

    fallback = _fallback_insights(
        aging_count=aging_count,
        oldest_unassigned_days=oldest_unassigned_days,
        active_count=active_count,
        follow_up_count=follow_up_count,
        stale_assigned_count=stale_assigned_count,
        province_stats=province_stats or [],
    )

    if result and result.get('insights'):
        ai_insights = result['insights'][:4]
        if len(ai_insights) >= 3:
            return {"insights": ai_insights}
        combined = ai_insights + fallback
        return {"insights": combined[:4]}

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

    return {"insights": fallback or [{"type": "trend", "title": "Pipeline looks healthy", "body": "No urgent issues were detected in the current dashboard snapshot.", "action": "Review dashboard"}]}


@router.get("/churn-risks")
async def get_churn_risks(current_user: AdminUser = Depends(get_current_user)):
    """Identify leads at risk of churning based on inactivity."""
    at_risk = execute_query("""
        SELECT l.id as lead_id, l.name, l.city, l.province, l.status,
            l.assigned_dealer_id, d.name as dealer_name,
            EXTRACT(DAY FROM CURRENT_TIMESTAMP - COALESCE(
                (SELECT MAX(created_at) FROM lead_logs WHERE lead_id = l.id),
                l.created_at
            )) as days_inactive
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        WHERE l.status IN ('active', 'follow_up')
        AND COALESCE(
            (SELECT MAX(created_at) FROM lead_logs WHERE lead_id = l.id),
            l.created_at
        ) < CURRENT_TIMESTAMP - INTERVAL '7 days'
        ORDER BY days_inactive DESC
        LIMIT 20
    """)

    results = []
    for lead in (at_risk or []):
        days = int(lead.get('days_inactive', 0))
        risk_level = 'high' if days > 14 else 'medium' if days > 7 else 'low'
        reason = f"No interaction in {days} days."
        if not lead.get('assigned_dealer_id'):
            reason += " Lead is unassigned."
            risk_level = 'high'
        action = "Reassign to more responsive dealer" if risk_level == 'high' else "Follow up with assigned dealer"
        results.append({
            "lead_id": lead['lead_id'],
            "name": lead.get('name', ''),
            "risk_level": risk_level,
            "days_inactive": days,
            "reason": reason,
            "recommended_action": action,
            "dealer_name": lead.get('dealer_name')
        })

    return {"at_risk_leads": results}
