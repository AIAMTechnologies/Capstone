from typing import Optional, List
from fastapi import APIRouter, Depends
from auth import AdminUser, get_current_user
from db import execute_query
from ai_service import ai_client

router = APIRouter(prefix="/api/ai", tags=["AI Insights"])


@router.get("/insights")
async def get_insights(current_user: AdminUser = Depends(get_current_user)):
    """AI-generated actionable insights from current lead/dealer data."""
    # Gather aggregations
    aging = execute_query(
        "SELECT COUNT(*) as cnt FROM leads WHERE assigned_dealer_id IS NULL AND status = 'active' AND created_at < CURRENT_TIMESTAMP - INTERVAL '48 hours'"
    )
    aging_count = aging[0]['cnt'] if aging else 0

    province_stats = execute_query("""
        SELECT province, COUNT(*) as total,
            COUNT(CASE WHEN status = 'converted' THEN 1 END) as converted
        FROM leads WHERE province IS NOT NULL
        GROUP BY province ORDER BY total DESC LIMIT 5
    """)

    dealer_response = execute_query("""
        SELECT d.name, COUNT(la.id) as assignments,
            COUNT(CASE WHEN la.responded_at IS NULL AND la.assigned_at < CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as unresponsive
        FROM dealers d
        LEFT JOIN lead_assignments la ON d.id = la.dealer_id
        GROUP BY d.id, d.name
        HAVING COUNT(CASE WHEN la.responded_at IS NULL AND la.assigned_at < CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) > 0
        ORDER BY unresponsive DESC LIMIT 3
    """)

    summary = f"Aging unassigned leads: {aging_count}. "
    summary += "Province breakdown: " + ", ".join(
        f"{p['province']}: {p['total']} total/{p['converted']} converted" for p in (province_stats or [])
    ) + ". "
    summary += "Unresponsive dealers: " + ", ".join(
        f"{d['name']} ({d['unresponsive']} unresponded)" for d in (dealer_response or [])
    )

    result = ai_client.call_json(
        system="You are a business intelligence AI for Window Film Canada. Generate 2-4 actionable insights based on the data. Return JSON: {\"insights\": [{\"type\": \"warning|trend|alert\", \"title\": \"short title\", \"body\": \"1-2 sentence insight\", \"action\": \"suggested action text\"}]}",
        user=f"Current data summary:\n{summary}",
        cache_key=f"insights_{aging_count}_{len(province_stats or [])}"
    )

    if result and 'insights' in result:
        return result

    # Fallback with real data
    insights = []
    if aging_count > 0:
        insights.append({"type": "warning", "title": f"{aging_count} Aging Leads", "body": f"{aging_count} unassigned leads are over 48 hours old.", "action": "View unassigned leads"})
    for d in (dealer_response or []):
        insights.append({"type": "alert", "title": f"Unresponsive: {d['name']}", "body": f"{d['name']} has {d['unresponsive']} unresponded assignments this week.", "action": f"Review dealer"})
    return {"insights": insights or [{"type": "trend", "title": "All Clear", "body": "No urgent issues detected.", "action": "View dashboard"}]}


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
