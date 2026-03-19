from typing import Optional
from fastapi import APIRouter, Depends
from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/admin/reports", tags=["Admin Reports"])


# ============================================
# TAB 1: Overall Summary (timeframe table)
# ============================================

@router.get("/lead-report")
async def lead_report(current_user: AdminUser = Depends(get_current_user)):
    """
    Returns lead summary for each timeframe row:
    This Month, Last Month, Last 3 Months, Last 6 Months, Last 12 Months, All Time
    """
    timeframes = [
        ("This Month", "date_trunc('month', CURRENT_DATE)", "CURRENT_DATE"),
        ("Last Month", "date_trunc('month', CURRENT_DATE) - INTERVAL '1 month'", "date_trunc('month', CURRENT_DATE)"),
        ("Last 3 Months", "CURRENT_DATE - INTERVAL '3 months'", None),
        ("Last 6 Months", "CURRENT_DATE - INTERVAL '6 months'", None),
        ("Last 12 Months", "CURRENT_DATE - INTERVAL '12 months'", None),
        ("All Time", None, None),
    ]

    results = []
    for label, start_expr, end_expr in timeframes:
        if start_expr and end_expr:
            where = f"WHERE created_at >= {start_expr} AND created_at < {end_expr}"
        elif start_expr:
            where = f"WHERE created_at >= {start_expr}"
        else:
            where = ""

        query = f"""
            SELECT
                COUNT(*) as total_leads,
                COUNT(CASE WHEN status = 'converted' THEN 1 END) as converted,
                COUNT(CASE WHEN status = 'dead' THEN 1 END) as dead,
                COUNT(CASE WHEN status IN ('active', 'follow_up') THEN 1 END) as active
            FROM leads {where}
        """
        row = execute_query(query)
        if row:
            r = row[0]
            total = r["total_leads"] or 0
            results.append({
                "timeframe": label,
                "total_leads": total,
                "converted": r["converted"] or 0,
                "dead": r["dead"] or 0,
                "active": r["active"] or 0,
                "converted_pct": round((r["converted"] or 0) / total * 100, 1) if total > 0 else 0,
                "dead_pct": round((r["dead"] or 0) / total * 100, 1) if total > 0 else 0,
                "active_pct": round((r["active"] or 0) / total * 100, 1) if total > 0 else 0,
            })

    return {"data": results}


@router.get("/lead-graph")
async def lead_graph(tf: int = 6, opt: int = 1, current_user: AdminUser = Depends(get_current_user)):
    """Graph data: leads over time. opt=1 monthly, opt=2 weekly."""
    mapping = {1: "1 month", 3: "3 months", 6: "6 months", 12: "12 months"}
    interval = mapping.get(tf, "100 years")
    where = f"WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '{interval}'" if tf > 0 else ""
    trunc = "week" if opt == 2 else "month"

    query = f"""
        SELECT
            DATE_TRUNC('{trunc}', created_at) as period,
            COUNT(*) as total,
            COUNT(CASE WHEN status = 'converted' THEN 1 END) as converted,
            COUNT(CASE WHEN status = 'dead' THEN 1 END) as dead
        FROM leads {where}
        GROUP BY period
        ORDER BY period
    """
    return {"data": execute_query(query)}


# ============================================
# TAB 2: Dealer Performance (Report 1)
# ============================================

@router.get("/dealer-performance")
async def dealer_performance(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """
    Report 1: All dealers with Active/Converted/Dead counts + Avg Response Time.
    Uses final_installer_selection (dealer name stored on leads) for matching since
    not all leads have assigned_dealer_id set.
    """
    source_filter = ""
    params = None
    if source:
        source_filter = "AND (l.lead_source = %s OR l.source = %s)"
        params = (source, source)

    query = f"""
        SELECT
            d.id as dealer_id,
            d.name as dealer_name,
            COUNT(CASE WHEN l.status IN ('active', 'follow_up') THEN 1 END) as active_leads,
            COUNT(CASE WHEN l.status = 'converted' THEN 1 END) as converted,
            COUNT(CASE WHEN l.status = 'dead' THEN 1 END) as dead,
            COUNT(l.id) as total_leads
        FROM dealers d
        LEFT JOIN leads l ON (l.assigned_dealer_id = d.id
            OR LOWER(TRIM(l.final_installer_selection)) = LOWER(TRIM(d.name)))
            {source_filter}
        GROUP BY d.id, d.name
        ORDER BY d.name
    """
    rows = execute_query(query, params)

    # Avg response time: calculate from lead_assignments if available,
    # otherwise from time between lead created_at and assigned_at
    resp_query = """
        SELECT
            d.id as dealer_id,
            AVG(EXTRACT(EPOCH FROM (COALESCE(l.assigned_at, l.created_at + INTERVAL '1 day') - l.created_at)) / 3600) as avg_response_hours
        FROM dealers d
        JOIN leads l ON (l.assigned_dealer_id = d.id
            OR LOWER(TRIM(l.final_installer_selection)) = LOWER(TRIM(d.name)))
        WHERE l.assigned_at IS NOT NULL
        GROUP BY d.id
    """
    resp_rows = execute_query(resp_query)
    resp_map = {r["dealer_id"]: r["avg_response_hours"] for r in resp_rows} if resp_rows else {}

    result = []
    for r in rows:
        avg_hrs = resp_map.get(r["dealer_id"])
        if avg_hrs is not None:
            hours = int(avg_hrs)
            minutes = int((avg_hrs - hours) * 60)
            avg_response_str = f"{hours} hours, {minutes} minutes" if hours > 0 else f"{minutes} minutes"
        else:
            avg_response_str = ""

        result.append({
            "dealer_id": r["dealer_id"],
            "dealer_name": r["dealer_name"],
            "active_leads": r["active_leads"] or 0,
            "converted": r["converted"] or 0,
            "dead": r["dead"] or 0,
            "total_leads": r["total_leads"] or 0,
            "avg_response_hours": float(avg_hrs) if avg_hrs else None,
            "avg_response_str": avg_response_str,
        })

    return {"data": result}


# ============================================
# TAB 3: Project Size Breakdown (Report 2)
# ============================================

@router.get("/dealer-projects")
async def dealer_projects(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """
    Report 2: Project size breakdown by dealer.
    Categories: 1-499, 500-999, 1000-3499, 3500-7499, 7500-19999, 20000+
    Uses custom_pick_1 field which stores the raw square footage text from CSV.
    """
    source_filter = ""
    params = None
    if source:
        source_filter = "AND (l.lead_source = %s OR l.source = %s)"
        params = (source, source)

    query = f"""
        SELECT
            d.id as dealer_id,
            d.name as dealer_name,
            COUNT(CASE WHEN l.custom_pick_1 ILIKE '%%1 - 499%%' THEN 1 END) as sqft_1_499,
            COUNT(CASE WHEN l.custom_pick_1 ILIKE '%%500 - 999%%' THEN 1 END) as sqft_500_999,
            COUNT(CASE WHEN l.custom_pick_1 ILIKE '%%1,000%%' OR l.custom_pick_1 ILIKE '%%1000%%3499%%' OR l.custom_pick_1 ILIKE '%%1000%%3,499%%' OR l.custom_pick_1 ILIKE '%%1,000 - 3,499%%' THEN 1 END) as sqft_1000_3499,
            COUNT(CASE WHEN l.custom_pick_1 ILIKE '%%3,500%%' OR l.custom_pick_1 ILIKE '%%3500%%7499%%' OR l.custom_pick_1 ILIKE '%%3,500 - 7,499%%' THEN 1 END) as sqft_3500_7499,
            COUNT(CASE WHEN l.custom_pick_1 ILIKE '%%7,500%%' OR l.custom_pick_1 ILIKE '%%7500%%19999%%' OR l.custom_pick_1 ILIKE '%%7,500 - 19,999%%' THEN 1 END) as sqft_7500_19999,
            COUNT(CASE WHEN l.custom_pick_1 ILIKE '%%20,000%%' OR l.custom_pick_1 ILIKE '%%20000%%' THEN 1 END) as sqft_20000_plus,
            COUNT(l.id) as total_leads
        FROM dealers d
        LEFT JOIN leads l ON (l.assigned_dealer_id = d.id
            OR LOWER(TRIM(l.final_installer_selection)) = LOWER(TRIM(d.name)))
            {source_filter}
        GROUP BY d.id, d.name
        HAVING COUNT(l.id) > 0
        ORDER BY d.name
    """
    return {"data": execute_query(query, params)}


# ============================================
# TAB 4: Lead Status & Conversion Scores (Report 3)
# ============================================

@router.get("/lead-status")
async def lead_status_report(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """
    Report 3: Lead status & conversion scores per dealer.
    Shows Client Reviewing/Undecided, Client Building Budget, Converted Total $, Lead Score %.
    """
    source_filter = ""
    params = None
    if source:
        source_filter = "AND (l.lead_source = %s OR l.source = %s)"
        params = (source, source)

    query = f"""
        SELECT
            d.id as dealer_id,
            d.name as dealer_name,
            COUNT(CASE WHEN l.status IN ('active', 'follow_up') THEN 1 END) as reviewing_undecided,
            0 as building_budget,
            COALESCE(SUM(CASE WHEN l.status = 'converted' THEN l.value_of_order END), 0) as converted_total_value,
            COUNT(l.id) as total_leads,
            COUNT(CASE WHEN l.status = 'converted' THEN 1 END) as converted_count,
            CASE WHEN COUNT(l.id) > 0
                THEN ROUND(COUNT(CASE WHEN l.status = 'converted' THEN 1 END)::NUMERIC / COUNT(l.id) * 100, 0)
                ELSE 0
            END as lead_score_pct
        FROM dealers d
        LEFT JOIN leads l ON (l.assigned_dealer_id = d.id
            OR LOWER(TRIM(l.final_installer_selection)) = LOWER(TRIM(d.name)))
            {source_filter}
        GROUP BY d.id, d.name
        HAVING COUNT(l.id) > 0
        ORDER BY lead_score_pct DESC
    """
    return {"data": execute_query(query, params)}


# ============================================
# Legacy endpoints for backward compatibility
# ============================================

@router.get("/response-times")
async def response_times(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """Legacy: Average response times by dealer."""
    return await dealer_performance(source, current_user)
