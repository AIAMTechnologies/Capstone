from typing import Optional
from fastapi import APIRouter, Depends
from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/admin/reports", tags=["Admin Reports"])


def _lead_report_fallback() -> list[dict]:
    return []


def _lead_graph_fallback(tf: int, opt: int) -> list[dict]:
    return []


# ============================================
# TAB 1: Overall Summary (timeframe table)
# ============================================

@router.get("/lead-report")
async def lead_report(current_user: AdminUser = Depends(get_current_user)):
    """
    Returns lead summary for each timeframe row:
    This Month, Last Month, Last 3 Months, Last 6 Months, Last 12 Months, All Time
    """
    rows = execute_query(
        """
        SELECT
            timeframe,
            total_leads,
            converted_count AS converted,
            dead_count AS dead,
            active_count AS active,
            converted_pct,
            dead_pct,
            active_pct
        FROM dashboard_dealer_lead_reporting
        ORDER BY CASE timeframe
            WHEN 'This Month' THEN 1
            WHEN 'Last Month' THEN 2
            WHEN 'Last 3 Months' THEN 3
            WHEN 'Last 6 Months' THEN 4
            WHEN 'Last 12 Months' THEN 5
            WHEN 'All Time' THEN 6
            ELSE 99
        END
        """
    )
    return {"data": rows or _lead_report_fallback()}


@router.get("/lead-graph")
async def lead_graph(tf: int = 6, opt: int = 1, current_user: AdminUser = Depends(get_current_user)):
    """Graph data: leads over time. opt=1 monthly, opt=2 weekly."""
    rows = execute_query(
        """
        SELECT
            period_start AS period,
            total_count AS total,
            converted_count AS converted,
            dead_count AS dead
        FROM dashboard_lead_trends
        ORDER BY period_start
        """
    )
    return {"data": rows or _lead_graph_fallback(tf, opt)}


# ============================================
# TAB 2: Dealer Performance (Report 1)
# ============================================

@router.get("/dealer-performance")
async def dealer_performance(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """
    Report 1: All dealers with Active/Converted/Dead counts + Avg Response Time.
    Uses final_dealer_selection (dealer name stored on leads) for matching since
    not all leads have assigned_dealer_id set.
    """
    rows = execute_query(
        """
        SELECT
            dealer_id,
            dealer_name,
            active_leads,
            converted_count AS converted,
            dead_count AS dead,
            total_leads,
            avg_response_hours,
            avg_response_str
        FROM dashboard_dealer_performance
        ORDER BY dealer_name
        """
    )
    return {"data": rows}


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
    rows = execute_query(
        """
        SELECT
            dealer_id,
            dealer_name,
            sqft_1_499,
            sqft_500_999,
            sqft_1000_3499,
            sqft_3500_7499,
            sqft_7500_19999,
            sqft_20000_plus,
            total_leads
        FROM dashboard_dealer_project_breakdown
        ORDER BY dealer_name
        """
    )
    return {"data": rows}


# ============================================
# TAB 4: Lead Status & Conversion Scores (Report 3)
# ============================================

@router.get("/lead-status")
async def lead_status_report(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """
    Report 3: Lead status & conversion scores per dealer.
    Shows Client Reviewing/Undecided, Client Building Budget, Converted Total $, Lead Score %.
    """
    rows = execute_query(
        """
        SELECT
            s.dealer_id,
            s.dealer_name,
            s.reviewing_undecided,
            s.building_budget,
            s.converted_total_value,
            COALESCE(p.total_leads, 0) AS total_leads,
            COALESCE(p.converted_count, 0) AS converted_count,
            ROUND(s.lead_score_pct::numeric, 0) AS lead_score_pct
        FROM dashboard_dealer_status s
        LEFT JOIN dashboard_dealer_performance p
            ON p.dealer_id = s.dealer_id
        ORDER BY s.lead_score_pct DESC, s.dealer_name
        """
    )
    return {"data": rows}


# ============================================
# Legacy endpoints for backward compatibility
# ============================================

@router.get("/response-times")
async def response_times(source: Optional[str] = None, current_user: AdminUser = Depends(get_current_user)):
    """Legacy: Average response times by dealer."""
    return await dealer_performance(source, current_user)
