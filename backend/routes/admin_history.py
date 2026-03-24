from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/admin", tags=["Admin History"])


@router.get("/history-leads")
async def get_history_leads(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    province: Optional[str] = None,
    dealer: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: AdminUser = Depends(get_current_user)
):
    """Get Lasso-backed historical leads with filters."""
    conditions = ["1=1"]
    params = []

    if start_date:
        conditions.append("COALESCE(h.submit_date, h.created_date, h.form_submit_date) >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("COALESCE(h.submit_date, h.created_date, h.form_submit_date) <= %s")
        params.append(end_date)
    if province:
        conditions.append("h.province = %s")
        params.append(province.upper())
    if dealer:
        conditions.append("h.dealer_name ILIKE %s")
        params.append(f"%{dealer}%")

    where_clause = " AND ".join(conditions)
    params.extend([limit, offset])

    query = f"""
        SELECT
            h.lasso_lead_id AS id,
            h.name,
            h.first_name,
            h.last_name,
            h.email,
            h.phone,
            h.address,
            h.city,
            h.province,
            h.status_bucket AS status,
            h.dealer_name AS dealer_name_assigned,
            h.project_type,
            h.product_type,
            h.square_footage_value AS square_footage,
            h.square_footage_text AS custom_pick_1,
            h.business_category,
            h.lead_source,
            h.company_name,
            h.comments,
            h.landing_page,
            h.landing_page_url,
            h.landing_page_variant,
            h.utm_source,
            h.utm_medium,
            h.utm_campaign,
            h.utm_content,
            h.value_of_order,
            COALESCE(h.submit_date, h.created_date, h.form_submit_date) AS created_at
        FROM dashboard_history_leads h
        WHERE {where_clause}
        ORDER BY h.last_interaction DESC NULLS LAST, h.created_date DESC NULLS LAST
        LIMIT %s OFFSET %s
    """
    leads = execute_query(query, tuple(params))

    count_params = params[:-2]  # remove limit/offset
    count_query = f"SELECT COUNT(*) as total FROM dashboard_history_leads h WHERE {where_clause}"
    total = execute_query(count_query, tuple(count_params))[0]['total']

    return {"leads": leads, "count": len(leads), "total": total}


@router.post("/leads/{lead_id}/update-value")
async def update_value_of_order(
    lead_id: int,
    value: float,
    current_user: AdminUser = Depends(get_current_user)
):
    raise HTTPException(
        status_code=409,
        detail="Historical leads are read-only in Lasso-only mode.",
    )
