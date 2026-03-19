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
    """Get historical leads with filters."""
    conditions = ["l.status IN ('converted', 'dead', 'archived')"]
    params = []

    if start_date:
        conditions.append("l.created_at >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("l.created_at <= %s")
        params.append(end_date)
    if province:
        conditions.append("l.province = %s")
        params.append(province.upper())
    if dealer:
        conditions.append("(d.name ILIKE %s OR l.final_installer_selection ILIKE %s)")
        params.extend([f"%{dealer}%", f"%{dealer}%"])

    where_clause = " AND ".join(conditions)
    params.extend([limit, offset])

    query = f"""
        SELECT l.*, d.name as dealer_name_assigned, i.name as installer_name
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        LEFT JOIN installers i ON l.assigned_installer_id = i.id
        WHERE {where_clause}
        ORDER BY l.updated_at DESC NULLS LAST, l.created_at DESC
        LIMIT %s OFFSET %s
    """
    leads = execute_query(query, tuple(params))

    # Count
    count_params = params[:-2]  # remove limit/offset
    count_query = f"SELECT COUNT(*) as total FROM leads l LEFT JOIN dealers d ON l.assigned_dealer_id = d.id WHERE {where_clause}"
    total = execute_query(count_query, tuple(count_params))[0]['total']

    return {"leads": leads, "count": len(leads), "total": total}


@router.post("/leads/{lead_id}/update-value")
async def update_value_of_order(
    lead_id: int,
    value: float,
    current_user: AdminUser = Depends(get_current_user)
):
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    execute_query(
        "UPDATE leads SET value_of_order = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
        (value, lead_id), fetch=False
    )
    return {"message": "Value updated", "lead_id": lead_id, "value_of_order": value}
