from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth import AdminUser, get_current_user
from db import execute_query, get_db_connection, release_db_connection
from services.lasso_dashboard_sync import (
    get_lasso_dashboard_status,
    start_lasso_dashboard_sync,
)

router = APIRouter(prefix="/api/admin/lasso-dashboard", tags=["Admin Lasso Dashboard"])


class CreateLocalLeadRequest(BaseModel):
    lasso_lead_id: int
    snapshot_type: str


@router.get("/status")
async def lasso_dashboard_status(current_user: AdminUser = Depends(get_current_user)):
    return get_lasso_dashboard_status()


@router.post("/sync")
async def trigger_lasso_dashboard_sync(
    sync_type: str = Query("fast", pattern="^(fast|full)$"),
    current_user: AdminUser = Depends(get_current_user),
):
    result = start_lasso_dashboard_sync(sync_type=sync_type, actor=current_user.username)
    return {
        "message": "Lasso dashboard sync started" if result["started"] else result["reason"],
        **result,
    }


@router.get("/unassigned-leads")
async def get_dashboard_unassigned_leads(current_user: AdminUser = Depends(get_current_user)):
    rows = execute_query(
        """
        SELECT
            lasso_lead_id,
            lead_id,
            first_name,
            last_name,
            name,
            email,
            city,
            province,
            location_text,
            current_status,
            record_date,
            last_interaction,
            synced_at
        FROM dashboard_unassigned_leads
        ORDER BY record_date DESC NULLS LAST, lasso_lead_id DESC
        """
    )
    return {"leads": rows, "count": len(rows)}


@router.get("/active-leads")
async def get_dashboard_active_leads(current_user: AdminUser = Depends(get_current_user)):
    rows = execute_query(
        """
        SELECT
            lasso_lead_id,
            lead_id,
            dealer_id,
            dealer_name,
            first_name,
            last_name,
            name,
            email,
            city,
            province,
            location_text,
            current_status,
            date_assigned,
            last_interaction,
            lead_details,
            synced_at
        FROM dashboard_active_leads
        ORDER BY last_interaction DESC NULLS LAST, date_assigned DESC NULLS LAST, lasso_lead_id DESC
        """
    )
    return {"leads": rows, "count": len(rows)}


@router.get("/snapshot")
async def get_dashboard_snapshot(current_user: AdminUser = Depends(get_current_user)):
    """Combined endpoint: returns sync status + unassigned + active leads on ONE connection."""
    status = get_lasso_dashboard_status()

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT lasso_lead_id, lead_id, first_name, last_name, name, email,
                       city, province, location_text, current_status,
                       record_date, last_interaction, synced_at
                FROM dashboard_unassigned_leads
                ORDER BY record_date DESC NULLS LAST, lasso_lead_id DESC
                """
            )
            unassigned = cur.fetchall()

            cur.execute(
                """
                SELECT lasso_lead_id, lead_id, dealer_id, dealer_name,
                       first_name, last_name, name, email,
                       city, province, location_text, current_status,
                       date_assigned, last_interaction, lead_details, synced_at
                FROM dashboard_active_leads
                ORDER BY last_interaction DESC NULLS LAST, date_assigned DESC NULLS LAST, lasso_lead_id DESC
                """
            )
            active = cur.fetchall()
    finally:
        release_db_connection(conn)

    return {
        "sync_status": status,
        "unassigned": {"leads": unassigned, "count": len(unassigned)},
        "active": {"leads": active, "count": len(active)},
    }


@router.post("/archive-lead")
async def archive_active_lead(
    lasso_lead_id: int = Query(...),
    current_user: AdminUser = Depends(get_current_user),
):
    """Remove a lead from the active dashboard view. The lead still exists in Lasso."""
    rows = execute_query(
        "SELECT lasso_lead_id FROM dashboard_active_leads WHERE lasso_lead_id = %s",
        (lasso_lead_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Lead not found in active dashboard.")
    execute_query(
        "DELETE FROM dashboard_active_leads WHERE lasso_lead_id = %s",
        (lasso_lead_id,),
        fetch=False,
    )
    return {"message": f"Lead #{lasso_lead_id} removed from active dashboard.", "lasso_lead_id": lasso_lead_id}


@router.post("/reassign-lead")
async def reassign_active_lead(
    lasso_lead_id: int = Query(...),
    dealer_id: Optional[int] = Query(None),
    dealer_name: Optional[str] = Query(None),
    current_user: AdminUser = Depends(get_current_user),
):
    """Update the dealer assignment on an active lead in the dashboard snapshot."""
    rows = execute_query(
        "SELECT lasso_lead_id FROM dashboard_active_leads WHERE lasso_lead_id = %s",
        (lasso_lead_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Lead not found in active dashboard.")
    execute_query(
        """
        UPDATE dashboard_active_leads
        SET dealer_id = %s, dealer_name = %s
        WHERE lasso_lead_id = %s
        """,
        (dealer_id, dealer_name, lasso_lead_id),
        fetch=False,
    )
    return {
        "message": f"Lead #{lasso_lead_id} reassigned to {dealer_name or dealer_id}.",
        "lasso_lead_id": lasso_lead_id,
        "dealer_id": dealer_id,
        "dealer_name": dealer_name,
    }


@router.post("/create-local-lead")
async def create_local_lead_from_snapshot(
    request: CreateLocalLeadRequest,
    current_user: AdminUser = Depends(get_current_user),
):
    raise HTTPException(
        status_code=409,
        detail="Local lead creation is disabled in Lasso-only mode.",
    )
