from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/dealer", tags=["Dealer Portal"])


class InteractionRequest(BaseModel):
    lead_id: int
    message: str


class WinLostRequest(BaseModel):
    lead_id: int
    status: str  # 'won' or 'lost'
    value_of_order: Optional[float] = None
    reason: Optional[str] = None


@router.get("/active-leads")
async def dealer_active_leads(current_user: AdminUser = Depends(get_current_user)):
    """Get active leads assigned to the current dealer (matched by email)."""
    query = """
        SELECT l.*, d.name as dealer_name_assigned
        FROM leads l
        JOIN dealers d ON l.assigned_dealer_id = d.id
        WHERE d.email = %s AND l.status NOT IN ('converted', 'dead', 'archived')
        ORDER BY l.created_at DESC
    """
    leads = execute_query(query, (current_user.email,))
    return {"leads": leads, "count": len(leads)}


@router.get("/history")
async def dealer_history(current_user: AdminUser = Depends(get_current_user)):
    """Get historical leads for the current dealer."""
    query = """
        SELECT l.*, d.name as dealer_name_assigned
        FROM leads l
        JOIN dealers d ON l.assigned_dealer_id = d.id
        WHERE d.email = %s AND l.status IN ('converted', 'dead', 'archived')
        ORDER BY l.updated_at DESC NULLS LAST
    """
    leads = execute_query(query, (current_user.email,))
    return {"leads": leads, "count": len(leads)}


@router.post("/submit-interaction")
async def submit_interaction(req: InteractionRequest, current_user: AdminUser = Depends(get_current_user)):
    """Dealer submits an interaction/note for a lead."""
    execute_query(
        """INSERT INTO lead_logs (lead_id, log_type, message, created_by, created_at)
        VALUES (%s, 'dealer_interaction', %s, %s, CURRENT_TIMESTAMP)""",
        (req.lead_id, req.message, current_user.username), fetch=False
    )
    # Update response time in assignment
    execute_query(
        """UPDATE lead_assignments
        SET responded_at = CURRENT_TIMESTAMP,
            response_time_hours = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - assigned_at)) / 3600,
            status = 'responded'
        WHERE lead_id = %s AND dealer_id = (SELECT id FROM dealers WHERE email = %s)
        AND responded_at IS NULL""",
        (req.lead_id, current_user.email), fetch=False
    )
    return {"message": "Interaction submitted"}


@router.post("/submit-win")
async def submit_win(req: WinLostRequest, current_user: AdminUser = Depends(get_current_user)):
    """Dealer reports a won or lost lead."""
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (req.lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    new_status = 'converted' if req.status == 'won' else 'dead'
    execute_query(
        "UPDATE leads SET status = %s, value_of_order = COALESCE(%s, value_of_order), updated_at = CURRENT_TIMESTAMP WHERE id = %s",
        (new_status, req.value_of_order, req.lead_id), fetch=False
    )

    msg = f"Lead marked as {req.status}"
    if req.reason:
        msg += f": {req.reason}"
    execute_query(
        """INSERT INTO lead_logs (lead_id, log_type, message, created_by, created_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)""",
        (req.lead_id, f'lead_{req.status}', msg, current_user.username), fetch=False
    )

    # Update assignment status
    execute_query(
        """UPDATE lead_assignments SET status = %s
        WHERE lead_id = %s AND dealer_id = (SELECT id FROM dealers WHERE email = %s)""",
        (req.status, req.lead_id, current_user.email), fetch=False
    )

    return {"message": f"Lead marked as {req.status}", "lead_id": req.lead_id}
