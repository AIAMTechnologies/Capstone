from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/admin", tags=["Admin Logs"])


class CreateLogRequest(BaseModel):
    lead_id: int
    log_type: str
    message: str
    dealer_id: Optional[int] = None


@router.get("/lead-log")
async def get_lead_log(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    logs = execute_query(
        """SELECT ll.*, d.name as dealer_name
        FROM lead_logs ll
        LEFT JOIN dealers d ON ll.dealer_id = d.id
        WHERE ll.lead_id = %s
        ORDER BY ll.created_at DESC""",
        (lead_id,)
    )
    return {"logs": logs, "count": len(logs)}


@router.post("/lead-log")
async def create_lead_log(req: CreateLogRequest, current_user: AdminUser = Depends(get_current_user)):
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (req.lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    execute_query(
        """INSERT INTO lead_logs (lead_id, log_type, message, dealer_id, created_by, created_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)""",
        (req.lead_id, req.log_type, req.message, req.dealer_id, current_user.username),
        fetch=False
    )
    return {"message": "Log entry created"}
