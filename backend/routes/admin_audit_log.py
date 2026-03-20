from typing import Optional

from fastapi import APIRouter, Depends, Query

from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/admin", tags=["Admin Audit Log"])


@router.get("/audit-log")
async def get_audit_log(
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: AdminUser = Depends(get_current_user),
):
    conditions = ["1=1"]
    params = []

    if event_type:
        conditions.append("event_type = %s")
        params.append(event_type)
    if start_date:
        conditions.append("created_at >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("created_at <= %s")
        params.append(end_date)
    if entity_id:
        conditions.append("entity_id = %s")
        params.append(entity_id)

    where_clause = " AND ".join(conditions)

    rows = execute_query(
        f"""
        SELECT *
        FROM audit_log
        WHERE {where_clause}
        ORDER BY created_at DESC, id DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params + [limit, offset]),
    )
    total = execute_query(
        f"SELECT COUNT(*) AS total FROM audit_log WHERE {where_clause}",
        tuple(params),
    )[0]["total"]

    return {"events": rows, "count": len(rows), "total": total}
