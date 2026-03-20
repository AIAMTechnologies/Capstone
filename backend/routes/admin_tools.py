import csv
import io
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from auth import AdminUser, get_current_user, pwd_context
from db import execute_query, get_db_connection

router = APIRouter(prefix="/api/admin/tools", tags=["Admin Tools"])


class ColumnMapping(BaseModel):
    csv_column: str
    db_column: str


class CSVImportRequest(BaseModel):
    mappings: List[ColumnMapping]
    data: List[dict]


class MassEmailRequest(BaseModel):
    dealer_ids: List[int]
    subject: str
    body: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/csv-import")
async def csv_import(req: CSVImportRequest, current_user: AdminUser = Depends(get_current_user)):
    """Import leads from CSV with column mapping."""
    if not req.data:
        raise HTTPException(status_code=400, detail="No data provided")

    inserted = 0
    errors = []
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for i, row in enumerate(req.data):
                try:
                    mapped = {}
                    for m in req.mappings:
                        if m.csv_column in row:
                            mapped[m.db_column] = row[m.csv_column]

                    if not mapped:
                        continue

                    # Ensure name field
                    if 'name' not in mapped:
                        fn = mapped.get('first_name', '')
                        ln = mapped.get('last_name', '')
                        if fn or ln:
                            mapped['name'] = f"{fn} {ln}".strip()
                        else:
                            mapped['name'] = f"Import Row {i+1}"

                    mapped['status'] = 'active'

                    columns = ', '.join(mapped.keys())
                    placeholders = ', '.join(['%s'] * len(mapped))
                    cursor.execute(
                        f"INSERT INTO leads ({columns}, created_at) VALUES ({placeholders}, CURRENT_TIMESTAMP)",
                        tuple(mapped.values())
                    )
                    inserted += 1
                except Exception as e:
                    errors.append(f"Row {i+1}: {str(e)}")
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return {"message": f"Imported {inserted} leads", "inserted": inserted, "errors": errors}


@router.post("/csv-upload")
async def csv_upload(file: UploadFile = File(...), current_user: AdminUser = Depends(get_current_user)):
    """Upload CSV file and return headers + preview data for column mapping."""
    content = await file.read()
    text = content.decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    preview = []
    for i, row in enumerate(reader):
        if i >= 5:
            break
        preview.append(dict(row))
    return {"headers": headers, "preview": preview, "total_rows": i + 1}


@router.post("/mass-email")
async def mass_email(req: MassEmailRequest, current_user: AdminUser = Depends(get_current_user)):
    """Send mass email to selected dealers (placeholder - logs the request)."""
    dealers = execute_query("SELECT id, name, email FROM dealers WHERE id = ANY(%s)", (req.dealer_ids,))
    # In production, integrate with email service. For now, log and create notifications.
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for dealer in dealers:
                cursor.execute(
                    """INSERT INTO notifications (dealer_id, notification_type, sent_at, responded)
                    VALUES (%s, 'mass_email', CURRENT_TIMESTAMP, FALSE)""",
                    (dealer['id'],)
                )
            conn.commit()
    finally:
        conn.close()
    return {"message": f"Email queued for {len(dealers)} dealers", "dealers": [d['name'] for d in dealers]}


@router.get("/notification-check")
async def notification_check(current_user: AdminUser = Depends(get_current_user)):
    """Check notification status for all dealers."""
    query = """
        SELECT d.id, d.name, d.email,
            COUNT(n.id) as total_notifications,
            COUNT(CASE WHEN n.responded = TRUE THEN 1 END) as responded,
            MAX(n.sent_at) as last_sent,
            MAX(CASE WHEN n.responded = TRUE THEN n.responded_at END) as last_responded
        FROM dealers d
        LEFT JOIN notifications n ON d.id = n.dealer_id
        WHERE d.is_active = TRUE
        GROUP BY d.id, d.name, d.email
        ORDER BY d.name
    """
    return {"dealers": execute_query(query)}


@router.post("/notification-resend/{dealer_id}")
async def resend_notification(dealer_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Resend notification to a dealer."""
    dealer = execute_query("SELECT id, name FROM dealers WHERE id = %s", (dealer_id,))
    if not dealer:
        raise HTTPException(status_code=404, detail="Dealer not found")
    execute_query(
        """INSERT INTO notifications (dealer_id, notification_type, sent_at, responded)
        VALUES (%s, 'resend', CURRENT_TIMESTAMP, FALSE)""",
        (dealer_id,), fetch=False
    )
    return {"message": f"Notification resent to {dealer[0]['name']}"}


@router.get("/lead-export")
async def lead_export(
    status: Optional[str] = None,
    province: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    """Export leads as JSON (frontend converts to CSV)."""
    conditions = []
    params = []
    if status:
        conditions.append("l.status = %s")
        params.append(status)
    if province:
        conditions.append("l.province = %s")
        params.append(province.upper())
    if start_date:
        conditions.append("l.created_at >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("l.created_at <= %s")
        params.append(end_date)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT l.*, d.name as dealer_name_assigned, rd.name as recommended_dealer_name
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        LEFT JOIN dealers rd ON l.recommended_dealer_id = rd.id
        {where}
        ORDER BY l.created_at DESC
    """
    leads = execute_query(query, tuple(params) if params else None)
    return {"leads": leads, "count": len(leads)}


@router.post("/change-password")
async def change_password(req: ChangePasswordRequest, current_user: AdminUser = Depends(get_current_user)):
    """Change admin password."""
    user = execute_query(
        "SELECT password_hash FROM admin_users WHERE id = %s",
        (current_user.id,)
    )
    if not user or not pwd_context.verify(req.current_password, user[0]['password_hash']):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    new_hash = pwd_context.hash(req.new_password)
    execute_query(
        "UPDATE admin_users SET password_hash = %s WHERE id = %s",
        (new_hash, current_user.id), fetch=False
    )
    return {"message": "Password changed successfully"}
