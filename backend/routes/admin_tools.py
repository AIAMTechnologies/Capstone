from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from auth import AdminUser, get_current_user, pwd_context
from db import execute_query

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
    """CSV import is retired in Lasso-only mode."""
    raise HTTPException(status_code=409, detail="CSV import is disabled in Lasso-only mode")


@router.post("/csv-upload")
async def csv_upload(file: UploadFile = File(...), current_user: AdminUser = Depends(get_current_user)):
    """CSV upload is retired in Lasso-only mode."""
    raise HTTPException(status_code=409, detail="CSV upload is disabled in Lasso-only mode")


@router.post("/mass-email")
async def mass_email(req: MassEmailRequest, current_user: AdminUser = Depends(get_current_user)):
    """Mass email is retired in Lasso-only mode."""
    raise HTTPException(status_code=409, detail="Mass email is disabled in Lasso-only mode")


@router.get("/notification-check")
async def notification_check(current_user: AdminUser = Depends(get_current_user)):
    """Notification tracking is retired in Lasso-only mode."""
    return {"dealers": []}


@router.post("/notification-resend/{dealer_id}")
async def resend_notification(dealer_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Notification resends are retired in Lasso-only mode."""
    raise HTTPException(status_code=409, detail="Notification resend is disabled in Lasso-only mode")


@router.get("/lead-export")
async def lead_export(
    status: Optional[str] = None,
    province: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    """Export live Lasso-backed rows as JSON (frontend converts to CSV)."""
    conditions = []
    params = []
    if status:
        conditions.append("status_bucket = %s")
        params.append(status)
    if province:
        conditions.append("province = %s")
        params.append(province.upper())
    if start_date:
        conditions.append("created_at >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("created_at <= %s")
        params.append(end_date)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT *
        FROM (
            SELECT
                'unassigned' AS source_bucket,
                lasso_lead_id AS id,
                lasso_lead_id,
                name,
                email,
                NULL::text AS phone,
                location_text AS address,
                city,
                province,
                NULL::text AS postal_code,
                NULL::text AS dealer_name_assigned,
                'active' AS status,
                NULL::text AS project_type,
                NULL::text AS product_type,
                NULL::numeric AS square_footage,
                NULL::text AS business_category,
                NULL::text AS lead_source,
                NULL::text AS company_name,
                NULL::text AS comments,
                NULL::text AS landing_page,
                NULL::text AS landing_page_url,
                NULL::text AS landing_page_variant,
                NULL::text AS utm_source,
                NULL::text AS utm_medium,
                NULL::text AS utm_campaign,
                NULL::text AS utm_content,
                NULL::numeric AS value_of_order,
                record_date AS created_at,
                last_interaction,
                'active' AS status_bucket
            FROM dashboard_unassigned_leads

            UNION ALL

            SELECT
                'active' AS source_bucket,
                lasso_lead_id AS id,
                lasso_lead_id,
                name,
                email,
                NULL::text AS phone,
                location_text AS address,
                city,
                province,
                NULL::text AS postal_code,
                dealer_name AS dealer_name_assigned,
                'active' AS status,
                NULL::text AS project_type,
                NULL::text AS product_type,
                NULL::numeric AS square_footage,
                NULL::text AS business_category,
                NULL::text AS lead_source,
                NULL::text AS company_name,
                NULL::text AS comments,
                NULL::text AS landing_page,
                NULL::text AS landing_page_url,
                NULL::text AS landing_page_variant,
                NULL::text AS utm_source,
                NULL::text AS utm_medium,
                NULL::text AS utm_campaign,
                NULL::text AS utm_content,
                NULL::numeric AS value_of_order,
                date_assigned AS created_at,
                last_interaction,
                'active' AS status_bucket
            FROM dashboard_active_leads

            UNION ALL

            SELECT
                'history' AS source_bucket,
                lasso_lead_id AS id,
                lasso_lead_id,
                name,
                email,
                phone,
                address,
                city,
                province,
                postal_code,
                dealer_name AS dealer_name_assigned,
                current_status AS status,
                project_type,
                product_type,
                square_footage_value AS square_footage,
                business_category,
                lead_source,
                company_name,
                comments,
                landing_page,
                landing_page_url,
                landing_page_variant,
                utm_source,
                utm_medium,
                utm_campaign,
                utm_content,
                value_of_order,
                COALESCE(submit_date, created_date, form_submit_date) AS created_at,
                last_interaction,
                status_bucket
            FROM dashboard_history_leads
        ) exported
        {where}
        ORDER BY created_at DESC NULLS LAST, last_interaction DESC NULLS LAST
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
