import json
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field, validator
from auth import AdminUser, get_current_user
from db import execute_query, get_db_connection

router = APIRouter(prefix="/api/admin", tags=["Admin Leads"])


class NewLeadRequest(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postal_code: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    job_type: Optional[str] = None
    product_type: Optional[str] = None
    product_type_2: Optional[str] = None
    product_type_3: Optional[str] = None
    square_footage: Optional[float] = None
    custom_pick_1: Optional[str] = None
    project_city: Optional[str] = None
    project_type: Optional[str] = None
    business_category: Optional[str] = None
    dealer_email: Optional[str] = None
    lead_source: Optional[str] = None
    opt_in: Optional[bool] = False
    landing_page: Optional[str] = None
    landing_page_url: Optional[str] = None
    landing_page_variant: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_content: Optional[str] = None
    utm_term: Optional[str] = None
    custom_pick_3: Optional[str] = None
    comments: Optional[str] = None


class AssignDealerRequest(BaseModel):
    dealer_ids: List[int]


class UpdateLeadRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postal_code: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    job_type: Optional[str] = None
    product_type: Optional[str] = None
    product_type_2: Optional[str] = None
    product_type_3: Optional[str] = None
    square_footage: Optional[float] = None
    project_city: Optional[str] = None
    project_type: Optional[str] = None
    business_category: Optional[str] = None
    lead_source: Optional[str] = None
    status: Optional[str] = None
    comments: Optional[str] = None
    value_of_order: Optional[float] = None


@router.get("/unassigned-leads")
async def get_unassigned_leads(current_user: AdminUser = Depends(get_current_user)):
    """Get leads that haven't been assigned to a dealer yet."""
    query = """
        SELECT l.*, i.name as installer_name
        FROM leads l
        LEFT JOIN installers i ON l.assigned_installer_id = i.id
        WHERE l.assigned_dealer_id IS NULL
        AND l.status NOT IN ('converted', 'dead', 'archived')
        ORDER BY l.created_at DESC
    """
    leads = execute_query(query)
    return {"leads": leads, "count": len(leads)}


@router.get("/active-leads")
async def get_active_leads(current_user: AdminUser = Depends(get_current_user)):
    """Get leads that are assigned to dealers and actively being worked."""
    query = """
        SELECT l.*, i.name as installer_name, d.name as dealer_name_assigned
        FROM leads l
        LEFT JOIN installers i ON l.assigned_installer_id = i.id
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        WHERE l.assigned_dealer_id IS NOT NULL
        AND l.status NOT IN ('converted', 'dead', 'archived')
        ORDER BY l.created_at DESC
    """
    leads = execute_query(query)
    return {"leads": leads, "count": len(leads)}


@router.post("/leads/new")
async def create_new_lead(lead: NewLeadRequest, current_user: AdminUser = Depends(get_current_user)):
    """Admin manually inserts a new lead with 35+ fields."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = """
                INSERT INTO leads (
                    name, first_name, last_name, email, phone, address, city, province,
                    postal_code, job_title, company_name, job_type, product_type,
                    product_type_2, product_type_3, square_footage, custom_pick_1,
                    project_city, project_type, business_category, dealer_email,
                    lead_source, opt_in, landing_page, landing_page_url,
                    landing_page_variant, utm_source, utm_medium, utm_campaign,
                    utm_content, utm_term, custom_pick_3, comments, status,
                    form_submit_date, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, 'active',
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                ) RETURNING id, created_at
            """
            full_name = f"{lead.first_name} {lead.last_name}"
            cursor.execute(query, (
                full_name, lead.first_name, lead.last_name, lead.email, lead.phone,
                lead.address, lead.city, lead.province, lead.postal_code,
                lead.job_title, lead.company_name, lead.job_type, lead.product_type,
                lead.product_type_2, lead.product_type_3, lead.square_footage,
                lead.custom_pick_1, lead.project_city, lead.project_type,
                lead.business_category, lead.dealer_email, lead.lead_source,
                lead.opt_in, lead.landing_page, lead.landing_page_url,
                lead.landing_page_variant, lead.utm_source, lead.utm_medium,
                lead.utm_campaign, lead.utm_content, lead.utm_term,
                lead.custom_pick_3, lead.comments
            ))
            result = cursor.fetchone()
            conn.commit()
            return {"message": "Lead created successfully", "lead_id": result['id'], "created_at": result['created_at']}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create lead: {str(e)}")
    finally:
        conn.close()


@router.post("/leads/{lead_id}/assign-dealer")
async def assign_dealer(lead_id: int, req: AssignDealerRequest, current_user: AdminUser = Depends(get_current_user)):
    """Assign one or more dealers to a lead."""
    # Verify lead exists
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Set the primary dealer (first in list) on leads table
            primary_dealer_id = req.dealer_ids[0]
            cursor.execute(
                "UPDATE leads SET assigned_dealer_id = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (primary_dealer_id, lead_id)
            )
            # Create assignment records for all dealers
            for dealer_id in req.dealer_ids:
                cursor.execute(
                    """INSERT INTO lead_assignments (lead_id, dealer_id, assigned_at, status)
                    VALUES (%s, %s, CURRENT_TIMESTAMP, 'pending')
                    ON CONFLICT DO NOTHING""",
                    (lead_id, dealer_id)
                )
            # Log the assignment
            dealer_names = execute_query(
                "SELECT id, name FROM dealers WHERE id = ANY(%s)",
                (req.dealer_ids,)
            )
            names = ", ".join(d['name'] for d in dealer_names) if dealer_names else str(req.dealer_ids)
            cursor.execute(
                """INSERT INTO lead_logs (lead_id, log_type, message, created_by, created_at)
                VALUES (%s, 'assignment', %s, %s, CURRENT_TIMESTAMP)""",
                (lead_id, f"Assigned to dealer(s): {names}", current_user.username)
            )
            conn.commit()
        return {"message": "Dealers assigned successfully", "lead_id": lead_id, "dealer_ids": req.dealer_ids}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/leads/{lead_id}/update")
async def update_lead(lead_id: int, req: UpdateLeadRequest, current_user: AdminUser = Depends(get_current_user)):
    """Update lead fields."""
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    updates = []
    params = []
    data = req.dict(exclude_none=True)
    for field, value in data.items():
        updates.append(f"{field} = %s")
        params.append(value)

    if not updates:
        return {"message": "No fields to update"}

    # Also update the name field if first/last name changed
    if 'first_name' in data or 'last_name' in data:
        # Get current values
        current = execute_query("SELECT first_name, last_name FROM leads WHERE id = %s", (lead_id,))
        if current:
            fn = data.get('first_name', current[0].get('first_name') or '')
            ln = data.get('last_name', current[0].get('last_name') or '')
            updates.append("name = %s")
            params.append(f"{fn} {ln}".strip())

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(lead_id)

    query = f"UPDATE leads SET {', '.join(updates)} WHERE id = %s"
    execute_query(query, tuple(params), fetch=False)

    return {"message": "Lead updated successfully", "lead_id": lead_id}


@router.post("/leads/{lead_id}/archive")
async def archive_lead(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Archive a lead (soft delete)."""
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    execute_query(
        "UPDATE leads SET status = 'archived', updated_at = CURRENT_TIMESTAMP WHERE id = %s",
        (lead_id,), fetch=False
    )
    execute_query(
        """INSERT INTO lead_logs (lead_id, log_type, message, created_by, created_at)
        VALUES (%s, 'status_change', 'Lead archived', %s, CURRENT_TIMESTAMP)""",
        (lead_id, current_user.username), fetch=False
    )
    return {"message": "Lead archived", "lead_id": lead_id}


@router.post("/leads/{lead_id}/delete")
async def delete_lead(lead_id: int, current_user: AdminUser = Depends(get_current_user)):
    """Permanently delete a lead."""
    lead = execute_query("SELECT id FROM leads WHERE id = %s", (lead_id,))
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    execute_query("DELETE FROM leads WHERE id = %s", (lead_id,), fetch=False)
    return {"message": "Lead deleted", "lead_id": lead_id}
