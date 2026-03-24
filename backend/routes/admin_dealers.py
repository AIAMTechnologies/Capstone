from typing import Optional
from fastapi import APIRouter, Depends
from auth import AdminUser, get_current_user
from db import execute_query

router = APIRouter(prefix="/api/admin/dealers", tags=["Admin Dealers"])


@router.get("")
async def list_dealers(
    province: Optional[str] = None,
    current_user: AdminUser = Depends(get_current_user)
):
    if province:
        dealers = execute_query(
            "SELECT * FROM dealers WHERE is_active = TRUE AND province = %s ORDER BY name",
            (province.upper(),)
        )
    else:
        dealers = execute_query("SELECT * FROM dealers WHERE is_active = TRUE ORDER BY province, name")
    return {"dealers": dealers, "count": len(dealers)}


@router.get("/options")
async def dealer_options(current_user: AdminUser = Depends(get_current_user)):
    dealers = execute_query("SELECT id, name, city, province, email FROM dealers WHERE is_active = TRUE ORDER BY name")
    return {"dealers": dealers}
