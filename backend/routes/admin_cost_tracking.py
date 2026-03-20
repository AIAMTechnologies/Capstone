from fastapi import APIRouter, Depends

from auth import AdminUser, get_current_user
from cost_control import get_cost_snapshot

router = APIRouter(prefix="/api/admin", tags=["Admin Cost Tracking"])


@router.get("/cost-tracking")
async def get_cost_tracking(current_user: AdminUser = Depends(get_current_user)):
    return get_cost_snapshot()
