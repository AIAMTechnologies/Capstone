from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from access_control import ensure_roles, is_superadmin, normalize_role
from audit_logger import log_event
from auth import AdminUser, get_current_user
from settings_store import get_setting_bool, get_setting_float, set_setting

router = APIRouter(prefix="/api/admin", tags=["Admin AI Controls"])


class AgentEnabledRequest(BaseModel):
    enabled: bool


class SpendLimitUpdateRequest(BaseModel):
    daily_spend_limit_cad: float = Field(..., ge=0)
    monthly_spend_limit_cad: float = Field(..., ge=0)


@router.get("/ai-controls")
async def get_ai_controls(current_user: AdminUser = Depends(get_current_user)):
    ensure_roles(current_user, {"viewer", "admin", "superadmin"})
    return {
        "agent_enabled": get_setting_bool("agent_enabled", True),
        "daily_spend_limit_cad": get_setting_float("daily_spend_limit_cad", 50.0),
        "monthly_spend_limit_cad": get_setting_float("monthly_spend_limit_cad", 500.0),
        "role": normalize_role(current_user.role),
        "can_manage": is_superadmin(current_user),
    }


@router.post("/ai-controls/agent-enabled")
async def update_agent_enabled(
    req: AgentEnabledRequest,
    current_user: AdminUser = Depends(get_current_user),
):
    ensure_roles(current_user, {"superadmin"})
    set_setting("agent_enabled", "true" if req.enabled else "false", updated_by=current_user.username, value_boolean=req.enabled)
    log_event(
        event_type="AI_AGENT_TOGGLE",
        entity_type="settings",
        entity_id="agent_enabled",
        actor=current_user.username,
        payload={"agent_enabled": req.enabled},
    )
    return {"message": "AI agent status updated", "agent_enabled": req.enabled}


@router.post("/ai-controls/spend-limits")
async def update_spend_limits(
    req: SpendLimitUpdateRequest,
    current_user: AdminUser = Depends(get_current_user),
):
    ensure_roles(current_user, {"superadmin"})
    set_setting("daily_spend_limit_cad", req.daily_spend_limit_cad, updated_by=current_user.username)
    set_setting("monthly_spend_limit_cad", req.monthly_spend_limit_cad, updated_by=current_user.username)
    log_event(
        event_type="AI_SPEND_LIMITS_UPDATED",
        entity_type="settings",
        entity_id="daily_spend_limit_cad,monthly_spend_limit_cad",
        actor=current_user.username,
        payload={
            "daily_spend_limit_cad": req.daily_spend_limit_cad,
            "monthly_spend_limit_cad": req.monthly_spend_limit_cad,
        },
    )
    return {
        "message": "AI spend limits updated",
        "daily_spend_limit_cad": req.daily_spend_limit_cad,
        "monthly_spend_limit_cad": req.monthly_spend_limit_cad,
    }
