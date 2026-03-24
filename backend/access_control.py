from dataclasses import dataclass
from typing import Iterable

from fastapi import HTTPException, status

from auth import AdminUser
from settings_store import get_setting_bool

ROLE_LEVELS = {
    "viewer": 0,
    "admin": 1,
    "superadmin": 2,
}


def normalize_role(role: str | None) -> str:
    normalized = (role or "admin").strip().lower()
    return normalized if normalized in ROLE_LEVELS else "admin"


def ensure_roles(current_user: AdminUser, allowed_roles: Iterable[str]) -> AdminUser:
    normalized_role = normalize_role(current_user.role)
    allowed = {normalize_role(role) for role in allowed_roles}
    if normalized_role not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires one of: {', '.join(sorted(allowed))}",
        )
    return current_user


def is_superadmin(current_user: AdminUser) -> bool:
    return normalize_role(current_user.role) == "superadmin"


@dataclass
class AIOperationsPausedError(Exception):
    message_text: str = "AI operations paused."

    def to_payload(self) -> dict:
        return {"type": "agent_paused", "message": self.message_text}


def assert_ai_operations_enabled() -> None:
    if not get_setting_bool("agent_enabled", True):
        raise AIOperationsPausedError()


def build_ai_pause_message(meta: dict | None) -> str:
    if not meta or meta.get("type") != "agent_paused":
        return "AI operations paused."
    return str(meta.get("message") or "AI operations paused.")
