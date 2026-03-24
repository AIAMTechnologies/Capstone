from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import AdminUser, get_current_user

router = APIRouter(prefix="/api/admin/resources", tags=["Admin Resources"])


class ResourcePageRequest(BaseModel):
    title: str
    slug: Optional[str] = None
    content: Optional[str] = None
    parent_id: Optional[int] = None
    sort_order: Optional[int] = 0
    is_published: Optional[bool] = True


class ResourcePageUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    content: Optional[str] = None
    parent_id: Optional[int] = None
    sort_order: Optional[int] = None
    is_published: Optional[bool] = None


@router.get("")
async def list_resources(current_user: AdminUser = Depends(get_current_user)):
    return {"pages": [], "message": "Resources are retired in Lasso-only mode."}


@router.get("/{page_id}")
async def get_resource(page_id: int, current_user: AdminUser = Depends(get_current_user)):
    raise HTTPException(status_code=410, detail="Resources are retired in Lasso-only mode.")


@router.post("")
async def create_resource(req: ResourcePageRequest, current_user: AdminUser = Depends(get_current_user)):
    raise HTTPException(status_code=410, detail="Resources are retired in Lasso-only mode.")


@router.put("/{page_id}")
async def update_resource(page_id: int, req: ResourcePageUpdate, current_user: AdminUser = Depends(get_current_user)):
    raise HTTPException(status_code=410, detail="Resources are retired in Lasso-only mode.")


@router.delete("/{page_id}")
async def delete_resource(page_id: int, current_user: AdminUser = Depends(get_current_user)):
    raise HTTPException(status_code=410, detail="Resources are retired in Lasso-only mode.")
