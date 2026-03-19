from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import AdminUser, get_current_user
from db import execute_query, get_db_connection

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
    pages = execute_query(
        "SELECT * FROM resource_pages ORDER BY parent_id NULLS FIRST, sort_order, title"
    )
    return {"pages": pages}


@router.get("/{page_id}")
async def get_resource(page_id: int, current_user: AdminUser = Depends(get_current_user)):
    page = execute_query("SELECT * FROM resource_pages WHERE id = %s", (page_id,))
    if not page:
        raise HTTPException(status_code=404, detail="Resource page not found")
    return page[0]


@router.post("")
async def create_resource(req: ResourcePageRequest, current_user: AdminUser = Depends(get_current_user)):
    slug = req.slug or req.title.lower().replace(" ", "-")
    result = execute_query(
        """INSERT INTO resource_pages (title, slug, content, parent_id, sort_order, is_published, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) RETURNING id""",
        (req.title, slug, req.content, req.parent_id, req.sort_order, req.is_published)
    )
    return {"message": "Resource created", "id": result[0]['id']}


@router.put("/{page_id}")
async def update_resource(page_id: int, req: ResourcePageUpdate, current_user: AdminUser = Depends(get_current_user)):
    page = execute_query("SELECT id FROM resource_pages WHERE id = %s", (page_id,))
    if not page:
        raise HTTPException(status_code=404, detail="Resource page not found")

    updates = []
    params = []
    data = req.dict(exclude_none=True)
    for field, value in data.items():
        updates.append(f"{field} = %s")
        params.append(value)

    if not updates:
        return {"message": "No fields to update"}

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(page_id)
    execute_query(f"UPDATE resource_pages SET {', '.join(updates)} WHERE id = %s", tuple(params), fetch=False)
    return {"message": "Resource updated", "id": page_id}


@router.delete("/{page_id}")
async def delete_resource(page_id: int, current_user: AdminUser = Depends(get_current_user)):
    page = execute_query("SELECT id FROM resource_pages WHERE id = %s", (page_id,))
    if not page:
        raise HTTPException(status_code=404, detail="Resource page not found")
    execute_query("DELETE FROM resource_pages WHERE id = %s", (page_id,), fetch=False)
    return {"message": "Resource deleted"}
