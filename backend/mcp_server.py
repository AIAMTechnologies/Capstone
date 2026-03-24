from typing import Any

from fastapi import FastAPI, Header, HTTPException

from auth import resolve_admin_user_from_token
from mcp_tools import (
    get_dealer_performance_tool,
    get_dealer_tool,
    get_lead_score_tool,
    get_lead_tool,
    list_dealers_tool,
    search_leads_tool,
)

mcp_app = FastAPI(title="WFC Lead Portal MCP", version="1.0.0")


def _require_user(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user = resolve_admin_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return user


TOOLS = {
    "get_lead": {
        "description": "Fetch a full lead record by ID.",
        "schema": {
            "type": "object",
            "properties": {"lead_id": {"type": "integer"}},
            "required": ["lead_id"],
        },
        "handler": lambda arguments: get_lead_tool(int(arguments["lead_id"])),
    },
    "search_leads": {
        "description": "Search leads by status, province, dealer, date range, and score filters.",
        "schema": {
            "type": "object",
            "properties": {
                "filters": {"type": "object"},
            },
        },
        "handler": lambda arguments: search_leads_tool(arguments.get("filters") or {}),
    },
    "get_dealer": {
        "description": "Fetch a full dealer record by ID.",
        "schema": {
            "type": "object",
            "properties": {"dealer_id": {"type": "integer"}},
            "required": ["dealer_id"],
        },
        "handler": lambda arguments: get_dealer_tool(int(arguments["dealer_id"])),
    },
    "list_dealers": {
        "description": "List dealers filtered by province and historical product type.",
        "schema": {
            "type": "object",
            "properties": {
                "province": {"type": "string"},
                "product_type": {"type": "string"},
            },
        },
        "handler": lambda arguments: list_dealers_tool(
            province=arguments.get("province"),
            product_type=arguments.get("product_type"),
        ),
    },
    "get_lead_score": {
        "description": "Fetch the current AI score and reasoning for a lead.",
        "schema": {
            "type": "object",
            "properties": {"lead_id": {"type": "integer"}},
            "required": ["lead_id"],
        },
        "handler": lambda arguments: get_lead_score_tool(int(arguments["lead_id"])),
    },
    "get_dealer_performance": {
        "description": "Fetch conversion stats for a dealer over a period.",
        "schema": {
            "type": "object",
            "properties": {
                "dealer_id": {"type": "integer"},
                "period": {"type": "string", "enum": ["7d", "30d", "90d", "12m", "all"]},
            },
            "required": ["dealer_id"],
        },
        "handler": lambda arguments: get_dealer_performance_tool(
            dealer_id=int(arguments["dealer_id"]),
            period=arguments.get("period"),
        ),
    },
}


@mcp_app.get("/")
async def mcp_root():
    return {
        "name": "wfc-lead-portal-mcp",
        "version": "1.0.0",
        "transport": "jsonrpc-over-http",
        "docs": "/docs/mcp-server.md",
    }


@mcp_app.post("/")
async def mcp_rpc(payload: dict[str, Any], authorization: str | None = Header(None)):
    _require_user(authorization)

    method = payload.get("method")
    request_id = payload.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "wfc-lead-portal-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": tool["description"],
                        "inputSchema": tool["schema"],
                    }
                    for name, tool in TOOLS.items()
                ]
            },
        }

    if method == "tools/call":
        params = payload.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        tool = TOOLS.get(name)
        if not tool:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        try:
            result = tool["handler"](arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "json", "json": result}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Unsupported method: {method}"},
    }
