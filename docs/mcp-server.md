# WFC Lead Portal MCP Server

The portal now exposes an MCP-compatible JSON-RPC server at `/mcp`.

Base URL:
- `http://localhost:8000/mcp`

Authentication:
- Send the same admin bearer token used for the `/api/admin/*` endpoints.
- Example header: `Authorization: Bearer <jwt>`

Supported RPC methods:
- `initialize`
- `tools/list`
- `tools/call`

Exposed tools:
- `get_lead(lead_id)`
- `search_leads(filters)`
- `get_dealer(dealer_id)`
- `list_dealers(province, product_type)`
- `get_lead_score(lead_id)`
- `get_dealer_performance(dealer_id, period)`

Example `tools/list` request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list"
}
```

Example `tools/call` request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_lead",
    "arguments": {
      "lead_id": 15426
    }
  }
}
```

Notes:
- The server is mounted additively and does not replace any existing API routes.
- Internal AI routes now reuse the same tool layer where the data contract already matched.
- Future LLM clients can connect without code changes as long as they speak MCP-style JSON-RPC over HTTP.
