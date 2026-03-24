from datetime import datetime, timedelta
from typing import Any, Optional

from db import execute_query


def get_lead_tool(lead_id: int) -> dict[str, Any]:
    rows = execute_query(
        """
        SELECT l.*, d.name AS dealer_name_assigned, rd.name AS recommended_dealer_name
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        LEFT JOIN dealers rd ON l.recommended_dealer_id = rd.id
        WHERE l.id = %s
        """,
        (lead_id,),
    )
    if not rows:
        raise ValueError("Lead not found")
    return dict(rows[0])


def search_leads_tool(filters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    filters = filters or {}
    conditions = ["1=1"]
    params: list[Any] = []

    status_value = filters.get("status")
    if status_value:
        if isinstance(status_value, list):
            conditions.append("l.status = ANY(%s)")
            params.append(status_value)
        else:
            conditions.append("l.status = %s")
            params.append(status_value)

    province = filters.get("province")
    if province:
        conditions.append("l.province = %s")
        params.append(str(province).upper())

    dealer = filters.get("dealer")
    if dealer:
        if isinstance(dealer, int):
            conditions.append("l.assigned_dealer_id = %s")
            params.append(dealer)
        else:
            conditions.append("(d.name ILIKE %s OR rd.name ILIKE %s)")
            params.extend([f"%{dealer}%", f"%{dealer}%"])

    date_from = filters.get("date_from")
    if date_from:
        conditions.append("l.created_at >= %s")
        params.append(date_from)

    date_to = filters.get("date_to")
    if date_to:
        conditions.append("l.created_at <= %s")
        params.append(date_to)

    min_score = filters.get("min_score")
    if min_score is not None:
        conditions.append("COALESCE(l.ai_score, 0) >= %s")
        params.append(min_score)

    max_score = filters.get("max_score")
    if max_score is not None:
        conditions.append("COALESCE(l.ai_score, 0) <= %s")
        params.append(max_score)

    limit = min(int(filters.get("limit", 50) or 50), 200)
    offset = max(int(filters.get("offset", 0) or 0), 0)
    params.extend([limit, offset])

    where_clause = " AND ".join(conditions)
    rows = execute_query(
        f"""
        SELECT l.*, d.name AS dealer_name_assigned, rd.name AS recommended_dealer_name
        FROM leads l
        LEFT JOIN dealers d ON l.assigned_dealer_id = d.id
        LEFT JOIN dealers rd ON l.recommended_dealer_id = rd.id
        WHERE {where_clause}
        ORDER BY l.updated_at DESC NULLS LAST, l.created_at DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params),
    )
    return {"leads": [dict(row) for row in rows], "count": len(rows)}


def get_dealer_tool(dealer_id: int) -> dict[str, Any]:
    rows = execute_query("SELECT * FROM dealers WHERE id = %s", (dealer_id,))
    if not rows:
        raise ValueError("Dealer not found")
    return dict(rows[0])


def list_dealers_tool(province: Optional[str] = None, product_type: Optional[str] = None) -> dict[str, Any]:
    conditions = ["d.is_active = TRUE"]
    params: list[Any] = []

    if province:
        conditions.append("d.province = %s")
        params.append(str(province).upper())

    if product_type:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM leads l
                WHERE l.assigned_dealer_id = d.id
                  AND (
                    l.product_type ILIKE %s
                    OR l.product_type_2 ILIKE %s
                    OR l.product_type_3 ILIKE %s
                  )
            )
            """
        )
        like_value = f"%{product_type}%"
        params.extend([like_value, like_value, like_value])

    where_clause = " AND ".join(conditions)
    rows = execute_query(
        f"SELECT d.* FROM dealers d WHERE {where_clause} ORDER BY d.province, d.name",
        tuple(params),
    )
    return {"dealers": [dict(row) for row in rows], "count": len(rows)}


def get_lead_score_tool(lead_id: int) -> dict[str, Any]:
    rows = execute_query(
        """
        SELECT id, ai_priority, ai_score, ai_reasoning, ai_scored_at,
               ai_match_explanation, ai_conversion_likelihood, ai_conversion_explanation
        FROM leads
        WHERE id = %s
        """,
        (lead_id,),
    )
    if not rows:
        raise ValueError("Lead not found")
    return dict(rows[0])


def _period_start(period: Optional[str]) -> Optional[datetime]:
    if not period or period == "all":
        return None
    now = datetime.utcnow()
    mapping = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "12m": timedelta(days=365),
    }
    delta = mapping.get(period)
    return now - delta if delta else None


def get_dealer_performance_tool(dealer_id: int, period: Optional[str] = None) -> dict[str, Any]:
    date_from = _period_start(period)
    if date_from:
        rows = execute_query(
            """
            SELECT d.id, d.name,
                   COUNT(l.id) AS total_leads,
                   COUNT(CASE WHEN l.status = 'converted' THEN 1 END) AS converted,
                   COUNT(CASE WHEN l.status = 'dead' THEN 1 END) AS dead,
                   COUNT(CASE WHEN l.status IN ('active', 'follow_up') THEN 1 END) AS active
            FROM dealers d
            LEFT JOIN leads l
              ON l.assigned_dealer_id = d.id
             AND l.created_at >= %s
            WHERE d.id = %s
            GROUP BY d.id, d.name
            """,
            (date_from, dealer_id),
        )
    else:
        rows = execute_query(
            """
            SELECT d.id, d.name,
                   COUNT(l.id) AS total_leads,
                   COUNT(CASE WHEN l.status = 'converted' THEN 1 END) AS converted,
                   COUNT(CASE WHEN l.status = 'dead' THEN 1 END) AS dead,
                   COUNT(CASE WHEN l.status IN ('active', 'follow_up') THEN 1 END) AS active
            FROM dealers d
            LEFT JOIN leads l ON l.assigned_dealer_id = d.id
            WHERE d.id = %s
            GROUP BY d.id, d.name
            """,
            (dealer_id,),
        )

    if not rows:
        raise ValueError("Dealer not found")

    row = dict(rows[0])
    total = int(row.get("total_leads") or 0)
    converted = int(row.get("converted") or 0)
    row["conversion_rate_pct"] = round((converted / total) * 100, 2) if total else 0.0
    row["period"] = period or "all"
    return row
