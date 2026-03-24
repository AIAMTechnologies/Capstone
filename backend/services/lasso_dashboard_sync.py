import json
import logging
import os
import re
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import requests
from psycopg2.extras import Json

from audit_logger import log_event, timed_call_latency_ms, timed_call_start
from db import execute_query, get_db_connection, release_db_connection
from settings_store import get_setting_float

logger = logging.getLogger("lasso_dashboard_sync")

LASSO_BASE = os.getenv("LASSO_BASE_URL", "https://lap.conveniencegroup.com")
LASSO_USER = os.getenv("LASSO_USERNAME", "info@windowfilmcanada.ca")
LASSO_PASS = os.getenv("LASSO_PASSWORD", "890*()iopIOP@WFC")
LASSO_PAGE_LIMIT = int(os.getenv("LASSO_PAGE_LIMIT", "1000"))
LASSO_UNASSIGNED_PAGE_LIMIT = int(os.getenv("LASSO_UNASSIGNED_PAGE_LIMIT", str(LASSO_PAGE_LIMIT)))
LASSO_ACTIVE_PAGE_LIMIT = int(os.getenv("LASSO_ACTIVE_PAGE_LIMIT", "100"))
LASSO_HISTORY_PAGE_LIMIT = int(os.getenv("LASSO_HISTORY_PAGE_LIMIT", "100"))
LASSO_MIN_PAGE_LIMIT = int(os.getenv("LASSO_MIN_PAGE_LIMIT", "25"))
LASSO_TIMEOUT_CONNECT_SECONDS = int(os.getenv("LASSO_TIMEOUT_CONNECT_SECONDS", "15"))
LASSO_TIMEOUT_READ_SECONDS = int(os.getenv("LASSO_TIMEOUT_READ_SECONDS", "180"))
LASSO_REQUEST_RETRIES = int(os.getenv("LASSO_REQUEST_RETRIES", "2"))
LASSO_SYNC_STALE_MINUTES = int(os.getenv("LASSO_SYNC_STALE_MINUTES", "45"))

SYNC_STATUS: dict[str, Any] = {
    "in_progress": False,
    "sync_type": None,
    "started_at": None,
    "last_error": None,
}
SYNC_LOCK = threading.Lock()

ACTIVE_QUEUE_EXCLUDED_STATUSES = {"Dead Lead"}
TIMEFRAMES = [
    ("This Month", "this_month"),
    ("Last Month", "last_month"),
    ("Last 3 Months", "last_3_months"),
    ("Last 6 Months", "last_6_months"),
    ("Last 12 Months", "last_12_months"),
    ("All Time", "all_time"),
]
LEAD_REPORT_TIMEFRAME_CODES = [
    ("This Month", 1),
    ("Last Month", 2),
    ("Last 3 Months", 3),
    ("Last 6 Months", 4),
    ("Last 12 Months", 5),
    ("All Time", 6),
]


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "invalid date":
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> Optional[int]:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except ValueError:
        return None


def _safe_decimal(value: Any) -> Optional[Decimal]:
    text = _clean(value)
    if not text:
        return None
    try:
        return Decimal(text.replace(",", ""))
    except Exception:
        return None


def _status_bucket(raw_status: Optional[str]) -> str:
    normalized = (_clean(raw_status) or "").lower()
    if normalized == "converted sale":
        return "converted"
    if normalized == "dead lead":
        return "dead"
    return "active"


def _lead_anchor_datetime(row: dict[str, Any]) -> Optional[datetime]:
    return (
        _parse_datetime(row.get("lastInteraction"))
        or _parse_datetime(row.get("submitDate"))
        or _parse_datetime(row.get("createdDate"))
        or _parse_datetime(row.get("formSubmitDate"))
    )


def _lead_display_name(row: dict[str, Any]) -> str:
    first_name = _clean(row.get("firstName")) or ""
    last_name = _clean(row.get("lastName")) or ""
    full_name = f"{first_name} {last_name}".strip()
    return full_name or _clean(row.get("email")) or f"Lead #{row.get('leadId')}"


def _lead_location_text(row: dict[str, Any]) -> str:
    parts = [_clean(row.get("city")), _clean(row.get("province"))]
    return ", ".join(part for part in parts if part)


def _lead_date_only(row: dict[str, Any]) -> Optional[date]:
    anchor = (
        _parse_datetime(row.get("submitDate"))
        or _parse_datetime(row.get("createdDate"))
        or _parse_datetime(row.get("formSubmitDate"))
        or _lead_anchor_datetime(row)
    )
    return anchor.date() if anchor else None


def _parse_square_footage_value(raw_value: Any) -> Optional[float]:
    text = (_clean(raw_value) or "").lower().replace("sqft", "").strip()
    if not text:
        return None

    normalized = text.replace(",", "")
    if normalized.endswith("+"):
        normalized = normalized[:-1].strip()
        try:
            return float(normalized)
        except ValueError:
            return None

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", normalized)
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        return (lower + upper) / 2

    try:
        return float(normalized)
    except ValueError:
        return None


def _project_bucket_key(raw_value: Any) -> Optional[str]:
    numeric = _parse_square_footage_value(raw_value)
    if numeric is None:
        text = (_clean(raw_value) or "").lower()
        if "20,000" in text or "20000" in text:
            return "sqft_20000_plus"
        return None
    if numeric < 500:
        return "sqft_1_499"
    if numeric < 1000:
        return "sqft_500_999"
    if numeric < 3500:
        return "sqft_1000_3499"
    if numeric < 7500:
        return "sqft_3500_7499"
    if numeric < 20000:
        return "sqft_7500_19999"
    return "sqft_20000_plus"


def _parse_avg_response_hours(raw_value: Any) -> Optional[float]:
    text = (_clean(raw_value) or "").lower()
    if not text:
        return None

    hours = 0.0
    minutes = 0.0

    hours_match = re.search(r"(\d+)\s*hour", text)
    minutes_match = re.search(r"(\d+)\s*minute", text)

    if hours_match:
        hours = float(hours_match.group(1))
    if minutes_match:
        minutes = float(minutes_match.group(1))

    if hours == 0.0 and minutes == 0.0:
        return None
    return hours + (minutes / 60.0)


def _dealer_name_map(
    active: list[dict[str, Any]],
    unassigned: list[dict[str, Any]],
    history: list[dict[str, Any]],
    lead_status_rows: list[dict[str, Any]],
    lead_response_rows: list[dict[str, Any]],
) -> dict[int, str]:
    result: dict[int, str] = {}
    sources = [
        lead_status_rows,
        lead_response_rows,
        active,
        unassigned,
        history,
    ]

    for rows in sources:
        for row in rows or []:
            dealer_id = _safe_int(row.get("dealer_id") if isinstance(row, dict) else None)
            if dealer_id is None and isinstance(row, dict):
                dealer_id = _safe_int(row.get("dealerId"))
            if dealer_id is None:
                continue

            dealer_name = None
            if isinstance(row, dict):
                dealer_name = (
                    _clean(row.get("dealer_name"))
                    or _clean(row.get("dealerName"))
                    or _clean(row.get("name"))
                    or _clean(row.get("sourceDealer"))
                )
            if dealer_name and dealer_id not in result:
                result[dealer_id] = dealer_name
    return result


def _build_local_lead_lookup() -> dict[str, dict[tuple[Any, ...], int]]:
    rows = execute_query(
        """
        SELECT
            id,
            first_name,
            last_name,
            name,
            email,
            city,
            province,
            created_at,
            form_submit_date
        FROM leads
        """
    )
    by_email_name_date: dict[tuple[Any, ...], int] = {}
    by_name_city_date: dict[tuple[Any, ...], int] = {}
    by_name_date: dict[tuple[Any, ...], int] = {}

    for row in rows:
        first_name = (_clean(row.get("first_name")) or "").lower()
        last_name = (_clean(row.get("last_name")) or "").lower()
        name = (_clean(row.get("name")) or "").lower()
        email = (_clean(row.get("email")) or "").lower()
        city = (_clean(row.get("city")) or "").lower()
        province = (_clean(row.get("province")) or "").lower()
        date_value = row.get("form_submit_date") or row.get("created_at")
        date_key = date_value.date() if isinstance(date_value, datetime) else date_value
        if email and first_name and last_name and date_key:
            by_email_name_date[(email, first_name, last_name, date_key)] = row["id"]
        if first_name and last_name and city and date_key:
            by_name_city_date[(first_name, last_name, city, province, date_key)] = row["id"]
        if (first_name or name) and date_key:
            by_name_date[(first_name or name, last_name, date_key)] = row["id"]

    return {
        "by_email_name_date": by_email_name_date,
        "by_name_city_date": by_name_city_date,
        "by_name_date": by_name_date,
    }


def _resolve_local_lead_id(row: dict[str, Any], lookup: dict[str, dict[tuple[Any, ...], int]]) -> Optional[int]:
    first_name = (_clean(row.get("firstName")) or "").lower()
    last_name = (_clean(row.get("lastName")) or "").lower()
    name = _lead_display_name(row).lower()
    email = (_clean(row.get("email")) or "").lower()
    city = (_clean(row.get("city")) or "").lower()
    province = (_clean(row.get("province")) or "").lower()
    date_key = _lead_date_only(row)

    if date_key and email and first_name and last_name:
        match = lookup["by_email_name_date"].get((email, first_name, last_name, date_key))
        if match:
            return match
    if date_key and first_name and last_name and city:
        match = lookup["by_name_city_date"].get((first_name, last_name, city, province, date_key))
        if match:
            return match
    if date_key:
        return lookup["by_name_date"].get((first_name or name, last_name, date_key))
    return None


def _dedupe_rows(unassigned: list[dict[str, Any]], active: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"active": 3, "unassigned": 2, "history": 1}
    seen: dict[int, tuple[str, dict[str, Any]]] = {}
    for source_name, rows in (("unassigned", unassigned), ("active", active), ("history", history)):
        for row in rows:
            lead_id = row.get("leadId")
            if not lead_id:
                continue
            existing = seen.get(lead_id)
            if existing is None or priority[source_name] >= priority[existing[0]]:
                seen[int(lead_id)] = (source_name, row)
    return [row for _, row in seen.values()]


def _dedupe_history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for row in history:
        lead_id = _safe_int(row.get("leadId"))
        if lead_id is None:
            continue
        seen[lead_id] = row
    return list(seen.values())


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _timeframe_bounds(label: str, today: date) -> tuple[Optional[date], Optional[date]]:
    if label == "This Month":
        return _month_start(today), today + timedelta(days=1)
    if label == "Last Month":
        this_month = _month_start(today)
        last_month = _month_start(this_month - timedelta(days=1))
        return last_month, this_month
    if label == "Last 3 Months":
        return today - timedelta(days=91), today + timedelta(days=1)
    if label == "Last 6 Months":
        return today - timedelta(days=183), today + timedelta(days=1)
    if label == "Last 12 Months":
        return today - timedelta(days=365), today + timedelta(days=1)
    return None, None


def _fetch_lasso_session() -> requests.Session:
    logger.info("Logging in to Lasso dashboard")
    session = requests.Session()
    timeout = (LASSO_TIMEOUT_CONNECT_SECONDS, LASSO_TIMEOUT_READ_SECONDS)
    response = session.get(f"{LASSO_BASE}/lasso9/cgi/action/login.html", timeout=timeout)
    match = re.search(r'form action="([^"]+)"', response.text)
    if not match:
        raise RuntimeError("Could not locate Lasso login form action")
    login_url = f"{LASSO_BASE}/lasso9/cgi/action/login.html{match.group(1)}"
    login_response = session.post(
        login_url,
        data={"username": LASSO_USER, "password": LASSO_PASS, "do-login": "Login"},
        allow_redirects=True,
        timeout=timeout,
    )
    if "login.html" in login_response.url and "username" in login_response.text[:1000]:
        raise RuntimeError("Lasso login failed")
    return session


def _fetch_paginated(session: requests.Session, endpoint_name: str, limit: int = LASSO_PAGE_LIMIT) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    page_limit = max(limit, 1)
    min_page_limit = min(LASSO_MIN_PAGE_LIMIT, page_limit)
    page = 1
    rows_fetched = 0
    timeout = (LASSO_TIMEOUT_CONNECT_SECONDS, LASSO_TIMEOUT_READ_SECONDS)
    while True:
        url = f"{LASSO_BASE}/lasso9/cgi/action/{endpoint_name}.xhr?limit={page_limit}&page={page}"
        payload = None
        retry_with_smaller_page = False
        for attempt in range(LASSO_REQUEST_RETRIES + 1):
            try:
                response = session.get(url, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except requests.ReadTimeout:
                if page_limit > min_page_limit:
                    next_limit = max(min_page_limit, page_limit // 2)
                    if next_limit != page_limit:
                        logger.warning(
                            "Read timeout on Lasso page fetch for %s page %s at limit %s; reducing limit to %s and retrying",
                            endpoint_name,
                            page,
                            page_limit,
                            next_limit,
                        )
                        page_limit = next_limit
                        page = (rows_fetched // page_limit) + 1
                        retry_with_smaller_page = True
                        break
                if attempt >= LASSO_REQUEST_RETRIES:
                    raise
                logger.warning(
                    "Retrying timed out Lasso page fetch for %s page %s (attempt %s/%s)",
                    endpoint_name,
                    page,
                    attempt + 2,
                    LASSO_REQUEST_RETRIES + 1,
                )
            except requests.RequestException:
                if attempt >= LASSO_REQUEST_RETRIES:
                    raise
                logger.warning(
                    "Retrying Lasso page fetch for %s page %s (attempt %s/%s)",
                    endpoint_name,
                    page,
                    attempt + 2,
                    LASSO_REQUEST_RETRIES + 1,
                )
        if retry_with_smaller_page:
            continue
        if not payload:
            break
        all_rows.extend(payload)
        rows_fetched += len(payload)
        if len(payload) < page_limit:
            break
        page += 1
    return all_rows


def _fetch_json(session: requests.Session, endpoint_name: str) -> Any:
    timeout = (LASSO_TIMEOUT_CONNECT_SECONDS, LASSO_TIMEOUT_READ_SECONDS)
    url = f"{LASSO_BASE}/lasso9/cgi/action/{endpoint_name}.xhr"
    for attempt in range(LASSO_REQUEST_RETRIES + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt >= LASSO_REQUEST_RETRIES:
                raise
            logger.warning(
                "Retrying Lasso JSON fetch for %s (attempt %s/%s)",
                endpoint_name,
                attempt + 2,
                LASSO_REQUEST_RETRIES + 1,
            )


def _fetch_lead_report_timeframes(session: requests.Session) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timeout = (LASSO_TIMEOUT_CONNECT_SECONDS, LASSO_TIMEOUT_READ_SECONDS)
    for label, tf_code in LEAD_REPORT_TIMEFRAME_CODES:
        url = f"{LASSO_BASE}/lasso9/cgi/action/lead-report.xhr?tf={tf_code}"
        payload = None
        for attempt in range(LASSO_REQUEST_RETRIES + 1):
            try:
                response = session.get(url, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException:
                if attempt >= LASSO_REQUEST_RETRIES:
                    raise
                logger.warning(
                    "Retrying Lasso lead report for timeframe %s (attempt %s/%s)",
                    label,
                    attempt + 2,
                    LASSO_REQUEST_RETRIES + 1,
                )
        rows.append({
            "timeframe": label,
            "tf_code": tf_code,
            "results": payload.get("results") if isinstance(payload, dict) else payload,
            "raw_payload": payload,
        })
    return rows


def _update_sync_run_progress(
    sync_run_id: int,
    *,
    fetched_counts: Optional[dict[str, Any]] = None,
    materialized_counts: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    assignments: list[str] = []
    params: list[Any] = []

    if fetched_counts is not None:
        assignments.append("fetched_counts = %s")
        params.append(Json(fetched_counts))
    if materialized_counts is not None:
        assignments.append("materialized_counts = %s")
        params.append(Json(materialized_counts))
    if metadata is not None:
        assignments.append("metadata = %s")
        params.append(Json(metadata))

    if not assignments:
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE lasso_sync_runs
                SET {", ".join(assignments)}
                WHERE id = %s
                """,
                (*params, sync_run_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _insert_raw_rows(cursor, table_name: str, sync_run_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    insert_sql = f"""
        INSERT INTO {table_name} (
            sync_run_id,
            lasso_lead_id,
            first_name,
            last_name,
            full_name,
            email,
            city,
            province,
            dealer_id,
            dealer_name,
            current_status,
            submit_date,
            form_submit_date,
            created_date,
            last_interaction,
            raw_payload
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    for row in rows:
        dealer_id = _clean(row.get("dealerId"))
        cursor.execute(
            insert_sql,
            (
                sync_run_id,
                int(row["leadId"]),
                _clean(row.get("firstName")),
                _clean(row.get("lastName")),
                _lead_display_name(row),
                _clean(row.get("email")),
                _clean(row.get("city")),
                _clean(row.get("province")),
                int(dealer_id) if dealer_id and dealer_id.isdigit() else None,
                _clean(row.get("dealerName")),
                _clean(row.get("currentStatus")),
                _parse_datetime(row.get("submitDate")),
                _parse_datetime(row.get("formSubmitDate")),
                _parse_datetime(row.get("createdDate")),
                _parse_datetime(row.get("lastInteraction")),
                Json(row),
            ),
        )


def _insert_report_rows(cursor, table_name: str, sync_run_id: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    if table_name == "lasso_lead_status_report_snapshot":
        for row in rows:
            converted_summary = row.get("converted_summary")
            lead_score = row.get("lead_score")
            cursor.execute(
                """
                INSERT INTO lasso_lead_status_report_snapshot (
                    sync_run_id,
                    dealer_id,
                    dealer_name,
                    review_leads,
                    budget_leads,
                    converted_summary,
                    lead_score,
                    raw_payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sync_run_id,
                    row.get("dealer_id"),
                    row.get("name"),
                    row.get("review_leads"),
                    row.get("budget_leads"),
                    Decimal(str(converted_summary)) if converted_summary not in (None, "") else None,
                    Decimal(str(lead_score)) if lead_score not in (None, "") else None,
                    Json(row),
                ),
            )
        return

    for index, row in enumerate(rows):
        cursor.execute(
            f"""
            INSERT INTO {table_name} (sync_run_id, report_key, raw_payload)
            VALUES (%s, %s, %s)
            """,
            (sync_run_id, str(row.get("timeframe") or row.get("report_key") or index), Json(row)),
        )


def _materialize_dashboard_tables(
    cursor,
    sync_run_id: int,
    unassigned: list[dict[str, Any]],
    active: list[dict[str, Any]],
    history: list[dict[str, Any]],
    lead_report_rows: Optional[list[dict[str, Any]]] = None,
    lead_status_rows: Optional[list[dict[str, Any]]] = None,
    lead_response_rows: Optional[list[dict[str, Any]]] = None,
    preserve_existing_trends: bool = False,
    preserve_existing_full_data: bool = False,
) -> dict[str, int]:
    deduped_history = _dedupe_history_rows(history)
    cursor.execute("DELETE FROM dashboard_unassigned_leads")
    cursor.execute("DELETE FROM dashboard_active_leads")
    cursor.execute("DELETE FROM dashboard_dealer_lead_reporting")
    if not preserve_existing_trends:
        cursor.execute("DELETE FROM dashboard_lead_trends")
    if not preserve_existing_full_data:
        cursor.execute("DELETE FROM dashboard_history_leads")
        cursor.execute("DELETE FROM dashboard_dealer_performance")
        cursor.execute("DELETE FROM dashboard_dealer_project_breakdown")
        cursor.execute("DELETE FROM dashboard_dealer_status")

    unassigned_inserted = 0
    for row in unassigned:
        record_date = _parse_datetime(row.get("submitDate")) or _parse_datetime(row.get("createdDate")) or _parse_datetime(row.get("formSubmitDate"))
        cursor.execute(
            """
            INSERT INTO dashboard_unassigned_leads (
                lasso_lead_id,
                lead_id,
                sync_run_id,
                first_name,
                last_name,
                name,
                email,
                city,
                province,
                location_text,
                current_status,
                record_date,
                last_interaction,
                raw_payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                int(row["leadId"]),
                None,
                sync_run_id,
                _clean(row.get("firstName")),
                _clean(row.get("lastName")),
                _lead_display_name(row),
                _clean(row.get("email")),
                _clean(row.get("city")),
                _clean(row.get("province")),
                _lead_location_text(row),
                _clean(row.get("currentStatus")),
                record_date,
                _parse_datetime(row.get("lastInteraction")),
                Json(row),
            ),
        )
        unassigned_inserted += 1

    active_inserted = 0
    max_age_days = int(round(get_setting_float("lasso_active_queue_max_age_days", 110.0)))
    active_cutoff = datetime.now() - timedelta(days=max_age_days)
    for row in active:
        raw_status = _clean(row.get("currentStatus")) or ""
        anchor = _lead_anchor_datetime(row)
        if raw_status in ACTIVE_QUEUE_EXCLUDED_STATUSES:
            continue
        if anchor is None or anchor < active_cutoff:
            continue
        lead_details = " | ".join(
            part
            for part in [
                _lead_display_name(row),
                _lead_location_text(row),
                _clean(row.get("dealerName")),
                raw_status,
            ]
            if part
        )
        date_assigned = _parse_datetime(row.get("submitDate")) or _parse_datetime(row.get("createdDate")) or _parse_datetime(row.get("formSubmitDate"))
        dealer_id = _clean(row.get("dealerId"))
        cursor.execute(
            """
            INSERT INTO dashboard_active_leads (
                lasso_lead_id,
                lead_id,
                sync_run_id,
                dealer_id,
                dealer_name,
                first_name,
                last_name,
                name,
                email,
                city,
                province,
                location_text,
                current_status,
                date_assigned,
                last_interaction,
                lead_details,
                raw_payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                int(row["leadId"]),
                None,
                sync_run_id,
                int(dealer_id) if dealer_id and dealer_id.isdigit() else None,
                _clean(row.get("dealerName")),
                _clean(row.get("firstName")),
                _clean(row.get("lastName")),
                _lead_display_name(row),
                _clean(row.get("email")),
                _clean(row.get("city")),
                _clean(row.get("province")),
                _lead_location_text(row),
                raw_status,
                date_assigned,
                _parse_datetime(row.get("lastInteraction")) or anchor,
                lead_details,
                Json(row),
            ),
        )
        active_inserted += 1

    deduped = _dedupe_rows(unassigned, active, deduped_history)
    reporting_inserted = 0
    source_rows = lead_report_rows or []
    if not source_rows:
        today = date.today()
        for label, _ in TIMEFRAMES:
            start_date, end_date = _timeframe_bounds(label, today)
            subset = []
            for row in deduped:
                row_date = _lead_date_only(row)
                if row_date is None:
                    continue
                if start_date and row_date < start_date:
                    continue
                if end_date and row_date >= end_date:
                    continue
                subset.append(row)

            total_leads = len(subset)
            converted_count = sum(1 for row in subset if _status_bucket(row.get("currentStatus")) == "converted")
            dead_count = sum(1 for row in subset if _status_bucket(row.get("currentStatus")) == "dead")
            active_count = sum(1 for row in subset if _status_bucket(row.get("currentStatus")) == "active")

            converted_pct = round((converted_count / total_leads) * 100, 1) if total_leads else 0.0
            dead_pct = round((dead_count / total_leads) * 100, 1) if total_leads else 0.0
            active_pct = round((active_count / total_leads) * 100, 1) if total_leads else 0.0

            cursor.execute(
                """
                INSERT INTO dashboard_dealer_lead_reporting (
                    timeframe,
                    converted_count,
                    converted_pct,
                    dead_count,
                    dead_pct,
                    active_count,
                    active_pct,
                    total_leads,
                    sync_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    label,
                    converted_count,
                    converted_pct,
                    dead_count,
                    dead_pct,
                    active_count,
                    active_pct,
                    total_leads,
                    sync_run_id,
                ),
            )
            reporting_inserted += 1
    else:
        for row in source_rows:
            results = row.get("results") or {}
            total_leads = int(results.get("total") or 0)
            converted_count = int(results.get("converted") or 0)
            dead_count = int(results.get("dead") or 0)
            active_count = int(results.get("active") or 0)
            converted_pct = round((converted_count / total_leads) * 100, 1) if total_leads else 0.0
            dead_pct = round((dead_count / total_leads) * 100, 1) if total_leads else 0.0
            active_pct = round((active_count / total_leads) * 100, 1) if total_leads else 0.0

            cursor.execute(
                """
                INSERT INTO dashboard_dealer_lead_reporting (
                    timeframe,
                    converted_count,
                    converted_pct,
                    dead_count,
                    dead_pct,
                    active_count,
                    active_pct,
                    total_leads,
                    sync_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["timeframe"],
                    converted_count,
                    converted_pct,
                    dead_count,
                    dead_pct,
                    active_count,
                    active_pct,
                    total_leads,
                    sync_run_id,
                ),
            )
            reporting_inserted += 1

    if preserve_existing_trends:
        cursor.execute("SELECT COUNT(*) AS cnt FROM dashboard_lead_trends")
        trend_count = cursor.fetchone()["cnt"]
    else:
        monthly_buckets: defaultdict[date, dict[str, int]] = defaultdict(lambda: {"total": 0, "converted": 0, "dead": 0, "active": 0})
        for row in deduped:
            row_date = _lead_date_only(row)
            if row_date is None:
                continue
            period_start = row_date.replace(day=1)
            monthly_buckets[period_start]["total"] += 1
            bucket = _status_bucket(row.get("currentStatus"))
            monthly_buckets[period_start][bucket] += 1

        for period_start, stats in sorted(monthly_buckets.items()):
            cursor.execute(
                """
                INSERT INTO dashboard_lead_trends (
                    period_start,
                    total_count,
                    converted_count,
                    dead_count,
                    active_count,
                    sync_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    period_start,
                    stats["total"],
                    stats["converted"],
                    stats["dead"],
                    stats["active"],
                    sync_run_id,
                ),
            )
        trend_count = len(monthly_buckets)

    history_count = 0
    performance_count = 0
    project_count = 0
    status_count = 0

    if preserve_existing_full_data:
        cursor.execute("SELECT COUNT(*) AS cnt FROM dashboard_history_leads")
        history_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) AS cnt FROM dashboard_dealer_performance")
        performance_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) AS cnt FROM dashboard_dealer_project_breakdown")
        project_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) AS cnt FROM dashboard_dealer_status")
        status_count = cursor.fetchone()["cnt"]
    else:
        dealer_names = _dealer_name_map(
            active,
            unassigned,
            deduped_history,
            lead_status_rows or [],
            lead_response_rows or [],
        )
        project_buckets: defaultdict[tuple[Optional[int], str], dict[str, int]] = defaultdict(
            lambda: {
                "sqft_1_499": 0,
                "sqft_500_999": 0,
                "sqft_1000_3499": 0,
                "sqft_3500_7499": 0,
                "sqft_7500_19999": 0,
                "sqft_20000_plus": 0,
                "total_leads": 0,
            }
        )

        for row in deduped_history:
            dealer_id = _safe_int(row.get("dealerId"))
            dealer_name = (
                dealer_names.get(dealer_id) if dealer_id is not None else None
            ) or _clean(row.get("dealerName")) or _clean(row.get("sourceDealer"))

            address1 = _clean(row.get("address1"))
            address2 = _clean(row.get("address2"))
            if address1 and address2:
                address = f"{address1}, {address2}"
            else:
                address = address1 or address2

            square_text = _clean(row.get("squareFootage")) or _clean(row.get("customPick1"))
            square_value = _parse_square_footage_value(square_text)

            cursor.execute(
                """
                INSERT INTO dashboard_history_leads (
                    lasso_lead_id,
                    sync_run_id,
                    dealer_id,
                    dealer_name,
                    first_name,
                    last_name,
                    name,
                    email,
                    phone,
                    address,
                    city,
                    province,
                    postal_code,
                    current_status,
                    status_bucket,
                    submit_date,
                    form_submit_date,
                    created_date,
                    last_interaction,
                    project_type,
                    product_type,
                    square_footage_text,
                    square_footage_value,
                    business_category,
                    company_name,
                    lead_source,
                    landing_page,
                    landing_page_url,
                    landing_page_variant,
                    utm_source,
                    utm_medium,
                    utm_campaign,
                    utm_content,
                    comments,
                    value_of_order,
                    raw_payload
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (lasso_lead_id) DO UPDATE SET
                    sync_run_id = EXCLUDED.sync_run_id,
                    dealer_id = EXCLUDED.dealer_id,
                    dealer_name = EXCLUDED.dealer_name,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    address = EXCLUDED.address,
                    city = EXCLUDED.city,
                    province = EXCLUDED.province,
                    postal_code = EXCLUDED.postal_code,
                    current_status = EXCLUDED.current_status,
                    status_bucket = EXCLUDED.status_bucket,
                    submit_date = EXCLUDED.submit_date,
                    form_submit_date = EXCLUDED.form_submit_date,
                    created_date = EXCLUDED.created_date,
                    last_interaction = EXCLUDED.last_interaction,
                    project_type = EXCLUDED.project_type,
                    product_type = EXCLUDED.product_type,
                    square_footage_text = EXCLUDED.square_footage_text,
                    square_footage_value = EXCLUDED.square_footage_value,
                    business_category = EXCLUDED.business_category,
                    company_name = EXCLUDED.company_name,
                    lead_source = EXCLUDED.lead_source,
                    landing_page = EXCLUDED.landing_page,
                    landing_page_url = EXCLUDED.landing_page_url,
                    landing_page_variant = EXCLUDED.landing_page_variant,
                    utm_source = EXCLUDED.utm_source,
                    utm_medium = EXCLUDED.utm_medium,
                    utm_campaign = EXCLUDED.utm_campaign,
                    utm_content = EXCLUDED.utm_content,
                    comments = EXCLUDED.comments,
                    value_of_order = EXCLUDED.value_of_order,
                    raw_payload = EXCLUDED.raw_payload,
                    synced_at = CURRENT_TIMESTAMP
                """,
                (
                    int(row["leadId"]),
                    sync_run_id,
                    dealer_id,
                    dealer_name,
                    _clean(row.get("firstName")),
                    _clean(row.get("lastName")),
                    _lead_display_name(row),
                    _clean(row.get("email")),
                    _clean(row.get("primaryPhone")) or _clean(row.get("cellPhone")) or _clean(row.get("workPhone")),
                    address,
                    _clean(row.get("city")),
                    _clean(row.get("province")),
                    _clean(row.get("postal")),
                    _clean(row.get("currentStatus")) or _clean(row.get("status")),
                    _status_bucket(row.get("currentStatus") or row.get("status")),
                    _parse_datetime(row.get("submitDate")),
                    _parse_datetime(row.get("formSubmitDate")),
                    _parse_datetime(row.get("createdDate")),
                    _parse_datetime(row.get("lastInteraction")),
                    _clean(row.get("projectType")),
                    _clean(row.get("productType")),
                    square_text,
                    square_value,
                    _clean(row.get("businessCategory")),
                    _clean(row.get("companyName")),
                    _clean(row.get("source")),
                    _clean(row.get("landingPageName")),
                    _clean(row.get("landingUrl")),
                    _clean(row.get("landingVariant")),
                    _clean(row.get("utmSource")),
                    _clean(row.get("utmMedium")),
                    _clean(row.get("utmCampaign")),
                    _clean(row.get("utmContent")),
                    _clean(row.get("comments")),
                    _safe_decimal(row.get("valueOfOrder")),
                    Json(row),
                ),
            )
            history_count += 1

            bucket_key = _project_bucket_key(square_text)
            if dealer_name:
                bucket = project_buckets[(dealer_id, dealer_name)]
                bucket["total_leads"] += 1
                if bucket_key:
                    bucket[bucket_key] += 1

        for (dealer_id, dealer_name), counts in sorted(
            project_buckets.items(),
            key=lambda item: (item[0][1] or "").lower(),
        ):
            cursor.execute(
                """
                INSERT INTO dashboard_dealer_project_breakdown (
                    dealer_id,
                    dealer_name,
                    sqft_1_499,
                    sqft_500_999,
                    sqft_1000_3499,
                    sqft_3500_7499,
                    sqft_7500_19999,
                    sqft_20000_plus,
                    total_leads,
                    sync_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    dealer_id,
                    dealer_name,
                    counts["sqft_1_499"],
                    counts["sqft_500_999"],
                    counts["sqft_1000_3499"],
                    counts["sqft_3500_7499"],
                    counts["sqft_7500_19999"],
                    counts["sqft_20000_plus"],
                    counts["total_leads"],
                    sync_run_id,
                ),
            )
            project_count += 1

        for row in lead_response_rows or []:
            active_leads = _safe_int(row.get("active_leads")) or 0
            converted = _safe_int(row.get("won_leads")) or 0
            dead = _safe_int(row.get("dead_leads")) or 0
            total = active_leads + converted + dead
            cursor.execute(
                """
                INSERT INTO dashboard_dealer_performance (
                    dealer_id,
                    dealer_name,
                    active_leads,
                    converted_count,
                    dead_count,
                    total_leads,
                    avg_response_hours,
                    avg_response_str,
                    sync_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    _safe_int(row.get("dealer_id")),
                    _clean(row.get("name")),
                    active_leads,
                    converted,
                    dead,
                    total,
                    _parse_avg_response_hours(row.get("avg_response")),
                    _clean(row.get("avg_response")),
                    sync_run_id,
                ),
            )
            performance_count += 1

        for row in lead_status_rows or []:
            cursor.execute(
                """
                INSERT INTO dashboard_dealer_status (
                    dealer_id,
                    dealer_name,
                    reviewing_undecided,
                    building_budget,
                    converted_total_value,
                    lead_score_pct,
                    sync_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    _safe_int(row.get("dealer_id")),
                    _clean(row.get("name")),
                    _safe_int(row.get("review_leads")) or 0,
                    _safe_int(row.get("budget_leads")) or 0,
                    _safe_decimal(row.get("converted_summary")) or Decimal("0"),
                    _safe_decimal(row.get("lead_score")) or Decimal("0"),
                    sync_run_id,
                ),
            )
            status_count += 1

    return {
        "dashboard_unassigned_leads": unassigned_inserted,
        "dashboard_active_leads": active_inserted,
        "dashboard_dealer_lead_reporting": reporting_inserted,
        "dashboard_lead_trends": trend_count,
        "dashboard_history_leads": history_count,
        "dashboard_dealer_performance": performance_count,
        "dashboard_dealer_project_breakdown": project_count,
        "dashboard_dealer_status": status_count,
    }


def _extract_status_report_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return results
    return payload if isinstance(payload, list) else []


def _extract_generic_report_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return payload["results"]
        return [payload]
    return payload if isinstance(payload, list) else []


def _create_sync_run(sync_type: str, actor: str) -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO lasso_sync_runs (sync_type, status, actor, started_at)
                VALUES (%s, 'running', %s, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (sync_type, actor),
            )
            sync_run_id = cursor.fetchone()["id"]
        conn.commit()
        return sync_run_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _running_sync_rows() -> list[dict[str, Any]]:
    return execute_query(
        """
        SELECT *
        FROM lasso_sync_runs
        WHERE status = 'running'
        ORDER BY started_at ASC, id ASC
        """
    )


def _clear_stale_running_syncs() -> None:
    cutoff = datetime.utcnow() - timedelta(minutes=LASSO_SYNC_STALE_MINUTES)
    stale_rows = execute_query(
        """
        SELECT id
        FROM lasso_sync_runs
        WHERE status = 'running' AND started_at < %s
        """,
        (cutoff,),
    )
    if not stale_rows:
        return

    stale_ids = [row["id"] for row in stale_rows]
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE lasso_sync_runs
                SET status = 'failed',
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = %s
                WHERE id = ANY(%s)
                """,
                ("Marked failed after exceeding stale sync timeout window", stale_ids),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _current_running_sync() -> Optional[dict[str, Any]]:
    _clear_stale_running_syncs()
    rows = _running_sync_rows()
    return rows[0] if rows else None


def _complete_sync_run(sync_run_id: int, status: str, fetched_counts: dict[str, Any], materialized_counts: dict[str, Any], error_message: Optional[str] = None) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE lasso_sync_runs
                SET status = %s,
                    completed_at = CURRENT_TIMESTAMP,
                    fetched_counts = %s,
                    materialized_counts = %s,
                    error_message = %s
                WHERE id = %s
                """,
                (status, Json(fetched_counts), Json(materialized_counts), error_message, sync_run_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_lasso_dashboard_sync(sync_type: str = "full", actor: str = "system") -> dict[str, Any]:
    if sync_type not in {"fast", "full"}:
        raise ValueError(f"Unsupported sync_type: {sync_type}")
    started_at = timed_call_start()
    sync_run_id = _create_sync_run(sync_type, actor)
    fetched_counts: dict[str, Any] = {}
    materialized_counts: dict[str, Any] = {}
    sync_metadata: dict[str, Any] = {"stage": "starting", "sync_type": sync_type}
    try:
        _update_sync_run_progress(sync_run_id, metadata=sync_metadata)
        session = _fetch_lasso_session()
        sync_metadata = {"stage": "fetching_unassigned", "sync_type": sync_type}
        _update_sync_run_progress(sync_run_id, metadata=sync_metadata)
        unassigned = _fetch_paginated(session, "unassigned-leads", limit=LASSO_UNASSIGNED_PAGE_LIMIT)
        fetched_counts.update({"sync_type": sync_type, "unassigned": len(unassigned)})
        sync_metadata = {"stage": "fetching_active", "sync_type": sync_type}
        _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, metadata=sync_metadata)
        active = _fetch_paginated(session, "active-leads", limit=LASSO_ACTIVE_PAGE_LIMIT)
        fetched_counts.update({"active": len(active)})
        sync_metadata = {"stage": "fetching_lead_report", "sync_type": sync_type}
        _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, metadata=sync_metadata)
        lead_report_rows = _fetch_lead_report_timeframes(session)
        fetched_counts.update({"lead_report_rows": len(lead_report_rows)})
        history: list[dict[str, Any]] = []
        lead_status_payload: Any = []
        lead_response_payload: Any = []
        preserve_existing_trends = sync_type == "fast"
        preserve_existing_full_data = sync_type == "fast"

        if sync_type == "full":
            sync_metadata = {"stage": "fetching_history", "sync_type": sync_type}
            _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, metadata=sync_metadata)
            history = _fetch_paginated(session, "admin-history-leads", limit=LASSO_HISTORY_PAGE_LIMIT)
            fetched_counts.update({"history": len(history)})
            sync_metadata = {"stage": "fetching_lead_status", "sync_type": sync_type}
            _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, metadata=sync_metadata)
            lead_status_payload = _fetch_json(session, "lead-status-report")
            fetched_counts.update({"lead_status_rows": len(_extract_status_report_rows(lead_status_payload))})
            sync_metadata = {"stage": "fetching_lead_response", "sync_type": sync_type}
            _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, metadata=sync_metadata)
            lead_response_payload = _fetch_json(session, "lead-response-report")
            fetched_counts.update({"lead_response_rows": len(_extract_generic_report_rows(lead_response_payload))})

        fetched_counts = {
            "sync_type": sync_type,
            "unassigned": len(unassigned),
            "active": len(active),
            "history": len(history),
            "lead_report_rows": len(lead_report_rows),
            "lead_status_rows": len(_extract_status_report_rows(lead_status_payload)),
            "lead_response_rows": len(_extract_generic_report_rows(lead_response_payload)),
            "dealer_project_rows": 0,
        }
        sync_metadata = {"stage": "materializing", "sync_type": sync_type}
        _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, metadata=sync_metadata)

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                materialized_counts = _materialize_dashboard_tables(
                    cursor,
                    sync_run_id,
                    unassigned,
                    active,
                    history,
                    lead_report_rows=lead_report_rows,
                    lead_status_rows=_extract_status_report_rows(lead_status_payload),
                    lead_response_rows=_extract_generic_report_rows(lead_response_payload),
                    preserve_existing_trends=preserve_existing_trends,
                    preserve_existing_full_data=preserve_existing_full_data,
                )
            conn.commit()
            sync_metadata = {"stage": "completed", "sync_type": sync_type}
            _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, materialized_counts=materialized_counts, metadata=sync_metadata)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        _complete_sync_run(sync_run_id, "completed", fetched_counts, materialized_counts)
        log_event(
            event_type="LASSO_DASHBOARD_SYNC",
            entity_type="lasso_dashboard",
            entity_id=str(sync_run_id),
            actor=actor,
            latency_ms=timed_call_latency_ms(started_at),
            payload={
                "sync_type": sync_type,
                "fetched_counts": fetched_counts,
                "materialized_counts": materialized_counts,
            },
        )
        return {
            "sync_run_id": sync_run_id,
            "sync_type": sync_type,
            "fetched_counts": fetched_counts,
            "materialized_counts": materialized_counts,
        }
    except Exception as exc:
        logger.exception("Lasso dashboard sync failed")
        sync_metadata = {
            "stage": sync_metadata.get("stage", "failed"),
            "sync_type": sync_type,
            "error_type": type(exc).__name__,
        }
        _update_sync_run_progress(sync_run_id, fetched_counts=fetched_counts, materialized_counts=materialized_counts, metadata=sync_metadata)
        _complete_sync_run(sync_run_id, "failed", fetched_counts, materialized_counts, str(exc))
        log_event(
            event_type="LASSO_DASHBOARD_SYNC_FAILED",
            entity_type="lasso_dashboard",
            entity_id=str(sync_run_id),
            actor=actor,
            latency_ms=timed_call_latency_ms(started_at),
            payload={"sync_type": sync_type, "error": str(exc)},
        )
        raise


def _background_sync(sync_type: str, actor: str) -> None:
    with SYNC_LOCK:
        SYNC_STATUS["in_progress"] = True
        SYNC_STATUS["sync_type"] = sync_type
        SYNC_STATUS["started_at"] = datetime.utcnow().isoformat()
        SYNC_STATUS["last_error"] = None
    try:
        run_lasso_dashboard_sync(sync_type=sync_type, actor=actor)
    except Exception as exc:  # pragma: no cover - defensive guard
        with SYNC_LOCK:
            SYNC_STATUS["last_error"] = str(exc)
    finally:
        with SYNC_LOCK:
            SYNC_STATUS["in_progress"] = False
            SYNC_STATUS["sync_type"] = None
            SYNC_STATUS["started_at"] = None


def start_lasso_dashboard_sync(sync_type: str = "full", actor: str = "system", force: bool = False) -> dict[str, Any]:
    running_sync = _current_running_sync()
    if running_sync and not force:
        return {
            "started": False,
            "reason": "sync already in progress",
            "running_sync_id": running_sync["id"],
            "running_sync_type": running_sync.get("sync_type"),
        }

    with SYNC_LOCK:
        if SYNC_STATUS["in_progress"] and not force:
            return {"started": False, "reason": "sync already in progress"}
        if SYNC_STATUS["in_progress"] and force:
            return {"started": False, "reason": "sync already in progress"}
        thread = threading.Thread(target=_background_sync, args=(sync_type, actor), daemon=True)
        thread.start()
    return {"started": True, "sync_type": sync_type}


def _last_successful_sync() -> Optional[dict[str, Any]]:
    rows = execute_query(
        """
        SELECT *
        FROM lasso_sync_runs
        WHERE status = 'completed'
        ORDER BY completed_at DESC NULLS LAST, id DESC
        LIMIT 1
        """
    )
    return rows[0] if rows else None


def maybe_trigger_stale_sync(actor: str = "system") -> None:
    if SYNC_STATUS["in_progress"] or _current_running_sync():
        return
    last_sync = _last_successful_sync()
    interval_minutes = int(round(get_setting_float("lasso_fast_sync_interval_minutes", 5.0)))
    should_sync = last_sync is None
    if last_sync and last_sync.get("completed_at"):
        should_sync = last_sync["completed_at"] < datetime.utcnow() - timedelta(minutes=interval_minutes)
    if should_sync:
        start_lasso_dashboard_sync(sync_type="fast", actor=actor)


def get_lasso_dashboard_status() -> dict[str, Any]:
    # Use one pooled connection for all queries — avoids 13 separate TCP
    # handshakes to Azure Postgres while keeping straightforward SQL.
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, sync_type, completed_at, error_message
                FROM lasso_sync_runs
                WHERE status = 'completed'
                ORDER BY completed_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            )
            last_success = cur.fetchone() or {}

            cur.execute(
                """
                SELECT completed_at FROM lasso_sync_runs
                WHERE status = 'completed' AND sync_type = 'full'
                ORDER BY completed_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            )
            last_full_row = cur.fetchone() or {}

            cur.execute(
                """
                SELECT completed_at, error_message FROM lasso_sync_runs
                WHERE status = 'failed'
                ORDER BY completed_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            )
            last_failed = cur.fetchone() or {}

            cur.execute(
                """
                SELECT id, sync_type, started_at FROM lasso_sync_runs
                WHERE status = 'running'
                ORDER BY started_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            )
            running = cur.fetchone() or {}

            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM dashboard_unassigned_leads)               AS unassigned_count,
                    (SELECT COUNT(*) FROM dashboard_active_leads)                   AS active_count,
                    (SELECT COUNT(*) FROM dashboard_history_leads)                  AS history_count,
                    (SELECT MAX(synced_at) FROM dashboard_dealer_lead_reporting)    AS report_synced_at,
                    (SELECT MAX(synced_at) FROM dashboard_history_leads)            AS history_synced_at,
                    (SELECT MAX(synced_at) FROM dashboard_dealer_performance)       AS perf_synced_at,
                    (SELECT MAX(synced_at) FROM dashboard_dealer_project_breakdown) AS proj_synced_at,
                    (SELECT MAX(synced_at) FROM dashboard_dealer_status)            AS status_synced_at
                """
            )
            tbl = cur.fetchone() or {}

            cur.execute(
                """
                SELECT key, value_text FROM settings
                WHERE key IN (
                    'lasso_fast_sync_interval_minutes',
                    'lasso_full_sync_interval_minutes',
                    'lasso_active_queue_max_age_days'
                )
                """
            )
            sett = {row["key"]: row["value_text"] for row in cur.fetchall()}
    finally:
        release_db_connection(conn)

    def _iso(val):
        if val is None:
            return None
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return val

    def _sett_float(key: str, default: float) -> int:
        v = sett.get(key)
        try:
            return int(round(float(v))) if v is not None else int(default)
        except (TypeError, ValueError):
            return int(default)

    visible_error = SYNC_STATUS["last_error"]
    if not visible_error and last_failed.get("completed_at"):
        last_success_at = last_success.get("completed_at")
        failed_at = last_failed["completed_at"]
        if last_success_at is None or failed_at > last_success_at:
            visible_error = last_failed.get("error_message")

    in_progress = SYNC_STATUS["in_progress"] or bool(running)

    return {
        "sync_in_progress": in_progress,
        "sync_type": SYNC_STATUS["sync_type"] or (running.get("sync_type") if running else None),
        "started_at": SYNC_STATUS["started_at"] or _iso(running.get("started_at") if running else None),
        "last_error": visible_error,
        "last_successful_sync_at": _iso(last_success.get("completed_at")),
        "last_successful_sync_type": last_success.get("sync_type"),
        "last_full_successful_sync_at": _iso(last_full_row.get("completed_at")),
        "unassigned_count": tbl.get("unassigned_count", 0),
        "active_count": tbl.get("active_count", 0),
        "history_count": tbl.get("history_count", 0),
        "dealer_lead_reporting_synced_at": _iso(tbl.get("report_synced_at")),
        "history_synced_at": _iso(tbl.get("history_synced_at")),
        "dealer_performance_synced_at": _iso(tbl.get("perf_synced_at")),
        "dealer_project_breakdown_synced_at": _iso(tbl.get("proj_synced_at")),
        "dealer_status_synced_at": _iso(tbl.get("status_synced_at")),
        "fast_sync_interval_minutes": _sett_float("lasso_fast_sync_interval_minutes", 5.0),
        "full_sync_interval_minutes": _sett_float("lasso_full_sync_interval_minutes", 15.0),
        "active_queue_max_age_days": _sett_float("lasso_active_queue_max_age_days", 110.0),
    }
