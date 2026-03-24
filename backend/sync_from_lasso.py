"""
Sync data from Lasso 9 (lap.conveniencegroup.com) to our PostgreSQL database.

1. Logs in via session cookie
2. Pulls dealers, unassigned leads, active leads (paginated)
3. Pulls report data (lead-report, lead-status-report, lead-response-report, dealer-project-report)
4. Clears existing data and inserts fresh
"""

import requests
import re
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

from audit_logger import log_event, timed_call_latency_ms, timed_call_start

# Config
LASSO_BASE = "https://lap.conveniencegroup.com"
LASSO_USER = "info@windowfilmcanada.ca"
LASSO_PASS = "890*()iopIOP@WFC"
DB_URL = "postgresql://q4gems_admin:890*()iopIOP@capstone25.postgres.database.azure.com:5432/capstone25db"

STATUS_MAP = {
    "new": "active",
    "converted sale": "converted",
    "dead lead": "dead",
    "follow up": "follow_up",
    "called": "active",
    "client reviewing/undecided": "active",
    "client building budget": "active",
    "current": "active",
    "": "active",
}


def parse_datetime(val):
    if not val or val.strip() == "" or val.strip().lower() == "invalid date":
        return None
    val = val.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def parse_float(val):
    if not val or str(val).strip() == "":
        return None
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def clean(val):
    if val is None:
        return None
    val = str(val).strip()
    return val if val else None


def lasso_login():
    """Login to Lasso 9 and return authenticated session."""
    print("Logging in to Lasso 9...")
    s = requests.Session()

    # GET login page to capture session token from form action
    r = s.get(f"{LASSO_BASE}/lasso9/cgi/action/login.html")
    match = re.search(r'form action="([^"]+)"', r.text)
    if not match:
        raise Exception("Could not find form action on login page")

    form_action = match.group(1)
    login_url = f"{LASSO_BASE}/lasso9/cgi/action/login.html{form_action}"

    r = s.post(login_url, data={
        "username": LASSO_USER,
        "password": LASSO_PASS,
        "do-login": "Login",
    }, allow_redirects=True)

    if "login.html" in r.url and "username" in r.text[:1000]:
        raise Exception("Login failed — still on login page")

    print(f"  Logged in. Session: {dict(s.cookies)}")
    return s


def fetch_paginated(session, endpoint_name, limit=50):
    """Fetch all pages from a paginated endpoint."""
    all_records = []
    page = 1
    while True:
        url = f"{LASSO_BASE}/lasso9/cgi/action/{endpoint_name}.xhr?limit={limit}&page={page}"
        r = session.get(url)
        if r.status_code != 200:
            print(f"  {endpoint_name} page {page}: HTTP {r.status_code}")
            break
        data = r.json()
        if not data:
            break
        all_records.extend(data)
        page += 1

    print(f"  {endpoint_name}: {len(all_records)} records ({page - 1} pages)")
    return all_records


def fetch_json(session, endpoint_name):
    """Fetch a single JSON endpoint."""
    url = f"{LASSO_BASE}/lasso9/cgi/action/{endpoint_name}.xhr"
    r = session.get(url)
    data = r.json()
    print(f"  {endpoint_name}: {r.status_code}")
    return data


def sync_dealers(conn, dealers_data):
    """Clear and insert dealers from Lasso data."""
    print(f"\n{'='*60}")
    print(f"SYNCING {len(dealers_data)} DEALERS")
    print(f"{'='*60}")

    cur = conn.cursor()
    cur.execute("DELETE FROM dealers")

    for d in dealers_data:
        cur.execute("""
            INSERT INTO dealers (id, name, city, province, email, notification_email, phone, is_active)
            VALUES (%s, %s, %s, %s, '', '', '', TRUE)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name, city = EXCLUDED.city, province = EXCLUDED.province, is_active = TRUE
        """, (
            d["id"],
            d["name"].strip(),
            clean(d.get("city")) or "",
            clean(d.get("province")) or "",
        ))

    conn.commit()

    # Reset sequence
    cur.execute("SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM dealers")
    next_id = cur.fetchone()["next_id"]
    cur.execute(f"ALTER SEQUENCE dealers_id_seq RESTART WITH {next_id}")
    conn.commit()

    print(f"  Inserted {len(dealers_data)} dealers")
    return len(dealers_data)


def sync_leads(conn, unassigned, active, history):
    """Clear and insert all leads from Lasso: unassigned + active + history."""
    all_leads = unassigned + active + history
    print(f"\n{'='*60}")
    print(f"SYNCING {len(all_leads)} LEADS ({len(unassigned)} unassigned + {len(active)} active + {len(history)} history)")
    print(f"{'='*60}")

    cur = conn.cursor()

    # Clear existing leads and related tables
    cur.execute("DELETE FROM lead_logs")
    cur.execute("DELETE FROM lead_assignments")
    cur.execute("DELETE FROM leads")
    conn.commit()
    print("  Cleared existing leads, logs, assignments")

    # Build dealer name->id map from DB
    cur.execute("SELECT id, LOWER(TRIM(name)) as lname FROM dealers")
    rows = cur.fetchall()
    dealer_map = {r["lname"]: r["id"] for r in rows}
    dealer_id_set = {r["id"] for r in rows}

    inserted = 0
    errors = 0
    seen = set()

    for lead in all_leads:
        lead_id = lead.get("leadId")

        # Dedup by leadId within the sync
        if lead_id in seen:
            continue
        seen.add(lead_id)

        first_name = clean(lead.get("firstName")) or ""
        last_name = clean(lead.get("lastName")) or ""
        email = clean(lead.get("email"))
        full_name = f"{first_name} {last_name}".strip() or email or "Unknown"

        # Map status
        raw_status = (clean(lead.get("currentStatus")) or "").lower()
        status = STATUS_MAP.get(raw_status, "active")

        # Map dealer
        dealer_id_str = clean(lead.get("dealerId"))
        dealer_id = int(dealer_id_str) if dealer_id_str and dealer_id_str.isdigit() else None
        dealer_name = clean(lead.get("dealerName"))

        # If no dealerId but we have dealerName, look it up
        if not dealer_id and dealer_name:
            dealer_id = dealer_map.get(dealer_name.strip().lower())

        # Verify dealer_id exists in our dealers table; if not, clear it
        # (some leads reference dealers not in the active list)
        if dealer_id and dealer_id not in dealer_id_set:
            dealer_id = None

        submit_date = parse_datetime(lead.get("submitDate"))
        form_submit_date = parse_datetime(lead.get("formSubmitDate")) or submit_date
        created_date = parse_datetime(lead.get("createdDate")) or submit_date

        address = clean(lead.get("address1"))
        address2 = clean(lead.get("address2"))
        if address and address2:
            address = f"{address}, {address2}"
        elif address2:
            address = address2

        # Parse value_of_order (history leads have this)
        value_of_order = parse_float(lead.get("valueOfOrder"))
        # Parse square footage text
        sq_footage = clean(lead.get("squareFootage"))

        try:
            cur.execute("""
                INSERT INTO leads (
                    name, first_name, last_name, email, phone, address,
                    city, province, postal_code, job_title, company_name,
                    product_type, product_type_2, product_type_3,
                    custom_pick_1, custom_pick_3, project_city, project_type,
                    business_category, dealer_email, lead_source, source, opt_in,
                    landing_page, landing_page_url, landing_page_variant,
                    utm_source, utm_medium, utm_campaign, utm_content,
                    comments, status,
                    form_submit_date, created_at,
                    assigned_dealer_id, final_dealer_selection,
                    value_of_order
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s
                )
            """, (
                full_name, first_name, last_name, email,
                clean(lead.get("primaryPhone")),
                address,
                clean(lead.get("city")),
                clean(lead.get("province")),
                clean(lead.get("postal")),
                clean(lead.get("jobTitle")),
                clean(lead.get("companyName")),
                clean(lead.get("productType")),
                clean(lead.get("productType2")),
                clean(lead.get("productType3")),
                clean(lead.get("customPick1")) or sq_footage,
                clean(lead.get("customPick3")),
                clean(lead.get("projectCity")),
                clean(lead.get("projectType")),
                clean(lead.get("businessCategory")),
                clean(lead.get("dealerEmail")),
                clean(lead.get("source")),
                clean(lead.get("source")),
                lead.get("optIn") == "1",
                clean(lead.get("landingPageName")),
                clean(lead.get("landingUrl")),
                clean(lead.get("landingVariant")),
                clean(lead.get("utmSource")),
                clean(lead.get("utmMedium")),
                clean(lead.get("utmCampaign")),
                clean(lead.get("utmContent")),
                clean(lead.get("comments")),
                status,
                form_submit_date,
                created_date or submit_date,
                dealer_id,
                dealer_name,
                value_of_order,
            ))
            conn.commit()
            inserted += 1

            if inserted % 500 == 0:
                print(f"  Progress: {inserted} inserted")

        except Exception as e:
            conn.rollback()
            errors += 1
            if errors <= 3:
                print(f"  Error (leadId={lead_id}): {str(e)[:200]}")

    # Reset sequence
    cur.execute("SELECT setval('leads_id_seq', COALESCE((SELECT MAX(id) FROM leads), 1))")
    conn.commit()

    print(f"  Done: {inserted} inserted, {errors} errors")
    return {"inserted": inserted, "errors": errors, "source_total": len(all_leads)}


def print_summary(conn):
    cur = conn.cursor()
    print(f"\n{'='*60}")
    print("SYNC SUMMARY")
    print(f"{'='*60}")

    cur.execute("SELECT COUNT(*) as cnt FROM dealers WHERE is_active = TRUE")
    print(f"Dealers: {cur.fetchone()['cnt']}")

    cur.execute("SELECT COUNT(*) as cnt FROM leads")
    print(f"Total leads: {cur.fetchone()['cnt']}")

    cur.execute("SELECT status, COUNT(*) as cnt FROM leads GROUP BY status ORDER BY cnt DESC")
    print("Status breakdown:")
    for r in cur.fetchall():
        print(f"  {r['status']}: {r['cnt']}")

    cur.execute("SELECT COUNT(*) as cnt FROM leads WHERE assigned_dealer_id IS NOT NULL")
    print(f"Leads assigned to dealers: {cur.fetchone()['cnt']}")

    cur.execute("""
        SELECT d.name, COUNT(l.id) as cnt
        FROM dealers d
        LEFT JOIN leads l ON l.assigned_dealer_id = d.id
        GROUP BY d.name
        ORDER BY cnt DESC
        LIMIT 10
    """)
    print("Top 10 dealers by assigned leads:")
    for r in cur.fetchall():
        print(f"  {r['name']}: {r['cnt']}")


if __name__ == "__main__":
    started_at = timed_call_start()
    # Step 1: Login to Lasso
    try:
        session = lasso_login()

        # Step 2: Fetch all data from Lasso
        print("\nFetching data from Lasso 9...")
        dealers_data = fetch_json(session, "list-dealers")
        unassigned = fetch_paginated(session, "unassigned-leads")
        active = fetch_paginated(session, "active-leads")
        history = fetch_paginated(session, "admin-history-leads")

        # Step 3: Fetch report data
        lead_report = fetch_json(session, "lead-report")
        status_report = fetch_json(session, "lead-status-report")
        response_report = fetch_json(session, "lead-response-report")
        project_report = fetch_json(session, "dealer-project-report")

        print(f"\nLasso data summary:")
        print(f"  Dealers: {len(dealers_data)}")
        print(f"  Unassigned leads: {len(unassigned)}")
        print(f"  Active leads: {len(active)}")
        print(f"  History leads: {len(history)}")

        # Step 4: Sync to database
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

        try:
            # Must clear leads before dealers due to FK constraint
            cur = conn.cursor()
            cur.execute("DELETE FROM lead_logs")
            cur.execute("DELETE FROM lead_assignments")
            cur.execute("DELETE FROM leads")
            conn.commit()
            print("Pre-cleared leads/logs/assignments (FK ordering)")

            dealers_inserted = sync_dealers(conn, dealers_data)
            lead_stats = sync_leads(conn, unassigned, active, history)
            print_summary(conn)
        finally:
            conn.close()

        log_event(
            event_type="LASSO_SYNC",
            entity_type="lasso_sync",
            actor="sync_from_lasso.py",
            latency_ms=timed_call_latency_ms(started_at),
            payload={
                "dealers_fetched": len(dealers_data),
                "dealers_inserted": dealers_inserted,
                "unassigned_fetched": len(unassigned),
                "active_fetched": len(active),
                "history_fetched": len(history),
                "leads_fetched": len(unassigned) + len(active) + len(history),
                "leads_inserted": lead_stats["inserted"],
                "lead_errors": lead_stats["errors"],
                "report_row_counts": {
                    "lead_report": len(lead_report or []),
                    "status_report": len(status_report or []),
                    "response_report": len(response_report or []),
                    "project_report": len(project_report or []),
                },
            },
        )
        print("\nSync complete!")
    except Exception as exc:
        log_event(
            event_type="LASSO_SYNC_FAILED",
            entity_type="lasso_sync",
            actor="sync_from_lasso.py",
            latency_ms=timed_call_latency_ms(started_at),
            payload={"error": str(exc)},
        )
        raise
