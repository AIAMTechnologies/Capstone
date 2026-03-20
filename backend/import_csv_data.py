"""
Import CSV data into the LAP Portal database.
- Upserts 48 dealers from dealer_summary_export
- Imports ~27K leads from lead_export, deduplicating by (first_name, last_name, email, submit_date)
"""

import csv
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

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
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


def parse_float(val):
    if not val or val.strip() == "":
        return None
    try:
        return float(val.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def clean(val):
    if val is None:
        return None
    val = str(val).strip()
    return val if val else None


def parse_bool(val):
    if not val:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "t")


def import_dealers(conn, filepath):
    print(f"\n{'='*60}")
    print("IMPORTING DEALERS")
    print(f"{'='*60}")

    cur = conn.cursor()

    # Delete seed dealers that don't have leads referencing them
    cur.execute("DELETE FROM dealers WHERE id NOT IN (SELECT DISTINCT assigned_dealer_id FROM leads WHERE assigned_dealer_id IS NOT NULL)")
    conn.commit()

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            original_id = row.get("id", "").strip()
            name = clean(row.get("Dealer Name"))
            if not name or not original_id:
                continue
            try:
                original_id = int(original_id)
            except ValueError:
                continue

            cur.execute("""
                INSERT INTO dealers (id, name, city, province, email, notification_email, is_active)
                VALUES (%s, %s, '', '', '', '', TRUE)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, is_active = TRUE
            """, (original_id, name.strip()))
            count += 1

    conn.commit()

    # Reset sequence past max id
    cur.execute("SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM dealers")
    next_id = cur.fetchone()["next_id"]
    cur.execute(f"ALTER SEQUENCE dealers_id_seq RESTART WITH {next_id}")
    conn.commit()

    cur.execute("SELECT COUNT(*) as cnt FROM dealers")
    print(f"Total dealers: {cur.fetchone()['cnt']}")
    return count


def import_leads(conn, filepath):
    print(f"\n{'='*60}")
    print("IMPORTING LEADS")
    print(f"{'='*60}")

    cur = conn.cursor()

    # Clear ALL existing leads — user wants only CSV data
    cur.execute("DELETE FROM lead_logs")
    cur.execute("DELETE FROM lead_assignments")
    cur.execute("DELETE FROM leads")
    conn.commit()
    print("Cleared all existing leads, logs, and assignments")

    # Build dealer name -> id map
    cur.execute("SELECT id, LOWER(TRIM(name)) as lname FROM dealers")
    dealer_map = {r["lname"]: r["id"] for r in cur.fetchall()}

    # Dedup set for within-file duplicates
    existing = set()

    inserted = 0
    skipped = 0
    errors = 0

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            first_name = clean(row.get("first_name")) or ""
            last_name = clean(row.get("last_name")) or ""
            email = clean(row.get("email"))
            submit_date = parse_datetime(row.get("submit_date"))

            # Dedup: use first_name + last_name + email + submit_date
            fp = (
                first_name.strip().lower(),
                last_name.strip().lower(),
                (email or "").strip().lower(),
                str(submit_date or "")[:19]
            )
            if fp in existing:
                skipped += 1
                continue
            existing.add(fp)

            # Map status
            raw_status = (clean(row.get("current_status")) or "").lower()
            status = STATUS_MAP.get(raw_status, "active")

            # Map dealer
            dealer_name = clean(row.get("dealer_name"))
            dealer_id = None
            if dealer_name:
                dealer_id = dealer_map.get(dealer_name.strip().lower())

            full_name = f"{first_name} {last_name}".strip() or email or "Unknown"

            form_submit_date = parse_datetime(row.get("form_submit_date")) or submit_date
            phone = clean(row.get("primary_phone"))
            company_name = clean(row.get("company_name"))
            job_title = clean(row.get("job_title"))
            address = clean(row.get("address1"))
            address2 = clean(row.get("address2"))
            if address and address2:
                address = f"{address}, {address2}"
            elif address2:
                address = address2
            city = clean(row.get("city"))
            province = clean(row.get("province"))
            postal_code = clean(row.get("postal"))
            project_type = clean(row.get("project_type"))
            product_type = clean(row.get("product_type"))
            product_type_2 = clean(row.get("product_type2"))
            product_type_3 = clean(row.get("product_type3"))
            comments = clean(row.get("comments"))
            business_category = clean(row.get("business_category"))
            sq_ft_raw = clean(row.get("square_footage"))
            opt_in = parse_bool(row.get("opt_in"))
            lead_source = clean(row.get("source"))
            dealer_email_val = clean(row.get("dealer_email"))
            custom_pick_1 = clean(row.get("custom_pick1")) or sq_ft_raw
            custom_pick_3 = clean(row.get("custom_pick3"))
            landing_page = clean(row.get("landing_page_name"))
            landing_page_url = clean(row.get("landing_url"))
            landing_page_variant = clean(row.get("landing_variant"))
            project_city = clean(row.get("project_city"))
            utm_source = clean(row.get("utm_source"))
            utm_medium = clean(row.get("utm_medium"))
            utm_campaign = clean(row.get("utm_campaign"))
            utm_content = clean(row.get("utm_content"))
            value_of_order = parse_float(row.get("value_of_order"))

            try:
                cur.execute("""
                    INSERT INTO leads (
                        name, first_name, last_name, email, phone, address,
                        city, province, postal_code, job_title, company_name,
                        product_type, product_type_2, product_type_3,
                        custom_pick_1, project_city, project_type,
                        business_category, dealer_email, lead_source, source, opt_in,
                        landing_page, landing_page_url, landing_page_variant,
                        utm_source, utm_medium, utm_campaign, utm_content,
                        custom_pick_3, comments, status,
                        form_submit_date, created_at,
                        assigned_dealer_id, value_of_order,
                        final_dealer_selection
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s
                    )
                """, (
                    full_name, first_name, last_name, email, phone, address,
                    city, province, postal_code, job_title, company_name,
                    product_type, product_type_2, product_type_3,
                    custom_pick_1, project_city, project_type,
                    business_category, dealer_email_val, lead_source, lead_source, opt_in,
                    landing_page, landing_page_url, landing_page_variant,
                    utm_source, utm_medium, utm_campaign, utm_content,
                    custom_pick_3, comments, status,
                    form_submit_date, submit_date,
                    dealer_id, value_of_order,
                    dealer_name,
                ))
                conn.commit()
                inserted += 1

                if inserted % 2000 == 0:
                    print(f"  Progress: {inserted} inserted, {skipped} skipped")

            except Exception as e:
                conn.rollback()
                errors += 1
                if errors <= 3:
                    print(f"  Error row {row_num}: {str(e)[:150]}")

    # Fix sequence
    cur.execute("SELECT setval('leads_id_seq', COALESCE((SELECT MAX(id) FROM leads), 1))")
    conn.commit()

    print(f"\nLead import done: {inserted} inserted, {skipped} skipped, {errors} errors")
    return inserted


if __name__ == "__main__":
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

    import_dealers(conn, "/Users/ammaralam/Downloads/dealer_summary_export_202603181140.csv")
    import_leads(conn, "/Users/ammaralam/Downloads/lead_export_202603181142.csv")

    # Summary
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM leads")
    print(f"\nFinal: {cur.fetchone()['cnt']} leads")
    cur.execute("SELECT status, COUNT(*) as cnt FROM leads GROUP BY status ORDER BY cnt DESC")
    for r in cur.fetchall():
        print(f"  {r['status']}: {r['cnt']}")
    cur.execute("SELECT COUNT(*) as cnt FROM leads WHERE assigned_dealer_id IS NOT NULL")
    print(f"Leads with dealer_id: {cur.fetchone()['cnt']}")
    cur.execute("SELECT COALESCE(SUM(value_of_order),0) as v FROM leads")
    print(f"Total value: ${cur.fetchone()['v']:,.2f}")

    conn.close()
    print("Done!")
