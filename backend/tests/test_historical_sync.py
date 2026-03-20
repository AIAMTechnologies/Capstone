import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend import main
from backend.dealer_ml_model import DealerMLModel


@pytest.fixture
def admin_user() -> main.AdminUser:
    return main.AdminUser(
        id=1,
        username="admin",
        email="admin@example.com",
        last_name="User",
        role="superadmin",
    )


@pytest.fixture
def memory_db(monkeypatch) -> Dict[str, Dict]:
    state = {
        "leads": {
            1: {
                "id": 1,
                "name": "Jane Doe",
                "email": "jane@example.com",
                "phone": "123",
                "address": "123 Main St",
                "city": "Calgary",
                "province": "AB",
                "postal_code": "T2A1B2",
                "job_type": "Residential",
                "status": "active",
                "assigned_dealer_id": 11,
                "recommended_dealer_id": 11,
                "final_dealer_selection": "Dealer A",
                "created_at": datetime(2024, 1, 1),
            }
        },
        "dealers": {
            11: {"id": 11, "name": "Dealer A", "city": "Calgary", "is_active": True},
            12: {"id": 12, "name": "Dealer B", "city": "Edmonton", "is_active": True},
        },
        "historical": {},
    }

    def fake_execute(query: str, params=None, fetch: bool = True):
        normalized = " ".join(query.split())

        if "FROM leads l" in normalized and "dealer_name_assigned" in normalized:
            lead_id = params[0]
            lead = state["leads"].get(lead_id)
            if not lead:
                return []

            row = {**lead}
            dealer = state["dealers"].get(lead.get("assigned_dealer_id"))
            if dealer:
                row["dealer_name_assigned"] = dealer.get("name")
            recommended = state["dealers"].get(lead.get("recommended_dealer_id"))
            if recommended:
                row["recommended_dealer_name"] = recommended.get("name")
            return [row]

        if normalized.startswith("UPDATE leads SET status"):
            status_value, lead_id = params
            state["leads"][lead_id]["status"] = status_value
            return True

        if normalized.startswith("UPDATE leads SET final_dealer_selection"):
            final_name, lead_id = params
            state["leads"][lead_id]["final_dealer_selection"] = final_name
            return True

        if normalized.startswith("SELECT 1 FROM historical_data WHERE id"):
            lead_id = params[0]
            return [1] if lead_id in state["historical"] else []

        if "INSERT INTO historical_data" in normalized:
            (
                lead_id,
                submit_date,
                first_name,
                address1,
                city,
                province,
                postal,
                dealer_name,
                project_type,
                current_status,
                final_dealer_selection,
            ) = params
            state["historical"][lead_id] = {
                "id": lead_id,
                "submit_date": submit_date,
                "first_name": first_name,
                "address1": address1,
                "city": city,
                "province": province,
                "postal": postal,
                "dealer_name": dealer_name,
                "project_type": project_type,
                "current_status": current_status,
                "final_dealer_selection": final_dealer_selection,
            }
            return True

        raise AssertionError(f"Unhandled query: {normalized}")

    monkeypatch.setattr(main, "execute_query", fake_execute)
    return state


def test_status_change_syncs_historical_data(memory_db, admin_user):
    asyncio.run(main.update_lead_status(1, "converted", current_user=admin_user))

    record = memory_db["historical"][1]
    assert record["current_status"] == "converted"
    assert record["final_dealer_selection"] == "Dealer A"


def test_non_standard_status_allows_historical_sync(memory_db, admin_user):
    asyncio.run(main.update_lead_status(1, "Converted Sale", current_user=admin_user))

    record = memory_db["historical"][1]
    assert record["current_status"] == "Converted Sale"


def test_subsequent_status_updates_refresh_historical(memory_db, admin_user):
    asyncio.run(main.update_lead_status(1, "converted", current_user=admin_user))
    asyncio.run(main.update_lead_status(1, "dead lead", current_user=admin_user))

    record = memory_db["historical"][1]
    assert record["current_status"] == "dead lead"


def test_training_uses_final_dealer_selection_label():
    rows = [
        {
            "final_dealer_selection": "Dealer A",
            "dealer_name": "Dealer A",
            "project_type": "Residential",
            "square_footage": 1200,
            "current_status": "converted",
        },
        {
            "final_dealer_selection": None,
            "dealer_name": "Dealer B",
            "project_type": "Commercial",
            "square_footage": 800,
            "current_status": "converted",
        },
        {
            "final_dealer_selection": "Dealer C",
            "dealer_name": "Dealer C",
            "project_type": "Residential",
            "square_footage": 900,
            "current_status": "converted",
        },
    ]

    def fake_query(_query: str, _params=None, _fetch: bool = True):
        return rows

    model = DealerMLModel(fake_query, min_training_rows=2)

    assert model.train(force=True)
    labels = set(model._pipeline.named_steps["model"].classes_)
    assert labels == {"Dealer A", "Dealer C"}
    assert model.status().get("training_rows") == 2


def test_backfill_historical_when_lead_already_non_active(memory_db, admin_user):
    memory_db["leads"][2] = {
        "id": 2,
        "name": "John Smith",
        "email": "john@example.com",
        "phone": "555",
        "address": "789 Oak",
        "city": "Calgary",
        "province": "AB",
        "postal_code": "T2A1B3",
        "job_type": "Commercial",
        "status": "converted",
        "assigned_dealer_id": 11,
        "recommended_dealer_id": 11,
        "final_dealer_selection": None,
        "created_at": datetime(2024, 2, 1),
    }

    asyncio.run(main.update_lead_status(2, "converted", current_user=admin_user))

    record = memory_db["historical"][2]
    assert record["current_status"] == "converted"
    assert record["final_dealer_selection"] == "Dealer A"
