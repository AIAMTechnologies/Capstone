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
from backend.ml_model import InstallerMLModel


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
                "assigned_installer_id": 11,
                "installer_override_id": None,
                "final_installer_selection": "Installer A",
                "created_at": datetime(2024, 1, 1),
            }
        },
        "installers": {
            11: {"id": 11, "name": "Installer A", "city": "Calgary", "is_active": True},
            12: {"id": 12, "name": "Installer B", "city": "Edmonton", "is_active": True},
        },
        "historical": {},
    }

    def fake_execute(query: str, params=None, fetch: bool = True):
        normalized = " ".join(query.split())

        if "FROM leads l" in normalized and "assigned_installer_name" in normalized:
            lead_id = params[0]
            lead = state["leads"].get(lead_id)
            if not lead:
                return []
            installer = state["installers"].get(lead.get("assigned_installer_id"))
            row = {**lead}
            if installer:
                row["assigned_installer_name"] = installer.get("name")
                row["assigned_installer_city"] = installer.get("city")
            return [row]

        if normalized.startswith("UPDATE leads SET status"):
            status_value, lead_id = params
            state["leads"][lead_id]["status"] = status_value
            return True

        if normalized.startswith("UPDATE leads SET final_installer_selection"):
            final_name, lead_id = params
            state["leads"][lead_id]["final_installer_selection"] = final_name
            return True

        if "UPDATE leads SET installer_override_id" in normalized:
            override_id, assigned_id, final_selection, lead_id = params
            lead = state["leads"][lead_id]
            lead["installer_override_id"] = override_id
            lead["assigned_installer_id"] = assigned_id
            lead["final_installer_selection"] = final_selection
            return True

        if normalized.startswith("SELECT id, name, city FROM installers"):
            installer_id = params[0]
            installer = state["installers"].get(installer_id)
            if installer and installer.get("is_active"):
                return [installer]
            return []

        if normalized.startswith("SELECT name, city FROM installers WHERE id"):
            installer_id = params[0]
            installer = state["installers"].get(installer_id)
            return [installer] if installer else []

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
                final_installer_selection,
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
                "final_installer_selection": final_installer_selection,
            }
            return True

        raise AssertionError(f"Unhandled query: {normalized}")

    monkeypatch.setattr(main, "execute_query", fake_execute)
    return state


def test_status_change_syncs_historical_data(memory_db, admin_user):
    asyncio.run(main.update_lead_status(1, "converted", current_user=admin_user))

    record = memory_db["historical"][1]
    assert record["current_status"] == "converted"
    assert record["final_installer_selection"] == "Installer A"


def test_non_standard_status_allows_historical_sync(memory_db, admin_user):
    # Frontend may send user-friendly labels; these should still flow through and
    # create a historical record when the lead leaves the active pipeline.
    asyncio.run(main.update_lead_status(1, "Converted Sale", current_user=admin_user))

    record = memory_db["historical"][1]
    assert record["current_status"] == "Converted Sale"


def test_override_updates_final_installer(memory_db, admin_user):
    asyncio.run(main.update_installer_override(1, installer_id=12, current_user=admin_user))
    asyncio.run(main.update_lead_status(1, "converted", current_user=admin_user))

    lead = memory_db["leads"][1]
    assert lead["final_installer_selection"] == "Installer B"

    record = memory_db["historical"][1]
    assert record["final_installer_selection"] == "Installer B"


def test_training_uses_final_installer_selection_label():
    rows = [
        {
            "final_installer_selection": "Installer A",
            "dealer_name": "Dealer A",
            "project_type": "Residential",
            "square_footage": 1200,
            "current_status": "converted",
        },
        {
            "final_installer_selection": None,
            "dealer_name": "Dealer B",
            "project_type": "Commercial",
            "square_footage": 800,
            "current_status": "converted",
        },
        {
            "final_installer_selection": "Installer C",
            "dealer_name": "Dealer C",
            "project_type": "Residential",
            "square_footage": 900,
            "current_status": "converted",
        },
    ]

    def fake_query(_query: str, _params=None, _fetch: bool = True):
        return rows

    model = InstallerMLModel(fake_query, min_training_rows=2)

    assert model.train(force=True)
    labels = set(model._pipeline.named_steps["model"].classes_)
    assert labels == {"Installer A", "Installer C"}
    assert model.status().get("training_rows") == 2
