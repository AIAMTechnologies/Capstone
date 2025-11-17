import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.allocation_logic import (
    enforce_distance_guardrail,
    score_installers,
    serialize_recommendations_for_logging,
)


def _build_installer(installer_id, name, latitude, longitude, **extra):
    base = {
        "installer_id": installer_id,
        "installer_name": name,
        "city": "Test City",
        "province": "ON",
        "latitude": latitude,
        "longitude": longitude,
        "total_leads": 10,
        "active_leads": 1,
        "converted_leads": 5,
        "dead_leads": 2,
    }
    base.update(extra)
    return base


def test_closest_installer_is_ranked_first_when_all_else_equal():
    installers = [
        _build_installer(1, "Near", 0.0, 0.0),
        _build_installer(2, "Far", 1.0, 0.0),
    ]
    probs = {"Near": 0.5, "Far": 0.5}

    scored = score_installers(
        installers,
        lead_lat=0.0,
        lead_lon=0.0,
        distance_fn=lambda a, b, c, d: math.dist((a, b), (c, d)) * 111,
        ml_probabilities=probs,
        max_distance_km=200,
        fallback_distance_km=600,
    )

    assert scored[0]["installer_id"] == 1
    assert scored[0]["distance_km"] < scored[1]["distance_km"]


def test_guardrail_blocks_far_option_without_probability_edge():
    installers = [
        _build_installer(1, "Closest", 0.0, 0.0),
        _build_installer(2, "Farther", 0.6, 0.0),
    ]
    probs = {"Closest": 0.45, "Farther": 0.55}

    scored = score_installers(
        installers,
        lead_lat=0.0,
        lead_lon=0.0,
        distance_fn=lambda a, b, c, d: math.dist((a, b), (c, d)) * 111,
        ml_probabilities=probs,
        max_distance_km=200,
        fallback_distance_km=600,
    )
    ranked = enforce_distance_guardrail(scored, guardrail_km=40, probability_advantage=0.15)

    assert ranked[0]["installer_id"] == 1
    assert ranked[0].get("guardrail_blocked") is None


def test_guardrail_allows_far_option_with_large_probability_edge():
    installers = [
        _build_installer(1, "Closest", 0.0, 0.0, converted_leads=1, total_leads=2, active_leads=4, dead_leads=1),
        _build_installer(2, "Farther", 0.7, 0.0, converted_leads=20, total_leads=25, active_leads=1, dead_leads=2),
    ]
    probs = {"Closest": 0.3, "Farther": 0.65}

    scored = score_installers(
        installers,
        lead_lat=0.0,
        lead_lon=0.0,
        distance_fn=lambda a, b, c, d: math.dist((a, b), (c, d)) * 111,
        ml_probabilities=probs,
        max_distance_km=200,
        fallback_distance_km=600,
    )
    ranked = enforce_distance_guardrail(scored, guardrail_km=40, probability_advantage=0.15)

    assert ranked[0]["installer_id"] == 2
    assert ranked[0].get("guardrail_override_applied") is True


def test_serialization_preserves_key_fields():
    installers = [
        {
            "installer_id": 5,
            "installer_name": "SnapShot",
            "distance_km": 25.0,
            "allocation_score": 0.8,
            "ml_probability": 0.6,
            "success_rate": 0.7,
            "quality_score": 0.65,
            "rank": 1,
            "is_within_max_distance": True,
            "is_fallback_option": False,
            "score_breakdown": {"distance_score": 0.9},
            "key_factors": ["distance:25km"],
        }
    ]

    serialized = serialize_recommendations_for_logging(installers)
    assert serialized[0]["installer_id"] == 5
    assert serialized[0]["rank"] == 1
    assert serialized[0]["key_factors"] == ["distance:25km"]
