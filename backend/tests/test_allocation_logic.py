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


def _build_stats(avg_sqft, project_matches, product_matches, status_matches, total_jobs=20):
    return {
        "avg_square_footage": avg_sqft,
        "project_matches": project_matches,
        "product_matches": product_matches,
        "status_matches": status_matches,
        "total_jobs": total_jobs,
    }


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
        lead_features={},
        historical_feature_stats={},
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
        lead_features={},
        historical_feature_stats={},
    )
    ranked = enforce_distance_guardrail(scored, guardrail_km=40, probability_advantage=0.15)

    assert ranked[0]["installer_id"] == 1
    assert ranked[0].get("guardrail_blocked") is None


def test_guardrail_allows_far_option_with_large_probability_edge():
    installers = [
        _build_installer(1, "CloserButMismatched", 2.25, 0.0, converted_leads=1, total_leads=2, active_leads=4, dead_leads=1),
        _build_installer(2, "SpecialistFar", 2.97, 0.0, converted_leads=20, total_leads=25, active_leads=1, dead_leads=2),
    ]
    probs = {"CloserButMismatched": 0.05, "SpecialistFar": 0.95}
    lead_features = {
        "project_type": "Commercial",
        "product_type": "Tint",
        "square_footage": 5000.0,
        "current_status": "won",
    }
    stats_lookup = {
        "closerbutmismatched": _build_stats(0.0, 0, 0, 0, total_jobs=25),
        "specialistfar": _build_stats(5000.0, 23, 23, 24, total_jobs=25),
    }

    scored = score_installers(
        installers,
        lead_lat=0.0,
        lead_lon=0.0,
        distance_fn=lambda a, b, c, d: math.dist((a, b), (c, d)) * 111,
        ml_probabilities=probs,
        max_distance_km=200,
        fallback_distance_km=600,
        lead_features=lead_features,
        historical_feature_stats=stats_lookup,
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


def test_fuzzy_logic_accounts_for_feature_matches():
    installers = [
        _build_installer(1, "FuzzyMatch", 0.0, 0.0),
        _build_installer(2, "Generic", 0.0, 0.0),
    ]
    probs = {"FuzzyMatch": 0.5, "Generic": 0.5}
    lead_features = {
        "project_type": "Residential",
        "product_type": "Film",
        "square_footage": 1200.0,
        "current_status": "won",
    }
    stats_lookup = {
        "fuzzymatch": _build_stats(1150.0, 15, 15, 16),
        "generic": _build_stats(3000.0, 1, 1, 2),
    }

    scored = score_installers(
        installers,
        lead_lat=0.0,
        lead_lon=0.0,
        distance_fn=lambda a, b, c, d: 0.0,
        ml_probabilities=probs,
        max_distance_km=200,
        fallback_distance_km=600,
        lead_features=lead_features,
        historical_feature_stats=stats_lookup,
    )

    assert scored[0]["installer_id"] == 1
    fuzzy_inputs_best = scored[0]["score_breakdown"]["fuzzy"]["inputs"]
    fuzzy_inputs_other = scored[1]["score_breakdown"]["fuzzy"]["inputs"]
    assert fuzzy_inputs_best["square_footage"] > fuzzy_inputs_other["square_footage"]
