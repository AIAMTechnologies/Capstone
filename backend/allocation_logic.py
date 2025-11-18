"""Reusable helpers for installer ranking and recommendation tracing."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

FUZZY_IMPORTANCE_WEIGHTS = {
    # Distance must dominate, with the remaining weights mirroring the requested priority order.
    "distance": 0.4,
    "square_footage": 0.2,
    "current_status": 0.2,
    "product_type": 0.15,
    "project_type": 0.05,
}

# After validating the initial fuzzy-only deployment with real leads, we saw situations where
# attribute matches could still outweigh distance. To prevent far-away installers from beating
# obviously closer options, we blend the final score so that distance remains the single most
# important factor.
DISTANCE_DOMINANCE_WEIGHT = 0.7

_ATTRIBUTE_WEIGHT_SUM = (
    FUZZY_IMPORTANCE_WEIGHTS["square_footage"]
    + FUZZY_IMPORTANCE_WEIGHTS["current_status"]
    + FUZZY_IMPORTANCE_WEIGHTS["product_type"]
    + FUZZY_IMPORTANCE_WEIGHTS["project_type"]
)

SQUARE_FOOTAGE_MAX_DELTA = 4000.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _distance_component(
    distance_km: float,
    *,
    max_distance_km: float,
    fallback_distance_km: float,
    has_local_installers: bool,
) -> Dict[str, Any]:
    """Return distance-derived metrics used during scoring."""

    within_max = distance_km <= max_distance_km
    fallback_limit = max(fallback_distance_km, max_distance_km)

    if within_max:
        normalized = max(0.0, 1 - (distance_km / max(max_distance_km, 1)))
    else:
        normalized = max(0.0, 1 - (distance_km / max(fallback_limit, 1)))
        if has_local_installers:
            normalized *= 0.3  # punish far candidates when local options exist
        else:
            normalized = max(normalized, 0.05)

    return {
        "distance_score": round(normalized, 4),
        "is_within_max_distance": within_max,
        "is_fallback_option": not within_max,
        "distance_review_required": not within_max,
    }


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _square_footage_score(
    lead_square_footage: Optional[float],
    average_square_footage: Optional[float],
) -> float:
    if lead_square_footage is None and average_square_footage is None:
        return 0.5
    if lead_square_footage is None:
        return 0.55  # installer has context but the lead does not
    if average_square_footage is None:
        return 0.6

    delta = abs(float(lead_square_footage) - float(average_square_footage))
    if delta <= 150:
        return 1.0
    if delta >= SQUARE_FOOTAGE_MAX_DELTA:
        return 0.0
    return _clamp(1 - ((delta - 150) / (SQUARE_FOOTAGE_MAX_DELTA - 150)))


def _ratio_score(matches: Optional[float], total: Optional[float]) -> float:
    if matches is None or total is None or total <= 0:
        return 0.5
    return _clamp(float(matches) / float(total))


def _build_fuzzy_inputs(
    *,
    lead_features: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, float]:
    total_jobs = stats.get("total_jobs") or 0

    def _feature_ratio(stat_key: str, feature_key: str) -> float:
        if not lead_features.get(feature_key):
            return 0.5
        return _ratio_score(stats.get(stat_key), total_jobs)

    project_score = _feature_ratio("project_matches", "project_type")
    product_score = _feature_ratio("product_matches", "product_type")
    status_score = _feature_ratio("status_matches", "current_status")
    square_score = _square_footage_score(
        lead_features.get("square_footage"),
        stats.get("avg_square_footage"),
    )

    return {
        "project_type": project_score,
        "product_type": product_score,
        "current_status": status_score,
        "square_footage": square_score,
    }


def _derive_membership_triple(value: float) -> Dict[str, float]:
    balanced = _clamp(1 - abs(value - 0.6) * 2)
    return {
        "strong": value,
        "balanced": balanced,
        "weak": _clamp(1 - value),
    }


def _fuzzy_score(distance_score: float, fuzzy_inputs: Dict[str, float]) -> Dict[str, Any]:
    attribute_strength = (
        FUZZY_IMPORTANCE_WEIGHTS["square_footage"] * fuzzy_inputs["square_footage"]
        + FUZZY_IMPORTANCE_WEIGHTS["current_status"] * fuzzy_inputs["current_status"]
        + FUZZY_IMPORTANCE_WEIGHTS["product_type"] * fuzzy_inputs["product_type"]
        + FUZZY_IMPORTANCE_WEIGHTS["project_type"] * fuzzy_inputs["project_type"]
    ) / max(_ATTRIBUTE_WEIGHT_SUM, 1e-6)

    distance_membership = {
        "close": distance_score,
        "moderate": _clamp(1 - abs(distance_score - 0.5) * 2),
        "far": _clamp(1 - distance_score),
    }
    attribute_membership = _derive_membership_triple(attribute_strength)

    rule_high = min(distance_membership["close"], attribute_membership["strong"])
    rule_medium = max(
        min(distance_membership["moderate"], attribute_membership["strong"]),
        min(distance_membership["close"], attribute_membership["balanced"]),
    )
    rule_low = max(
        min(distance_membership["far"], 1.0),
        min(attribute_membership["weak"], 1.0),
    )

    numerator = (rule_high * 0.95) + (rule_medium * 0.6) + (rule_low * 0.25)
    denominator = rule_high + rule_medium + rule_low
    if denominator <= 0:
        blended = (
            FUZZY_IMPORTANCE_WEIGHTS["distance"] * distance_score
            + (1 - FUZZY_IMPORTANCE_WEIGHTS["distance"]) * attribute_strength
        )
    else:
        blended = numerator / denominator

    return {
        "score": _clamp(blended),
        "attribute_strength": _clamp(attribute_strength),
        "memberships": {
            "distance": distance_membership,
            "attributes": attribute_membership,
            "rules": {
                "high": rule_high,
                "medium": rule_medium,
                "low": rule_low,
            },
        },
        "inputs": fuzzy_inputs,
    }


def score_installers(
    installers: Sequence[Dict[str, Any]],
    *,
    lead_lat: float,
    lead_lon: float,
    distance_fn: Callable[[float, float, float, float], float],
    ml_probabilities: Dict[str, float],
    max_distance_km: float,
    fallback_distance_km: float,
    lead_features: Optional[Dict[str, Any]] = None,
    historical_feature_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return scored installers ordered by fuzzy composite rank."""

    enriched: List[Dict[str, Any]] = []
    for installer in installers:
        lat = installer.get("latitude")
        lon = installer.get("longitude")
        if lat is None or lon is None:
            continue
        distance_km = distance_fn(lead_lat, lead_lon, lat, lon)
        if distance_km > fallback_distance_km:
            continue
        item = dict(installer)
        item["distance_km"] = round(distance_km, 2)
        enriched.append(item)

    if not enriched:
        return []

    has_local = any(inst["distance_km"] <= max_distance_km for inst in enriched)
    closest_distance = min(inst["distance_km"] for inst in enriched)
    total_max = max((inst.get("total_leads") or 0) for inst in enriched) or 0
    active_max = max((inst.get("active_leads") or 0) for inst in enriched) or 0

    lead_features = lead_features or {}
    stats_lookup = {k: v for k, v in (historical_feature_stats or {}).items()}
    scored: List[Dict[str, Any]] = []
    for inst in enriched:
        installer_name = inst.get("name") or inst.get("installer_name") or ""
        probability = ml_probabilities.get(installer_name) or ml_probabilities.get(installer_name.lower()) or 0.0
        converted = inst.get("converted_leads") or 0
        dead = inst.get("dead_leads") or 0
        total = inst.get("total_leads") or (converted + dead)
        active = inst.get("active_leads") or 0

        success_rate = _safe_ratio(converted, converted + dead)
        workload_ratio = _safe_ratio(active, active_max) if active_max else 0.0

        distance_bits = _distance_component(
            inst["distance_km"],
            max_distance_km=max_distance_km,
            fallback_distance_km=fallback_distance_km,
            has_local_installers=has_local,
        )

        installer_key = installer_name.strip().lower()
        stats = stats_lookup.get(installer_key, {})
        fuzzy_inputs = _build_fuzzy_inputs(
            lead_features=lead_features,
            stats=stats,
        )
        fuzzy_result = _fuzzy_score(distance_bits["distance_score"], fuzzy_inputs)
        distance_score = distance_bits["distance_score"]
        quality_score = round(fuzzy_result["attribute_strength"], 4)
        composite_score = (
            (DISTANCE_DOMINANCE_WEIGHT * distance_score)
            + ((1 - DISTANCE_DOMINANCE_WEIGHT) * fuzzy_result["score"])
        )
        allocation_score = composite_score - (workload_ratio * 0.05)
        if not distance_bits["is_within_max_distance"] and has_local:
            allocation_score *= 0.5

        scored.append(
            {
                **inst,
                **distance_bits,
                "closest_distance_km": closest_distance,
                "success_rate": round(success_rate, 4),
                "quality_score": quality_score,
                "ml_probability": round(float(probability), 4),
                "workload_ratio": round(workload_ratio, 4),
                "allocation_score": round(allocation_score, 4),
                "score_breakdown": {
                    "distance_score": distance_score,
                    "quality_score": quality_score,
                    "composite_score": round(composite_score, 4),
                    "workload_penalty": round(workload_ratio * 0.05, 4),
                    "fuzzy": fuzzy_result,
                },
                "key_factors": [
                    f"distance:{inst['distance_km']}km",
                    f"sqft_fit:{round(fuzzy_inputs['square_footage'], 2)}",
                    f"status_match:{round(fuzzy_inputs['current_status'], 2)}",
                    f"product_match:{round(fuzzy_inputs['product_type'], 2)}",
                ],
            }
        )

    scored.sort(key=lambda inst: (-inst["allocation_score"], inst["distance_km"]))
    return scored


def enforce_distance_guardrail(
    scored: List[Dict[str, Any]],
    *,
    guardrail_km: float,
    probability_advantage: float,
) -> List[Dict[str, Any]]:
    """Apply the hard guardrail policy to already-scored installers."""

    if not scored:
        return scored

    closest_distance = min(inst["distance_km"] for inst in scored)
    allowed_distance = closest_distance + guardrail_km
    top_candidate = scored[0]
    if top_candidate["distance_km"] <= allowed_distance:
        return scored

    preferred_locals = [inst for inst in scored if inst["distance_km"] <= allowed_distance]
    if not preferred_locals:
        return scored

    best_local = max(preferred_locals, key=lambda inst: inst["allocation_score"])
    if top_candidate.get("ml_probability", 0.0) >= best_local.get("ml_probability", 0.0) + probability_advantage:
        top_candidate["guardrail_override_applied"] = True
        return scored

    scored.remove(best_local)
    scored.insert(0, best_local)
    top_candidate["guardrail_blocked"] = True
    return scored


def serialize_recommendations_for_logging(
    scored: Sequence[Dict[str, Any]],
    *,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Return a JSON-friendly subset of recommendation data."""

    serialized: List[Dict[str, Any]] = []
    items = list(scored)
    if limit is not None:
        items = items[:limit]

    for inst in items:
        serialized.append(
            {
                "rank": inst.get("rank"),
                "installer_id": inst.get("installer_id") or inst.get("id"),
                "installer_name": inst.get("installer_name") or inst.get("name"),
                "distance_km": inst.get("distance_km"),
                "allocation_score": inst.get("allocation_score"),
                "ml_probability": inst.get("ml_probability"),
                "success_rate": inst.get("success_rate"),
                "quality_score": inst.get("quality_score"),
                "is_within_max_distance": inst.get("is_within_max_distance"),
                "is_fallback_option": inst.get("is_fallback_option"),
                "distance_review_required": inst.get("distance_review_required"),
                "score_breakdown": inst.get("score_breakdown"),
                "key_factors": inst.get("key_factors"),
            }
        )
    return serialized
