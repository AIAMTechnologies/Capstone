"""Reusable helpers for installer ranking and recommendation tracing."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence


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


def score_installers(
    installers: Sequence[Dict[str, Any]],
    *,
    lead_lat: float,
    lead_lon: float,
    distance_fn: Callable[[float, float, float, float], float],
    ml_probabilities: Dict[str, float],
    max_distance_km: float,
    fallback_distance_km: float,
) -> List[Dict[str, Any]]:
    """Return scored installers ordered by composite rank."""

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

    scored: List[Dict[str, Any]] = []
    for inst in enriched:
        installer_name = inst.get("name") or inst.get("installer_name") or ""
        probability = ml_probabilities.get(installer_name) or ml_probabilities.get(installer_name.lower()) or 0.0
        converted = inst.get("converted_leads") or 0
        dead = inst.get("dead_leads") or 0
        total = inst.get("total_leads") or (converted + dead)
        active = inst.get("active_leads") or 0

        success_rate = _safe_ratio(converted, converted + dead)
        experience_score = _safe_ratio(total, total_max) if total_max else 0.0
        workload_ratio = _safe_ratio(active, active_max) if active_max else 0.0

        distance_bits = _distance_component(
            inst["distance_km"],
            max_distance_km=max_distance_km,
            fallback_distance_km=fallback_distance_km,
            has_local_installers=has_local,
        )

        quality_score = round(
            (0.55 * probability) + (0.3 * success_rate) + (0.15 * experience_score),
            4,
        )
        distance_score = distance_bits["distance_score"]
        allocation_score = (
            (distance_score * 0.5)
            + (quality_score * 0.45)
            - (workload_ratio * 0.1)
        )
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
                    "workload_penalty": round(workload_ratio * 0.1, 4),
                },
                "key_factors": [
                    f"distance:{inst['distance_km']}km",
                    f"success_rate:{round(success_rate, 2)}",
                    f"ml_prob:{round(float(probability), 2)}",
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
