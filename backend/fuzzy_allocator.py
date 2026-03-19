"""Fuzzy scoring helpers for installer allocation.

This module keeps the fuzzy logic implementation isolated from the FastAPI
application so it can be tested independently and re-used by background jobs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any, Dict, Optional, Tuple

_CACHE_TTL = timedelta(minutes=20)


@dataclass
class _HistoricalCache:
    payload: Dict[str, Dict[str, Any]]
    expires_at: datetime


_HISTORICAL_CACHE: Optional[_HistoricalCache] = None


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def normalize_installer_name(value: Optional[str]) -> Optional[str]:
    """Public helper used by other modules to normalize installer names."""

    return _normalize(value)


def _increment(bucket: Dict[str, int], raw_value: Optional[str]) -> None:
    key = _normalize(raw_value)
    if key is None:
        return
    bucket[key] = bucket.get(key, 0) + 1


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def fetch_installer_historical_feature_stats(
    query_executor,
) -> Dict[str, Dict[str, Any]]:
    """Return cached aggregates describing installer historical features."""

    global _HISTORICAL_CACHE

    now = datetime.utcnow()
    if _HISTORICAL_CACHE and _HISTORICAL_CACHE.expires_at > now:
        return deepcopy(_HISTORICAL_CACHE.payload)

    rows = query_executor(
        """
        SELECT dealer_name, project_type, product_type, square_footage, current_status
        FROM historical_data
        WHERE dealer_name IS NOT NULL
        """,
        None,
        True,
    )

    aggregates: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        normalized = _normalize(row.get("dealer_name"))
        if not normalized:
            continue

        entry = aggregates.setdefault(
            normalized,
            {
                "total": 0,
                "project_type": {},
                "product_type": {},
                "current_status": {},
                "square_total": 0.0,
                "square_count": 0,
            },
        )
        entry["total"] += 1
        _increment(entry["project_type"], row.get("project_type"))
        _increment(entry["product_type"], row.get("product_type"))
        _increment(entry["current_status"], row.get("current_status"))

        sqft = _safe_float(row.get("square_footage"))
        if sqft is not None:
            entry["square_total"] += sqft
            entry["square_count"] += 1

    for normalized, entry in aggregates.items():
        count = entry.get("square_count", 0)
        entry["avg_square_footage"] = (
            entry["square_total"] / count if count else None
        )
        entry.pop("square_total", None)
        entry.pop("square_count", None)

    _HISTORICAL_CACHE = _HistoricalCache(
        payload=deepcopy(aggregates),
        expires_at=now + _CACHE_TTL,
    )
    return deepcopy(aggregates)


def reset_historical_cache() -> None:
    """Clear the in-memory cache (useful for tests)."""

    global _HISTORICAL_CACHE
    _HISTORICAL_CACHE = None


def _membership_near(distance_km: float) -> float:
    if distance_km <= 10:
        return 1.0
    if distance_km >= 80:
        return 0.0
    return max(0.0, (80 - distance_km) / 70)


def _membership_medium(distance_km: float) -> float:
    if distance_km <= 40 or distance_km >= 160:
        return 0.0
    if distance_km <= 80:
        return (distance_km - 40) / 40
    if distance_km <= 120:
        return 1.0
    return (160 - distance_km) / 40


def _membership_far(distance_km: float) -> float:
    if distance_km <= 100:
        return 0.0
    if distance_km >= 220:
        return 1.0
    return (distance_km - 100) / 120


def _categorical_match_score(
    target_value: Optional[str],
    distribution: Optional[Dict[str, int]],
) -> float:
    if not target_value or not distribution:
        return 0.5

    total = sum(distribution.values())
    if total <= 0:
        return 0.5

    normalized = _normalize(target_value)
    if not normalized:
        return 0.5

    matches = distribution.get(normalized, 0)
    if matches == 0:
        return 0.25

    ratio = matches / total
    return min(1.0, 0.6 + 0.4 * ratio)


def _square_footage_match(
    requested_sqft: Optional[Any],
    avg_sqft: Optional[float],
) -> float:
    requested = _safe_float(requested_sqft)
    average = _safe_float(avg_sqft)

    if requested is None or average is None:
        return 0.6

    diff_ratio = abs(requested - average) / max(average, 1.0)
    if diff_ratio <= 0.1:
        return 1.0
    if diff_ratio >= 1.0:
        return 0.1
    return max(0.1, 1.0 - diff_ratio)


FUZZY_WEIGHTS = {
    "distance": 0.4,  # Highest importance
    "project_type": 0.08,
    "square_footage": 0.22,
    "product_type": 0.12,
    "current_status": 0.18,
}


def score_installer_with_fuzzy_logic(
    distance_km: float,
    installer_stats: Optional[Dict[str, Any]],
    lead_features: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Return (score, breakdown) for an installer."""

    installer_stats = installer_stats or {}

    distance_components = {
        "near": _membership_near(distance_km),
        "medium": _membership_medium(distance_km),
        "far": _membership_far(distance_km),
    }
    distance_score = (
        distance_components["near"] * 1.0
        + distance_components["medium"] * 0.6
        + distance_components["far"] * 0.2
    )

    square_score = _square_footage_match(
        lead_features.get("square_footage"), installer_stats.get("avg_square_footage")
    )
    project_score = _categorical_match_score(
        lead_features.get("project_type"), installer_stats.get("project_type")
    )
    product_score = _categorical_match_score(
        lead_features.get("product_type"), installer_stats.get("product_type")
    )
    status_score = _categorical_match_score(
        lead_features.get("current_status"), installer_stats.get("current_status")
    )

    composite = (
        distance_score * FUZZY_WEIGHTS["distance"]
        + square_score * FUZZY_WEIGHTS["square_footage"]
        + project_score * FUZZY_WEIGHTS["project_type"]
        + product_score * FUZZY_WEIGHTS["product_type"]
        + status_score * FUZZY_WEIGHTS["current_status"]
    )

    breakdown = {
        "distance": round(distance_score, 4),
        "square_footage": round(square_score, 4),
        "project_type": round(project_score, 4),
        "product_type": round(product_score, 4),
        "current_status": round(status_score, 4),
    }

    return composite, breakdown
