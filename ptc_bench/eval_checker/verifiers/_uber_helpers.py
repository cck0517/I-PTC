"""Shared helpers for ptc_mt_multi_* verifiers (Uber + GoogleMaps + Yelp).

The cost formula matches UberAPI.get_ride_estimate exactly:
    dist = haversine(...)
    duration_min = dist / average_speed_mph * 60
    raw = base_fare + dist*per_mile_rate + duration_min*per_minute_rate
    cost = round(max(minimum_fare, raw) * surge_multiplier, 2)
where dist is rounded to 2 decimals (per UberAPI._haversine).
"""
from __future__ import annotations

import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return round(2 * R * math.asin(math.sqrt(a)), 2)


def uber_cost(origin: tuple[float, float], dest: tuple[float, float],
              pricing: dict) -> float:
    dist = haversine(origin[0], origin[1], dest[0], dest[1])
    speed = pricing.get("average_speed_mph", 25)
    duration = dist / speed * 60
    base = pricing.get("base_fare", 2.5)
    per_mile = pricing.get("per_mile_rate", 1.75)
    per_min = pricing.get("per_minute_rate", 0.35)
    minimum = pricing.get("minimum_fare", 8.0)
    surge = pricing.get("surge_multiplier", 1.0)
    raw = base + dist * per_mile + duration * per_min
    return round(max(minimum, raw) * surge, 2)


def geocode(maps_cfg: dict, address: str) -> tuple[float, float] | None:
    cache = maps_cfg.get("geocode_cache", {})
    entry = cache.get(address)
    if entry is None:
        return None
    return float(entry["latitude"]), float(entry["longitude"])


def find_business(yelp_cfg: dict, name: str) -> dict | None:
    """Return the business dict whose name matches exactly, or None."""
    for b in yelp_cfg.get("businesses", {}).values():
        if b.get("name") == name:
            return b
    return None
