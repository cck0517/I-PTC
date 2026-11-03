"""Shared helpers for ptc_mt_yelp_* verifiers.

Each function takes the YelpAPI sub-config (initial_config["YelpAPI"]) unless
otherwise stated.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


RECENT_CUTOFF = "2026-01-27"  # "last 90 days" anchored at 2026-04-26


def businesses(cfg: dict) -> dict[str, dict]:
    return cfg.get("businesses", {})


def reviews(cfg: dict) -> dict[str, dict]:
    return cfg.get("reviews", {})


def reviews_by_business(cfg: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in reviews(cfg).values():
        out[r["business_id"]].append(r)
    return dict(out)


def reviews_by_user(cfg: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in reviews(cfg).values():
        out[r["user_id"]].append(r)
    return dict(out)


def filter_recent(reviews_list: Iterable[dict], cutoff: str = RECENT_CUTOFF) -> list[dict]:
    return [r for r in reviews_list if r["created_at"] >= cutoff]


def has_categories(biz: dict, cats: Iterable[str]) -> bool:
    biz_cats = set(biz.get("categories", []))
    return any(c in biz_cats for c in cats)


def sample_stddev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def round2(x: float) -> float:
    return round(float(x), 2)


def round4(x: float) -> float:
    return round(float(x), 4)


def approx(a, b, tol: float = 0.005) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


# Five top-level food categories used across the SF dataset.
SF_FOOD_CATS = ("Italian", "Sushi", "Mexican", "Cafe", "Bakery")
