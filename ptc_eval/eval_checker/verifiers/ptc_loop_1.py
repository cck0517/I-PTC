"""Verifier for ptc_loop_1.

Search Yelp for coffee shops and bookmark those with rating >= 4.5.
Dynamically computes expected bookmarks from initial_config.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Compute expected bookmarks from initial_config
    businesses = initial_config.get("YelpAPI", {}).get("businesses", {})
    should_bookmark = {bid for bid, b in businesses.items() if b.get("rating", 0) >= 4.5}
    should_not = {bid for bid in businesses if bid not in should_bookmark}

    yelp = (final_state or {}).get("YelpAPI", {})

    # Model may use save_business() or bookmark_business(). Accept either.
    saved_places = yelp.get("saved_places", {})
    sp_ids = set(saved_places.keys()) if isinstance(saved_places, dict) else set(saved_places)
    bm_ids = set(yelp.get("bookmarks", []))
    saved_ids = sp_ids | bm_ids

    # Check high-rated businesses are saved
    for biz_id in sorted(should_bookmark):
        checks[f"bookmarked_{biz_id}"] = biz_id in saved_ids

    # Check low-rated businesses are NOT saved
    for biz_id in sorted(should_not):
        checks[f"not_bookmarked_{biz_id}"] = biz_id not in saved_ids

    # Count check
    checks[f"exactly_{len(should_bookmark)}_bookmarks"] = len(saved_ids) == len(should_bookmark)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "final_saved_places": sorted(saved_ids),
            "expected_bookmarks": sorted(should_bookmark),
        },
    )
