"""Verifier for ptc_loop_4.

Rate all completed Uber rides 5 stars. Only rides with status "completed"
should be rated. Dynamically discovers completed ride IDs from initial_config.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks

EXPECTED_RATING = 5


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Discover which rides are completed from initial_config
    initial_rides = initial_config.get("UberAPI", {}).get("rides", {})
    if isinstance(initial_rides, list):
        initial_rides = {r.get("id", f"ride_{i}"): r for i, r in enumerate(initial_rides)}

    completed_ids = [rid for rid, r in initial_rides.items()
                     if r.get("status") == "completed"]

    uber = (final_state or {}).get("UberAPI", {})
    rides = uber.get("rides", {})

    # Each completed ride should be rated 5
    for ride_id in completed_ids:
        ride = rides.get(ride_id, {})
        checks[f"{ride_id}_rated_5"] = ride.get("rating") == EXPECTED_RATING

    # All original rides still present
    checks["all_rides_present"] = all(rid in rides for rid in initial_rides)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "completed_ride_ids": completed_ids,
            "ride_ratings": {
                rid: rides.get(rid, {}).get("rating") for rid in completed_ids
            },
        },
    )
