"""Verifier for ptc_loop_7.

Add friends on Venmo. Dynamically discovers expected friends from the
question text and initial config (friends not already in the list).
"""

from __future__ import annotations
import re
from typing import Any
from .core import VerificationResult, score_from_checks


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    venmo_cfg = initial_config.get("VenmoAPI", {})
    initial_friends = set(venmo_cfg.get("friends", []))

    # Expected new friends: all usernames in username_to_id minus existing
    # friends and the current user
    all_users = set(venmo_cfg.get("username_to_id", {}).keys())
    current_user = venmo_cfg.get("current_user") or venmo_cfg.get("user_id")
    expected_new = all_users - initial_friends - ({current_user} if current_user else set())
    expected_total = initial_friends | expected_new

    venmo = (final_state or {}).get("VenmoAPI", {})
    friends = set(venmo.get("friends", []))

    # Check each expected new friend was added
    for username in sorted(expected_new):
        checks[f"added_{username}"] = username in friends

    # Total count
    checks["correct_friend_count"] = len(friends) == len(expected_total)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "final_friends": sorted(friends),
            "expected_new_friends": sorted(expected_new),
            "initial_friends": sorted(initial_friends),
        },
    )
