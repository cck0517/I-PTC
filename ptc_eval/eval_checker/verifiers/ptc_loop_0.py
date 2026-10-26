"""Verifier for ptc_loop_0.

Send Venmo payment requests to all friends for $30 each with correct note.
Dynamically discovers expected friends from initial_config.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks

EXPECTED_AMOUNT = 30.0
EXPECTED_NOTE = "Dinner split at Olive Garden"


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Discover expected friends from initial_config
    venmo_cfg = initial_config.get("VenmoAPI", {})
    expected_friends = set(venmo_cfg.get("friends", []))

    venmo = (final_state or {}).get("VenmoAPI", {})

    # request_payment() creates entries in transactions with type="request"
    all_txns = venmo.get("transactions", [])
    requests = [t for t in all_txns if t.get("type") == "request"]

    # Check correct number of requests
    checks["correct_request_count"] = len(requests) == len(expected_friends)

    # Check each request targets a unique expected friend
    requested_users = set()
    amounts_correct = True
    notes_correct = True
    for req in requests:
        to_user = req.get("receiver", "")
        requested_users.add(to_user)
        if req.get("amount") != EXPECTED_AMOUNT:
            amounts_correct = False
        note = req.get("note", "")
        if EXPECTED_NOTE.lower() not in note.lower():
            notes_correct = False

    checks["correct_recipients"] = requested_users == expected_friends
    checks["all_amounts_30"] = amounts_correct
    checks["correct_note"] = notes_correct

    # Balance should be unchanged (requests don't cost money)
    initial_balance = venmo_cfg.get("balance", 500.0)
    checks["balance_unchanged"] = venmo.get("balance") == initial_balance

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "request_transactions": requests,
            "expected_friends": sorted(expected_friends),
        },
    )
