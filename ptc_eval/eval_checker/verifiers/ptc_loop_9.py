"""Verifier for ptc_loop_9.

Accept all pending Venmo payment requests.
Dynamically computes expected balance from initial_config.
"""

from __future__ import annotations
from typing import Any
from .core import VerificationResult, score_from_checks

TOLERANCE = 0.02


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Compute expected values from initial_config
    venmo_cfg = initial_config.get("VenmoAPI", {})
    initial_balance = venmo_cfg.get("balance", 0.0)
    initial_pending = venmo_cfg.get("pending_requests", [])
    total_pending = sum(r.get("amount", 0) for r in initial_pending)
    expected_balance = initial_balance - total_pending

    venmo = (final_state or {}).get("VenmoAPI", {})
    pending = venmo.get("pending_requests", [])
    balance = venmo.get("balance", 0.0)

    # All pending requests should be cleared
    checks["no_pending_requests"] = len(pending) == 0

    # Balance should decrease by the total of all request amounts
    checks["correct_balance"] = abs(balance - expected_balance) <= TOLERANCE

    # Transactions should record the accepted payments
    transactions = venmo.get("transactions", [])
    checks["transactions_recorded"] = len(transactions) >= len(initial_pending)

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "final_pending_count": len(pending),
            "final_balance": balance,
            "expected_balance": expected_balance,
            "total_paid": total_pending,
            "num_initial_pending": len(initial_pending),
            "transaction_count": len(transactions),
        },
    )
