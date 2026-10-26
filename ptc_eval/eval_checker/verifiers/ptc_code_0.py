"""Verifier for ptc_code_0.

Analyse Venmo transactions, find the top recipient, compute spending
percentages, and send that person a message via Venmo.

Expected values (from the 15 initial transactions):
  alice_v:   55 + 42 + 78 + 62 + 45 = 282
  bob_smith: 65 + 48 + 40 = 153
  charlie99: 28 + 22.5 + 38 + 33 = 121.5
  diana_k:   35 + 25 + 30 = 90
  Total outgoing = 646.50
  Top recipient = alice_v ($282, 43.6% of total)
  Average txn size = 646.50 / 15 = 43.10
  Balance = 850, daily spending rate = 646.50 / 30 ~ 21.55, days = ~39

The model should send a Venmo message/payment to alice_v.  We check
that a new transaction to alice_v was created in the final state.
"""

from __future__ import annotations

import re
from typing import Any

from .core import VerificationResult, score_from_checks


# -- Pre-computed ground-truth values ----------------------------------------
_TOP_RECIPIENT = "alice_v"
_TOP_AMOUNT = 282.0
_TOTAL_OUTGOING = 646.5
_TOP_PCT = round(_TOP_AMOUNT / _TOTAL_OUTGOING * 100, 1)  # ~43.6
_TOLERANCE = 2.0  # dollars
_PCT_TOLERANCE = 2.0  # percentage points


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # --- 1. Check final_answer mentions the correct top recipient -----------
    ans = (final_answer or "").lower()
    checks["top_recipient_mentioned"] = "alice" in ans or "alice_v" in ans

    # --- 2. Check that $282 or close appears in answer ----------------------
    numbers = re.findall(r"\$?([\d,]+\.?\d*)", final_answer or "")
    found_total = False
    for n in numbers:
        try:
            val = float(n.replace(",", ""))
            if abs(val - _TOP_AMOUNT) <= _TOLERANCE:
                found_total = True
                break
        except ValueError:
            continue
    checks["top_amount_in_answer"] = found_total

    # --- 3. Check percentage of spending in answer --------------------------
    found_pct = False
    for n in numbers:
        try:
            val = float(n.replace(",", ""))
            if abs(val - _TOP_PCT) <= _PCT_TOLERANCE:
                found_pct = True
                break
        except ValueError:
            continue
    checks["percentage_in_answer"] = found_pct

    # --- 4. Check state: a new Venmo transaction to alice_v was created -----
    if final_state:
        venmo = final_state.get("VenmoAPI", {})
        txns = venmo.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        initial_ids = {t.get("id") for t in initial_txns}
        new_txns = [t for t in txns if t.get("id") not in initial_ids]
        sent_to_alice = any(
            t.get("receiver") == "alice_v"
            for t in new_txns
        )
        checks["venmo_message_sent"] = sent_to_alice
    else:
        checks["venmo_message_sent"] = False

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "expected_top_recipient": _TOP_RECIPIENT,
            "expected_top_amount": _TOP_AMOUNT,
            "expected_total_outgoing": _TOTAL_OUTGOING,
            "expected_percentage": _TOP_PCT,
        },
    )
