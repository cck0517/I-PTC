"""Verifier for ptc_combined_3.

End-of-month financial review: compute totals from Venmo, Uber,
DoorDash. Compare to last month from calendar event. Create next
month's budget review event. Send $0.01 to self via Venmo.

Pre-computed current month (March 2026):
  Venmo outgoing: 55+42+78+35+65+48+32+25+40+30 = 450.0
  Uber rides:     18.5+12+38+19+15.5+17+22+18 = 160.0
  DoorDash:       18.98+15.48+19.98+22.48+16.98+17.48 = 111.38
  Grand total:    450.0 + 160.0 + 111.38 = 721.38

Last month (from calendar event description):
  Venmo: $380, Uber: $165, DoorDash: $195, Total: $740

Comparisons:
  Venmo:    450 vs 380 -> up (+$70)
  Uber:     160 vs 165 -> down (-$5)
  DoorDash: 111.38 vs 195 -> down (-$83.62)
  Total:    721.38 vs 740 -> down (-$18.62)

Next month event: April 1, 2026 at 09:00, titled 'Monthly Budget Review'
  with totals in description.

Venmo self-send: $0.01 to user1.
"""

from __future__ import annotations

import re
from typing import Any

from .core import VerificationResult, score_from_checks


_VENMO_TOTAL = 450.0
_UBER_TOTAL = 160.0
_DOORDASH_TOTAL = 111.38
_GRAND_TOTAL = _VENMO_TOTAL + _UBER_TOTAL + _DOORDASH_TOTAL  # 721.38
_TOLERANCE = 5.0


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # --- 1. Calendar event for next month created ---------------------------
    cal = (final_state or {}).get("PersonalCalendarAPI", {})
    cal_events = cal.get("events", {})
    initial_events = initial_config.get("PersonalCalendarAPI", {}).get("events", {})
    new_event_ids = set(cal_events.keys()) - set(initial_events.keys())
    new_events = [cal_events[eid] for eid in new_event_ids]

    budget_events = [
        e for e in new_events
        if "budget" in e.get("title", "").lower()
        or "review" in e.get("title", "").lower()
        or "financial" in e.get("title", "").lower()
    ]
    checks["next_month_event_created"] = len(budget_events) > 0

    # Check event is on April 1, 2026
    if budget_events:
        start = budget_events[0].get("start_time", "")
        checks["event_date_correct"] = "2026-04-01" in start
    else:
        checks["event_date_correct"] = False

    # Check the event description contains current totals
    if budget_events:
        desc = budget_events[0].get("description", "").lower()
        # Should mention some dollar amounts
        has_numbers = bool(re.findall(r"\d+", desc))
        checks["event_has_totals"] = has_numbers and len(desc) > 20
    else:
        checks["event_has_totals"] = False

    # --- 2. Venmo self-send ($0.01) -----------------------------------------
    venmo = (final_state or {}).get("VenmoAPI", {})
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    initial_ids = {t.get("id") for t in initial_txns}
    new_txns = [t for t in txns if t.get("id") not in initial_ids]

    self_txns = [
        t for t in new_txns
        if t.get("receiver") == "user1"
        or t.get("receiver") == t.get("sender")
    ]
    checks["venmo_self_send"] = len(self_txns) > 0

    # Check the self-send has a note with breakdown info
    if self_txns:
        note = self_txns[0].get("note", "").lower()
        checks["venmo_note_has_summary"] = len(note) > 10
    else:
        checks["venmo_note_has_summary"] = False

    # --- 3. Answer mentions correct totals ----------------------------------
    ans = (final_answer or "").lower()
    numbers = re.findall(r"\$?([\d,]+\.?\d*)", final_answer or "")
    num_vals = []
    for n in numbers:
        try:
            num_vals.append(float(n.replace(",", "")))
        except ValueError:
            pass

    # Check Venmo total mentioned (~$450)
    checks["venmo_total_mentioned"] = any(
        abs(v - _VENMO_TOTAL) <= _TOLERANCE for v in num_vals
    )

    # Check Uber total mentioned (~$160)
    checks["uber_total_mentioned"] = any(
        abs(v - _UBER_TOTAL) <= _TOLERANCE for v in num_vals
    )

    # Check DoorDash total mentioned (~$111.38)
    checks["doordash_total_mentioned"] = any(
        abs(v - _DOORDASH_TOTAL) <= _TOLERANCE for v in num_vals
    )

    # --- 4. Answer discusses comparison to last month ----------------------
    checks["compares_to_last_month"] = (
        "last month" in ans or "previous" in ans or "compared" in ans
        or ("up" in ans and "down" in ans)
        or "increase" in ans or "decrease" in ans
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=checks,
        details={
            "expected_venmo_total": _VENMO_TOTAL,
            "expected_uber_total": _UBER_TOTAL,
            "expected_doordash_total": _DOORDASH_TOTAL,
            "expected_grand_total": _GRAND_TOTAL,
            "new_calendar_events": [(e.get("title"), e.get("start_time")) for e in new_events],
            "new_venmo_txns": len(new_txns),
        },
    )
