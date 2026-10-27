"""Verifier for ptc_combined_3.

End-of-month financial review: compute totals from Venmo, Uber,
DoorDash. Compare to last month from calendar event. Create next
month's budget review event. Send $0.01 to self via Venmo.

The expected totals are computed from the scenario (all of its data is in
March 2026): Venmo outgoing payments, Uber ride fares, and DoorDash order
totals. Seven Uber rides are cancelled but still carry a fare, so the Uber
total is accepted with or without them.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from .core import VerificationResult, score_from_checks

_TOLERANCE = 5.0


def _expected(initial_config: dict) -> dict:
    venmo = initial_config.get("VenmoAPI", {})
    me = venmo.get("current_user")
    venmo_total = sum(float(t["amount"]) for t in venmo.get("transactions", [])
                      if t.get("sender") == me and t.get("type", "payment") == "payment")
    rides = initial_config.get("UberAPI", {}).get("rides", [])
    rides = rides if isinstance(rides, list) else list(rides.values())
    uber_completed = sum(float(r["fare"]) for r in rides if r.get("status", "completed") == "completed")
    uber_all = sum(float(r["fare"]) for r in rides)
    orders = initial_config.get("DoorDashAPI", {}).get("orders", [])
    orders = orders if isinstance(orders, list) else list(orders.values())
    doordash_total = sum(float(o["total"]) for o in orders)
    today = date.fromisoformat(initial_config["PersonalCalendarAPI"]["current_date"][:10])
    first_next = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
    return {
        "venmo": round(venmo_total, 2),
        "uber": sorted({round(uber_completed, 2), round(uber_all, 2)}),
        "doordash": round(doordash_total, 2),
        "next_month_first": first_next.isoformat(),
    }


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    exp = _expected(initial_config)

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
    if budget_events:
        start = budget_events[0].get("start_time", "")
        checks["event_date_correct"] = exp["next_month_first"] in start
        desc = budget_events[0].get("description", "").lower()
        checks["event_has_totals"] = bool(re.findall(r"\d+", desc)) and len(desc) > 20
    else:
        checks["event_date_correct"] = False
        checks["event_has_totals"] = False

    # --- 2. Venmo self-send ($0.01) -----------------------------------------
    venmo = (final_state or {}).get("VenmoAPI", {})
    me = initial_config.get("VenmoAPI", {}).get("current_user", "user1")
    txns = venmo.get("transactions", [])
    initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
    # Transactions are append-only; take everything after the initial list so
    # detection does not depend on ids being unique.
    new_txns = txns[len(initial_txns):] if len(txns) >= len(initial_txns) else []
    self_txns = [
        t for t in new_txns
        if t.get("receiver") == me or t.get("receiver") == t.get("sender")
    ]
    checks["venmo_self_send"] = len(self_txns) > 0
    if self_txns:
        checks["venmo_note_has_summary"] = len(self_txns[0].get("note", "")) > 10
    else:
        checks["venmo_note_has_summary"] = False

    # --- 3. Answer mentions correct totals ----------------------------------
    ans = (final_answer or "").lower()
    num_vals = []
    for n in re.findall(r"\$?([\d,]+\.?\d*)", final_answer or ""):
        try:
            num_vals.append(float(n.replace(",", "")))
        except ValueError:
            pass
    near = lambda target: any(abs(v - target) <= _TOLERANCE for v in num_vals)  # noqa: E731
    checks["venmo_total_mentioned"] = near(exp["venmo"])
    checks["uber_total_mentioned"] = any(near(u) for u in exp["uber"])
    checks["doordash_total_mentioned"] = near(exp["doordash"])

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
            "expected": exp,
            "new_calendar_events": [(e.get("title"), e.get("start_time")) for e in new_events],
            "new_venmo_txns": len(new_txns),
        },
    )
