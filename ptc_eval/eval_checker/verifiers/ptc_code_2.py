"""Verifier for ptc_code_2.

Analyse Uber ride history: total spent, average fare, std dev,
weekday vs weekend splits, most expensive ride, most frequent
destination. Conditionally create calendar event and Venmo request.

The expected values are computed from the scenario. "Today" is the
PersonalCalendarAPI's current_date (2026-03-31). "The past 3 months" is read
as the three calendar months back from today, the calendar months
January-March, or the last 90 days; in this scenario all three select the same
25 rides ($588.50, most expensive $44.00, most frequent destination
"50 Fremont St"). The monthly average ($196.17) does not exceed $200, so no
calendar event or Venmo request should be created.
"""

from __future__ import annotations

import calendar
import re
from collections import Counter
from datetime import date, timedelta
from typing import Any

from .core import VerificationResult, score_from_checks

_TOLERANCE = 2.0
_MAX_FARE_TOLERANCE = 0.5
_THRESHOLD = 200.0
_PARTNER = "diana_k"


def _months_back(d: date, n: int) -> date:
    """Same day n months earlier, clamped to the last day of that month."""
    y, m = d.year, d.month - n
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _readings(initial_config: dict) -> list[dict]:
    rides = initial_config.get("UberAPI", {}).get("rides", [])
    rides = rides if isinstance(rides, list) else list(rides.values())
    rides = [r for r in rides if r.get("status", "completed") == "completed"]
    today = date.fromisoformat(initial_config["PersonalCalendarAPI"]["current_date"][:10])
    windows = {
        "three_months_back": (_months_back(today, 3), today),
        "calendar_months": (_months_back(today.replace(day=1), 2), today),
        "last_90_days": (today - timedelta(days=90), today),
    }
    out = []
    for label, (lo, hi) in windows.items():
        sel = [r for r in rides if lo <= date.fromisoformat(r["date"][:10]) <= hi]
        if not sel:
            continue
        total = round(sum(float(r["fare"]) for r in sel), 2)
        out.append({
            "reading": label, "rides": len(sel), "total": total,
            "max_fare": max(float(r["fare"]) for r in sel),
            "top_destination": Counter(
                r.get("end_location", r.get("destination", "")) for r in sel
            ).most_common(1)[0][0],
            "monthly_avg": round(total / 3, 2),
        })
    return out


def _new(final_state: dict | None, initial_config: dict, api: str, key: str) -> list:
    state = (final_state or {}).get(api, {})
    before_raw = initial_config.get(api, {}).get(key, {})
    after_raw = state.get(key, {})
    if isinstance(after_raw, dict):
        before = set(before_raw.keys()) if isinstance(before_raw, dict) else set()
        return [after_raw[k] for k in after_raw if k not in before]
    before_ids = {t.get("id") for t in (before_raw or [])}
    return [t for t in after_raw if t.get("id") not in before_ids]


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    try:
        return _verify(final_answer, final_state, initial_config)
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])


def _verify(final_answer: str | None, final_state: dict | None, initial_config: dict) -> VerificationResult:
    readings = _readings(initial_config)
    ans = (final_answer or "").lower()
    nums = []
    for n in re.findall(r"\$?([\d,]+\.?\d*)", final_answer or ""):
        try:
            nums.append(float(n.replace(",", "")))
        except ValueError:
            pass

    new_events = _new(final_state, initial_config, "PersonalCalendarAPI", "events")
    budget_events = [
        e for e in new_events
        if any(w in str(e.get("title", "")).lower() for w in ("budget", "transportation", "review"))
    ]
    venmo_new = (_new(final_state, initial_config, "VenmoAPI", "transactions")
                 + _new(final_state, initial_config, "VenmoAPI", "pending_requests"))
    partner_requests = [t for t in venmo_new if t.get("receiver") == _PARTNER]

    best_checks: dict[str, bool] | None = None
    best: dict | None = None
    for r in readings:
        needs_action = r["monthly_avg"] > _THRESHOLD
        dest = r["top_destination"].lower()
        street = " ".join(dest.split()[1:]) if dest.split() and dest.split()[0].isdigit() else dest
        checks = {
            "total_spent_mentioned": any(abs(v - r["total"]) <= _TOLERANCE for v in nums),
            "most_frequent_dest": dest in ans or (bool(street) and street in ans),
            "most_expensive_mentioned": any(abs(v - r["max_fare"]) <= _MAX_FARE_TOLERANCE for v in nums),
            "calendar_logic_correct": (len(budget_events) > 0) == needs_action,
            "venmo_logic_correct": (len(partner_requests) > 0) == needs_action,
            "weekday_weekend_analysis": "weekday" in ans or "weekend" in ans,
        }
        if best_checks is None or sum(checks.values()) > sum(best_checks.values()):
            best_checks, best = checks, r
    if best_checks is None:
        best_checks = {"scenario_has_rides_in_window": False}

    passed, score = score_from_checks(best_checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=best_checks,
        details={
            "matched_reading": best,
            "all_readings": readings,
            "new_calendar_events": len(new_events),
            "new_venmo_to_partner": len(partner_requests),
        },
    )
