"""Verifier for ptc_code_0.

Analyse Venmo transactions, find the top recipient, compute spending
percentages, and send that person a message via Venmo.

The expected values are computed from the scenario. All transactions are in
March 2026 and the API's "today" is 2026-03-31, so "the past month" is read
either as the calendar month of the latest transaction or as the 30 days
ending on the latest transaction date. An answer passes if the top recipient,
its total and its percentage all match one of those readings.

Venmo has no plain message call, so the message may be sent as a payment
(e.g. $0.01) or as a payment request; either must go to the top recipient.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from .core import VerificationResult, score_from_checks

_TOLERANCE = 2.0  # dollars
_PCT_TOLERANCE = 2.0  # percentage points


def _readings(initial_config: dict) -> list[dict]:
    venmo = initial_config.get("VenmoAPI", {})
    me = venmo.get("current_user")
    outgoing = [t for t in venmo.get("transactions", [])
                if t.get("sender") == me and t.get("type", "payment") == "payment"]
    if not outgoing:
        return []
    latest = max(date.fromisoformat(t["date"][:10]) for t in outgoing)
    windows = {
        "calendar_month": (latest.replace(day=1), latest),
        "last_30_days": (latest - timedelta(days=29), latest),
        "last_30_days_inclusive": (latest - timedelta(days=30), latest),
    }
    readings = []
    for label, (lo, hi) in windows.items():
        per: dict[str, float] = {}
        for t in outgoing:
            if lo <= date.fromisoformat(t["date"][:10]) <= hi:
                per[t["receiver"]] = per.get(t["receiver"], 0.0) + float(t["amount"])
        if not per:
            continue
        top_user, top_amount = max(per.items(), key=lambda kv: kv[1])
        total = sum(per.values())
        readings.append({
            "reading": label, "top_recipient": top_user, "top_amount": round(top_amount, 2),
            "total": round(total, 2), "top_pct": round(top_amount / total * 100, 2),
        })
    return readings


def _numbers(text: str) -> list[float]:
    vals = []
    for n in re.findall(r"\$?([\d,]+\.?\d*)", text or ""):
        try:
            vals.append(float(n.replace(",", "")))
        except ValueError:
            continue
    return vals


def _new_entries(final_state: dict | None, initial_config: dict) -> list[dict]:
    venmo = (final_state or {}).get("VenmoAPI", {})
    init = initial_config.get("VenmoAPI", {})
    out = []
    for key in ("transactions", "pending_requests"):
        before = {t.get("id") for t in init.get(key, [])}
        out += [t for t in venmo.get(key, []) if t.get("id") not in before]
    return out


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    readings = _readings(initial_config)
    ans = (final_answer or "").lower()
    nums = _numbers(final_answer or "")
    new = _new_entries(final_state, initial_config) if final_state else []

    best_checks: dict[str, bool] | None = None
    best_reading: dict | None = None
    for r in readings:
        user = r["top_recipient"].lower()
        checks = {
            "top_recipient_mentioned": user in ans or user.split("_")[0] in ans,
            "top_amount_in_answer": any(abs(v - r["top_amount"]) <= _TOLERANCE for v in nums),
            "percentage_in_answer": any(abs(v - r["top_pct"]) <= _PCT_TOLERANCE for v in nums),
            "venmo_message_sent": bool(final_state) and any(t.get("receiver") == r["top_recipient"] for t in new),
        }
        if best_checks is None or sum(checks.values()) > sum(best_checks.values()):
            best_checks, best_reading = checks, r
    if best_checks is None:
        best_checks = {"scenario_has_outgoing_payments": False}

    passed, score = score_from_checks(best_checks)
    return VerificationResult(
        passed=passed,
        score=score,
        checks=best_checks,
        details={"matched_reading": best_reading, "all_readings": readings},
    )
