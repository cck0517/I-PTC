"""Verifier for ptc_intermediate_calendar_dead_time_1on1s_0."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .core import VerificationResult, extract_json_from_text, score_from_checks


USER = "Lina Cho"
TEAMMATES = ["Mick Ott", "Nina Diaz", "Omar Bey", "Pia Roe", "Quin Vik"]
DAYS = ["2026-05-06", "2026-05-07"]


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "EnterpriseCalendarAPIAlt",
        initial_config.get("EnterpriseCalendarAPI", {}),
    )


def _emp_map(cfg: dict) -> dict:
    return {e["display_name"]: e for e in cfg["employees"]}


def _events_by_emp(cfg: dict) -> dict:
    by: dict[str, list[tuple[datetime, datetime]]] = {}
    for ev in cfg["events"]:
        emp_id = ev["employee_id"]
        emp = next((e for e in cfg["employees"] if e["id"] == emp_id), None)
        if emp is None:
            continue
        tz = ZoneInfo(emp["timezone"])
        es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
        ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
        by.setdefault(emp["display_name"], []).append((es, ee))
    return by


def _busy(evs, s, e) -> bool:
    for es, ee in evs:
        if s < ee and e > es:
            return True
    return False


def _feasible_slots(cfg: dict) -> dict:
    evs = _events_by_emp(cfg)
    emp_map = _emp_map(cfg)
    tz = ZoneInfo(emp_map[USER]["timezone"])
    slots: dict[str, list[tuple[str, int]]] = {m: [] for m in TEAMMATES}
    for m in TEAMMATES:
        if m not in emp_map:
            continue
        for d in DAYS:
            for h in range(9, 17):
                s = datetime.fromisoformat(f"{d}T{h:02d}:00:00").replace(tzinfo=tz)
                e = s + timedelta(hours=1)
                if _busy(evs.get(USER, []), s, e):
                    continue
                if _busy(evs.get(m, []), s, e):
                    continue
                slots[m].append((d, h))
    return slots


def _compute_ground_truth(cfg: dict) -> dict:
    slots = _feasible_slots(cfg)
    # Choose 3 teammates (among TEAMMATES) and 1 slot each such that no
    # two chosen (teammate_a, slot) and (teammate_b, slot) for Lina clash
    # (same (d,h)) — Lina can only be in one meeting per hour. Minimize
    # total dead time.
    from itertools import combinations, product
    best = None
    for trio in combinations([m for m in TEAMMATES if slots.get(m)], 3):
        for slot_combo in product(*(slots[m] for m in trio)):
            # no two slots on same day+hour
            used = set()
            ok = True
            for s in slot_combo:
                if s in used:
                    ok = False
                    break
                used.add(s)
            if not ok:
                continue
            # Group by day → compute per-day dead time
            by_day: dict[str, list[int]] = {}
            for (d, h) in slot_combo:
                by_day.setdefault(d, []).append(h)
            dead = 0
            for hs in by_day.values():
                hs.sort()
                for i in range(1, len(hs)):
                    # gap in hours between end of meeting i-1 (h+1) and start of i
                    gap = hs[i] - (hs[i - 1] + 1)
                    if gap > 0:
                        dead += gap
            # Preference: same-day placement preferred → penalize spread
            days_used = len(by_day)
            key = (days_used, dead, sorted([(d, h) for d, h in slot_combo]))
            if best is None or key < best[0]:
                best = (key, trio, slot_combo)
    if best is None:
        return {}
    _, trio, slot_combo = best
    meetings = []
    dead_total = 0
    by_day: dict[str, list[int]] = {}
    for m, (d, h) in zip(trio, slot_combo):
        meetings.append({"teammate": m, "date": d, "start_hour": h})
        by_day.setdefault(d, []).append(h)
    for hs in by_day.values():
        hs.sort()
        for i in range(1, len(hs)):
            gap = hs[i] - (hs[i - 1] + 1)
            if gap > 0:
                dead_total += gap
    return {"meetings": meetings, "total_dead_time_hours": dead_total}


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected"] = expected
    if not expected:
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["No feasible plan"])

    got = answer.get("meetings") or []
    if not isinstance(got, list):
        got = []

    def _triple(row):
        return (row.get("teammate"), row.get("date"), row.get("start_hour"))
    got_set = set()
    try:
        got_set = {(str(r["teammate"]), str(r["date"]), int(r["start_hour"]))
                   for r in got if isinstance(r, dict)}
    except (TypeError, KeyError, ValueError):
        pass
    exp_set = {(r["teammate"], r["date"], int(r["start_hour"]))
               for r in expected["meetings"]}
    checks["meetings_correct"] = got_set == exp_set
    try:
        checks["dead_time_correct"] = int(answer.get("total_dead_time_hours")) == expected["total_dead_time_hours"]
    except (TypeError, ValueError):
        checks["dead_time_correct"] = False
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
