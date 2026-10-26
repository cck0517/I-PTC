"""Verifier for ptc_intermediate_calendar_reschedule_freeblock_0."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .core import VerificationResult, extract_json_from_text, score_from_checks


USER = "Dan Reeves"
ATTENDEES = ["Dan Reeves", "Ava Smith", "Ben Tyler", "Cora Jung"]
OLD_DATE = "2026-05-07"
OLD_HOUR = 11
WEEK_DATES = [f"2026-05-0{d}" for d in (4, 5, 6, 7, 8)]


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "EnterpriseCalendarAPIAlt",
        initial_config.get("EnterpriseCalendarAPI", {}),
    )


def _emp_by_name(cfg: dict) -> dict:
    return {e["display_name"]: e for e in cfg["employees"]}


def _busy_for(emp, events_for_emp, tz, s, e) -> bool:
    for es, ee in events_for_emp:
        if s < ee and e > es:
            return True
    return False


def _events_by_emp(cfg: dict) -> dict:
    by_emp: dict[str, list[tuple[datetime, datetime]]] = {}
    emp_map = _emp_by_name(cfg)
    for ev in cfg["events"]:
        emp_id = ev["employee_id"]
        # find the employee from id
        emp = next((e for e in cfg["employees"] if e["id"] == emp_id), None)
        if emp is None:
            continue
        tz = ZoneInfo(emp["timezone"])
        es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
        ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
        by_emp.setdefault(emp["display_name"], []).append((es, ee))
    return by_emp


def _max_free_block_hours(dan_events: list, day: str, tz: ZoneInfo) -> int:
    ds = datetime.fromisoformat(f"{day}T09:00:00").replace(tzinfo=tz)
    de = datetime.fromisoformat(f"{day}T17:00:00").replace(tzinfo=tz)
    intervals = []
    for es, ee in dan_events:
        s = max(es, ds)
        e = min(ee, de)
        if s < e:
            intervals.append((s, e))
    intervals.sort()
    merged: list[tuple] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    best = timedelta(0)
    cursor = ds
    for s, e in merged:
        if cursor < s:
            best = max(best, s - cursor)
        cursor = max(cursor, e)
    if cursor < de:
        best = max(best, de - cursor)
    return int(best.total_seconds() // 3600)


def _compute_ground_truth(cfg: dict) -> dict:
    emp_map = _emp_by_name(cfg)
    evs_by_emp = _events_by_emp(cfg)

    # Build attendee-specific busy events EXCLUDING the "Design review".
    def _filter_design(evs):
        # by iso start
        return [
            (s, e) for (s, e) in evs
            if not (s.strftime("%Y-%m-%dT%H:%M:%S") == f"{OLD_DATE}T{OLD_HOUR:02d}:00:00"
                    and e.strftime("%Y-%m-%dT%H:%M:%S") == f"{OLD_DATE}T{OLD_HOUR+1:02d}:00:00")
        ]

    dan_tz = ZoneInfo(emp_map[USER]["timezone"])
    att_busy = {n: _filter_design(evs_by_emp.get(n, [])) for n in ATTENDEES}

    best = None
    for d in WEEK_DATES:
        for h in range(9, 17):
            s = datetime.fromisoformat(f"{d}T{h:02d}:00:00").replace(tzinfo=dan_tz)
            e = s + timedelta(hours=1)
            # skip the original slot
            if d == OLD_DATE and h == OLD_HOUR:
                continue
            if all(not _busy_for(n, att_busy[n], dan_tz, s, e) for n in ATTENDEES):
                # compute Dan's max free block on day d AFTER move:
                # Dan's schedule on d = existing dan events (minus design review
                # if d==OLD_DATE) plus new meeting at (d, h)..(d, h+1).
                dan_day_events = list(att_busy[USER])
                dan_day_events.append((s, e))
                max_block = _max_free_block_hours(dan_day_events, d, dan_tz)
                cand = (-max_block, d, h)
                if best is None or cand < best[0]:
                    best = (cand, d, h, max_block)
    if best is None:
        return {}
    (_, nd, nh, mb) = best
    return {
        "new_date": nd,
        "new_start_hour": nh,
        "max_free_block_hours": mb,
    }


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
                                  errors=["No feasible slot in initial_config"])
    checks["new_date_correct"] = str(answer.get("new_date")) == expected["new_date"]
    try:
        checks["new_hour_correct"] = int(answer.get("new_start_hour")) == expected["new_start_hour"]
    except (TypeError, ValueError):
        checks["new_hour_correct"] = False
    try:
        checks["max_free_block_correct"] = int(answer.get("max_free_block_hours")) == expected["max_free_block_hours"]
    except (TypeError, ValueError):
        checks["max_free_block_correct"] = False
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
