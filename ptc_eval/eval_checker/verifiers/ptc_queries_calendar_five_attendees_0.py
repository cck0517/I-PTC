"""Verifier for ptc_queries_calendar_five_attendees_0.

Check that the proposed 45-minute slot is next week, inside every attendee's
local meeting window, and has the fewest conflicts with the attendees'
existing events.

- Only the five attendees named in the prompt are checked.
- Timestamps are accepted with or without seconds.
- "Fewest conflicts": the slot's number of conflicting events must equal the
  minimum over all valid slots. Weekends next week have no events, so a slot
  on a weekday is compared with the weekday minimum and a weekend slot with
  the minimum over all days.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)

_ATTENDEES = ["alice", "bob", "charlie", "diana", "eric"]
_WINDOW = (time(10, 0), time(15, 0))
_DURATION = timedelta(minutes=45)
_GRID = timedelta(minutes=5)
_UTC = ZoneInfo("UTC")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=_UTC)


def _context(cfg: dict):
    tz = {e["id"]: e["timezone"] for e in cfg["employees"]}
    names = {e["id"]: e["display_name"] for e in cfg["employees"]}
    week_start = date.fromisoformat(cfg["metadata"]["current_week_start_date"]) + timedelta(days=7)
    week_end = week_start + timedelta(days=6)
    busy = {p: [] for p in _ATTENDEES}
    for ev in cfg["events"]:
        if ev["employee_id"] in busy:
            z = ZoneInfo(tz[ev["employee_id"]])
            busy[ev["employee_id"]].append(
                (datetime.fromisoformat(ev["start"]).replace(tzinfo=z).astimezone(_UTC),
                 datetime.fromisoformat(ev["end"]).replace(tzinfo=z).astimezone(_UTC)))
    return tz, names, week_start, week_end, busy


def _slot_info(start: datetime, tz, week_start, week_end, busy) -> dict:
    end = start + _DURATION
    per_person = {}
    weekday = True
    for p in _ATTENDEES:
        a, b = start.astimezone(ZoneInfo(tz[p])), end.astimezone(ZoneInfo(tz[p]))
        per_person[p] = {
            "in_next_week": week_start <= a.date() <= week_end,
            "in_window": a.date() == b.date() and a.time() >= _WINDOW[0] and b.time() <= _WINDOW[1],
        }
        weekday &= a.weekday() < 5
    conflicts = sum(1 for p in _ATTENDEES for s, f in busy[p] if s < end and f > start)
    valid = all(v["in_next_week"] and v["in_window"] for v in per_person.values())
    return {"per_person": per_person, "valid": valid, "weekday": weekday, "conflicts": conflicts}


def _minimums(tz, week_start, week_end, busy) -> dict:
    mins = {"weekday": None, "any": None}
    t = datetime.combine(week_start - timedelta(days=1), time(0), tzinfo=_UTC)
    stop = datetime.combine(week_end + timedelta(days=2), time(0), tzinfo=_UTC)
    while t < stop:
        info = _slot_info(t, tz, week_start, week_end, busy)
        if info["valid"]:
            c = info["conflicts"]
            mins["any"] = c if mins["any"] is None else min(mins["any"], c)
            if info["weekday"]:
                mins["weekday"] = c if mins["weekday"] is None else min(mins["weekday"], c)
        t += _GRID
    return mins


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    cfg = initial_config["EnterpriseCalendarAPI"]

    stamps = _TS.findall(final_answer or "")
    start_s = end_s = None
    if len(stamps) >= 2:
        start_s, end_s = stamps[0], stamps[1]
    else:
        js = extract_json_from_text(final_answer)
        if isinstance(js, dict):
            start_s = js.get("meeting_start_utc", js.get("start_utc"))
            end_s = js.get("meeting_end_utc", js.get("end_utc"))
    if not start_s or not end_s:
        return VerificationResult(passed=False, score=0.0,
                                  errors=["Could not extract UTC start/end from answer"],
                                  details={"raw_answer": final_answer})
    try:
        start, end = _parse(start_s), _parse(end_s)
    except ValueError:
        return VerificationResult(passed=False, score=0.0, errors=[f"Bad timestamps {start_s!r}, {end_s!r}"])

    tz, names, week_start, week_end, busy = _context(cfg)
    checks["duration_45min"] = abs((end - start) - _DURATION) < timedelta(minutes=1)
    info = _slot_info(start, tz, week_start, week_end, busy)
    checks["in_next_week"] = all(v["in_next_week"] for v in info["per_person"].values())
    for p, v in info["per_person"].items():
        checks[f"{names[p]}_in_window"] = v["in_window"]
        if not v["in_window"]:
            errors.append(f"{names[p]}: outside {_WINDOW[0]:%H:%M}-{_WINDOW[1]:%H:%M} local")

    mins = _minimums(tz, week_start, week_end, busy)
    target = mins["weekday"] if info["weekday"] else mins["any"]
    checks["fewest_conflicts"] = info["valid"] and target is not None and info["conflicts"] == target
    if not checks["fewest_conflicts"]:
        errors.append(f"Slot has {info['conflicts']} conflicting events; minimum is {target} "
                      f"({'weekdays' if info['weekday'] else 'all days'})")

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, errors=errors,
                              details={"slot_conflicts": info["conflicts"], "minimums": mins})
