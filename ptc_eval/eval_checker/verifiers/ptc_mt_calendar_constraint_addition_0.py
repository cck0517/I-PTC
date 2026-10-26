"""Verifier for ptc_mt_calendar_constraint_addition_0 (multi-turn).

Q1: All 1-hour Mon-Fri slots where at least 3 of {Jordan, Alex, Sam, Priya}
    are simultaneously free within their own 9-17 local business hours.
Q2: From those results, which slots have ALL 4 people free?

We recompute the expected Q2 set (4-way joint free slots inside the
intersection of 9-17 local business hours) from the initial config.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .core import (
    VerificationResult,
    extract_json_from_text,
    score_from_checks,
)

SLOT_MIN = 60
STEP_MIN = 15
PEOPLE = ["jordan", "alex", "sam", "priya"]


def _tz_by_id(config: dict) -> dict[str, ZoneInfo]:
    return {e["id"]: ZoneInfo(e["timezone"]) for e in config["employees"]}


def _events_for(config: dict, emp_id: str, tz: ZoneInfo) -> list[tuple[datetime, datetime]]:
    out = []
    for ev in config["events"]:
        if ev["employee_id"] != emp_id:
            continue
        s = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz).astimezone(ZoneInfo("UTC"))
        e = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz).astimezone(ZoneInfo("UTC"))
        out.append((s, e))
    return out


def _conflicts(s: datetime, e: datetime, events: list[tuple[datetime, datetime]]) -> bool:
    return any(s < ee and e > es for es, ee in events)


def _joint_slots(config: dict, emp_ids: list[str]) -> set[str]:
    tzs = _tz_by_id(config)
    events_by_id = {eid: _events_for(config, eid, tzs[eid]) for eid in emp_ids}
    week_start = datetime.fromisoformat(config["metadata"]["current_week_start_date"])

    slots: set[str] = set()
    for day_offset in range(5):
        day = week_start + timedelta(days=day_offset)
        window_start = None
        window_end = None
        for eid in emp_ids:
            tz = tzs[eid]
            w9 = datetime(day.year, day.month, day.day, 9, 0, tzinfo=tz).astimezone(ZoneInfo("UTC"))
            w17 = datetime(day.year, day.month, day.day, 17, 0, tzinfo=tz).astimezone(ZoneInfo("UTC"))
            window_start = w9 if window_start is None else max(window_start, w9)
            window_end = w17 if window_end is None else min(window_end, w17)

        if window_start is None or window_end is None or window_end <= window_start:
            continue

        cursor = window_start
        while cursor + timedelta(minutes=SLOT_MIN) <= window_end:
            end = cursor + timedelta(minutes=SLOT_MIN)
            if all(not _conflicts(cursor, end, events_by_id[eid]) for eid in emp_ids):
                slots.add(cursor.strftime("%Y-%m-%dT%H:%M:%SZ"))
            cursor += timedelta(minutes=STEP_MIN)

    return slots


def _normalize_utc(raw: str) -> str:
    if raw.endswith("+00:00"):
        return raw[:-6] + "Z"
    return raw


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
    turn_answers: list[str | None] | None = None,
) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )
    if isinstance(answer, dict):
        if "slots" in answer and isinstance(answer["slots"], list):
            answer = answer["slots"]
        else:
            answer = [answer]

    cfg = initial_config["EnterpriseCalendarAPI"]
    expected = _joint_slots(cfg, PEOPLE)

    answer_starts = {_normalize_utc(a.get("start_utc", "")) for a in answer if isinstance(a, dict)}

    checks: dict[str, bool] = {
        "correct_count": len(answer) == len(expected),
        "correct_slot_set": answer_starts == expected,
        "sorted_chronologically": True,
    }
    starts_list = [a.get("start_utc", "") for a in answer if isinstance(a, dict)]
    if len(starts_list) >= 2:
        checks["sorted_chronologically"] = all(
            starts_list[i] <= starts_list[i + 1] for i in range(len(starts_list) - 1)
        )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={
            "expected_count": len(expected),
            "expected_starts": sorted(expected),
            "answer_starts": sorted(answer_starts),
        },
    )
