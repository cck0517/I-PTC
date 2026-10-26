"""Verifier for ptc_mt_calendar_deep_work_slot_0.

Q2: Days where the longest free block >= 90 min. For each, the earliest
    90-min-or-longer free block's start time. Among ties, pick fewest total_events.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _my_events(cfg: dict) -> list[tuple[date, datetime, datetime]]:
    """Return Alex's events as (date, start_local, end_local) tuples."""
    emp = next(e for e in cfg["employees"] if e["id"] == "alex")
    tz = ZoneInfo(emp["timezone"])
    out = []
    for ev in cfg["events"]:
        if ev["employee_id"] != "alex":
            continue
        s = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
        e = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
        out.append((s.date(), s, e))
    return out


def _day_free_blocks(day: date, events: list[tuple[datetime, datetime]], tz: ZoneInfo) -> list[tuple[datetime, datetime]]:
    """Return non-overlapping free blocks within 9-17 local for a day."""
    w9 = datetime(day.year, day.month, day.day, 9, 0, tzinfo=tz)
    w17 = datetime(day.year, day.month, day.day, 17, 0, tzinfo=tz)
    # Sort events, clip to window
    busy = []
    for s, e in events:
        s = max(s, w9); e = min(e, w17)
        if s < e:
            busy.append((s, e))
    busy.sort()
    # Merge overlapping busy intervals
    merged = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    # Compute free blocks
    free = []
    cursor = w9
    for s, e in merged:
        if cursor < s:
            free.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < w17:
        free.append((cursor, w17))
    return free


def _compute_expected(config: dict) -> dict:
    cfg = config["EnterpriseCalendarAPI"]
    week_start = date.fromisoformat(cfg["metadata"]["current_week_start_date"])
    week_end = date.fromisoformat(cfg["metadata"]["current_week_end_date"])
    tz = ZoneInfo(cfg["employees"][0]["timezone"])
    events = _my_events(cfg)
    events_by_day: dict[date, list] = {}
    for d, s, e in events:
        events_by_day.setdefault(d, []).append((s, e))

    days_info = []
    d = week_start
    while d <= week_end:
        if d.weekday() < 5:
            day_events = events_by_day.get(d, [])
            blocks = _day_free_blocks(d, day_events, tz)
            longest = max((int((b[1] - b[0]).total_seconds() // 60) for b in blocks), default=0)
            earliest_90 = next(
                (b[0] for b in blocks if (b[1] - b[0]).total_seconds() / 60 >= 90),
                None,
            )
            days_info.append({
                "date": d.isoformat(),
                "longest_free_block_minutes": longest,
                "total_events": len(day_events),
                "earliest_90min_start_local": (earliest_90.strftime("%Y-%m-%dT%H:%M:%S")
                                                if earliest_90 else None),
            })
        d += timedelta(days=1)

    candidates = [x for x in days_info if x["longest_free_block_minutes"] >= 90]
    if not candidates:
        return {"candidate_days": [], "recommended_date": None}

    # Pick recommended: max longest_free_block, then fewest total_events, then earliest date
    recommended = min(
        candidates,
        key=lambda x: (-x["longest_free_block_minutes"], x["total_events"], x["date"]),
    )
    return {
        "candidate_days": sorted(candidates, key=lambda x: x["date"]),
        "recommended_date": recommended["date"],
    }


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    cand_ans = answer.get("candidate_days") or []
    ans_dates = {c.get("date") for c in cand_ans if isinstance(c, dict)}
    exp_dates = {c["date"] for c in expected["candidate_days"]}
    checks = {
        "candidate_date_set":   ans_dates == exp_dates,
        "recommended_correct":  answer.get("recommended_date") == expected["recommended_date"],
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
