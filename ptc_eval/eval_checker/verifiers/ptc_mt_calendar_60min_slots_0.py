"""Verifier for ptc_mt_calendar_60min_slots_0.

Q1: ALL >=60-min free slots for 5 attendees, weekday 09-17 UTC.
Q2: subset >=90 min + per-day slot count for those Q1 slots.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from ._calendar_helpers import (
    busy_intervals_utc_for, employee_id_by_display_name, fmt_utc,
    free_blocks_in_window, merge_intervals, parse_window, weekdays_in,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks

ATTENDEES = ["Alex Reyes", "Beth Chen", "Diana Park", "Gabe Harper", "Hana Mori"]
UTC = ZoneInfo("UTC")
MIN_DURATION = 60


def _q1_slots(cfg: dict) -> list[dict]:
    win_lo, win_hi = parse_window(cfg)
    emps = {e["id"]: e for e in cfg["employees"]}
    busy_all = []
    for name in ATTENDEES:
        eid = employee_id_by_display_name(cfg, name)
        if eid is None:
            continue
        tz = ZoneInfo(emps[eid]["timezone"])
        busy_all.extend(busy_intervals_utc_for(cfg, eid, tz))
    busy_union = merge_intervals(busy_all)
    out = []
    for d in weekdays_in(win_lo, win_hi):
        ws = datetime(d.year, d.month, d.day, 9, 0, tzinfo=UTC)
        we = datetime(d.year, d.month, d.day, 17, 0, tzinfo=UTC)
        for fs, fe in free_blocks_in_window(busy_union, ws, we):
            mins = int((fe - fs).total_seconds() // 60)
            if mins >= MIN_DURATION:
                out.append({
                    "start_utc": fmt_utc(fs),
                    "end_utc": fmt_utc(fe),
                    "duration_minutes": mins,
                })
    out.sort(key=lambda s: s["start_utc"])
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    slots = _q1_slots(cfg)
    long_slots = [s for s in slots if s["duration_minutes"] >= 90]
    by_day = defaultdict(int)
    for s in slots:
        by_day[s["start_utc"][:10]] += 1
    spd = sorted(
        [{"date_utc": d, "slot_count": c} for d, c in by_day.items()],
        key=lambda x: x["date_utc"],
    )
    return {
        "long_slots": long_slots,
        "long_slot_count": len(long_slots),
        "slots_per_day": spd,
    }


def verify(final_answer, final_state, initial_config, ground_truth,
           turn_answers=None) -> VerificationResult:
    answer = extract_json_from_text(final_answer)
    if not isinstance(answer, dict):
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Expected JSON object at top level"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)

    ls_ans = answer.get("long_slots") or []
    ls_exp = expected["long_slots"]
    checks: dict[str, bool] = {
        "long_slot_count": answer.get("long_slot_count") == expected["long_slot_count"],
        "long_slots_count_match": len(ls_ans) == len(ls_exp),
    }
    ans_starts = [s.get("start_utc") for s in ls_ans if isinstance(s, dict)]
    exp_starts = [s["start_utc"] for s in ls_exp]
    checks["long_slots_set"] = set(ans_starts) == set(exp_starts)
    checks["long_slots_order"] = ans_starts == exp_starts

    spd_ans = answer.get("slots_per_day") or []
    spd_exp = expected["slots_per_day"]
    checks["spd_count"] = len(spd_ans) == len(spd_exp)
    spd_ans_set = {(s.get("date_utc"), s.get("slot_count"))
                   for s in spd_ans if isinstance(s, dict)}
    spd_exp_set = {(s["date_utc"], s["slot_count"]) for s in spd_exp}
    checks["spd_match"] = spd_ans_set == spd_exp_set

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
