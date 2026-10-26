"""Verifier for ptc_mt_calendar_team_free_block_0.

Q1: longest contiguous free UTC block (weekday 09:00-17:00 UTC) for 5 attendees.
Q2: top-5 longest free blocks for the same 5 attendees in the same window.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ._calendar_helpers import (
    busy_intervals_utc_for, employee_id_by_display_name, fmt_utc,
    free_blocks_in_window, merge_intervals, parse_window, weekdays_in,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks

ATTENDEES = ["Alex Reyes", "Beth Chen", "Diana Park", "Gabe Harper", "Hana Mori"]
UTC = ZoneInfo("UTC")


def _all_blocks(cfg: dict) -> list[dict]:
    """All free blocks (weekday 09-17 UTC) where ALL 5 attendees are free."""
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
            if mins <= 0:
                continue
            out.append({
                "start_utc": fmt_utc(fs),
                "end_utc": fmt_utc(fe),
                "duration_minutes": mins,
            })
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    blocks = _all_blocks(cfg)
    top5 = sorted(blocks, key=lambda b: (-b["duration_minutes"], b["start_utc"]))[:5]
    return {"attendees": list(ATTENDEES), "top5_blocks": top5}


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
    checks: dict[str, bool] = {
        "attendees_match": (answer.get("attendees") or []) == expected["attendees"],
    }
    b_ans = answer.get("top5_blocks") or []
    b_exp = expected["top5_blocks"]
    checks["count_match"] = len(b_ans) == len(b_exp)

    ans_starts = [b.get("start_utc") for b in b_ans if isinstance(b, dict)]
    exp_starts = [b["start_utc"] for b in b_exp]
    checks["block_set"] = set(ans_starts) == set(exp_starts)
    checks["block_order"] = ans_starts == exp_starts

    exp_by_start = {b["start_utc"]: b for b in b_exp}
    fields_ok = True
    for b in b_ans:
        if not isinstance(b, dict):
            fields_ok = False; break
        e = exp_by_start.get(b.get("start_utc"))
        if e is None:
            continue
        if b.get("duration_minutes") != e["duration_minutes"]:
            fields_ok = False; break
        if b.get("end_utc") != e["end_utc"]:
            fields_ok = False; break
    checks["fields_match"] = fields_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
