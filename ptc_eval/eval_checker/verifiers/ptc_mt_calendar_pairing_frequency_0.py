"""Verifier for ptc_mt_calendar_pairing_frequency_0.

Q1: top-5 employee pairs by shared meeting count (matched by (UTC start, title, duration_min)).
Q2: for each top-5 pair, list the actual shared meetings (title, start_utc, duration_minutes).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from zoneinfo import ZoneInfo

from ._calendar_helpers import (
    employees_by_id, event_duration_minutes, fmt_utc, to_utc,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _shared_meetings(cfg: dict) -> dict[tuple, list[str]]:
    """{(start_utc_str, title, duration_min): [employee_ids]}."""
    emps = employees_by_id(cfg)
    out: dict[tuple, list[str]] = defaultdict(list)
    for ev in cfg["events"]:
        emp = emps.get(ev["employee_id"])
        if not emp:
            continue
        tz = ZoneInfo(emp["timezone"])
        s_utc = fmt_utc(to_utc(ev["start"], tz))
        dur = event_duration_minutes(ev)
        key = (s_utc, ev["title"], dur)
        if ev["employee_id"] not in out[key]:
            out[key].append(ev["employee_id"])
    return dict(out)


def _q1_top5_pairs(cfg: dict, shared: dict) -> list[dict]:
    emps = employees_by_id(cfg)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for atts in shared.values():
        if len(atts) < 2:
            continue
        for a, b in combinations(sorted(atts), 2):
            pair_counts[(a, b)] += 1
    rows = []
    for (a, b), n in pair_counts.items():
        names = sorted([emps[a]["display_name"], emps[b]["display_name"]])
        rows.append({
            "pair": [a, b],
            "display_names": names,
            "shared_meeting_count": n,
        })
    rows.sort(key=lambda r: (-r["shared_meeting_count"], r["display_names"]))
    return rows[:5]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    shared = _shared_meetings(cfg)
    top5 = _q1_top5_pairs(cfg, shared)
    out = []
    for r in top5:
        a, b = r["pair"]
        meetings = []
        for (s_utc, title, dur), atts in shared.items():
            if a in atts and b in atts:
                meetings.append({
                    "title": title,
                    "start_utc": s_utc,
                    "duration_minutes": dur,
                })
        meetings.sort(key=lambda m: m["start_utc"])
        out.append({
            "pair": r["pair"],
            "display_names": r["display_names"],
            "meetings": meetings,
        })
    return {"top5_pair_meetings": out}


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
    p_ans = answer.get("top5_pair_meetings") or []
    p_exp = expected["top5_pair_meetings"]
    checks: dict[str, bool] = {
        "count_match": len(p_ans) == len(p_exp),
    }
    ans_pairs = [tuple(p.get("pair") or []) for p in p_ans if isinstance(p, dict)]
    exp_pairs = [tuple(p["pair"]) for p in p_exp]
    checks["pair_set"] = set(ans_pairs) == set(exp_pairs)
    checks["pair_order"] = ans_pairs == exp_pairs

    exp_by_pair = {tuple(p["pair"]): p for p in p_exp}
    meet_ok = True
    for p in p_ans:
        if not isinstance(p, dict):
            meet_ok = False; break
        e = exp_by_pair.get(tuple(p.get("pair") or []))
        if e is None:
            continue
        ans_keys = {
            (m.get("title"), m.get("start_utc"), m.get("duration_minutes"))
            for m in (p.get("meetings") or []) if isinstance(m, dict)
        }
        exp_keys = {
            (m["title"], m["start_utc"], m["duration_minutes"])
            for m in e["meetings"]
        }
        if ans_keys != exp_keys:
            meet_ok = False; break
    checks["meetings_match"] = meet_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
