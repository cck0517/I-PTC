"""Verifier for ptc_mt_calendar_back_to_back_0.

Q1: largest threshold T s.t. >=5 employees have accepted_event_count >= T.
Q2: full threshold sweep + top_loaded employee.
"""
from __future__ import annotations

from collections import defaultdict

from ._calendar_helpers import employees_by_id
from .core import VerificationResult, extract_json_from_text, score_from_checks

THRESHOLDS = [10, 15, 20, 25, 30, 35]


def _accepted_counts(cfg: dict) -> dict[str, int]:
    counts = defaultdict(int)
    for ev in cfg["events"]:
        if ev.get("rsvp") == "accepted":
            counts[ev["employee_id"]] += 1
    return dict(counts)


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["EnterpriseCalendarAPI"]
    emps = employees_by_id(cfg)
    counts = _accepted_counts(cfg)
    sweep = []
    for t in THRESHOLDS:
        n = sum(1 for c in counts.values() if c >= t)
        sweep.append({"T": t, "qualifying_count": n})
    rows = [
        {"employee_id": eid,
         "display_name": emps[eid]["display_name"],
         "accepted_event_count": c}
        for eid, c in counts.items()
        if eid in emps
    ]
    if not rows:
        return {"threshold_sweep": sweep, "top_loaded": None}
    top = sorted(rows, key=lambda r: (-r["accepted_event_count"], r["display_name"]))[0]
    return {"threshold_sweep": sweep, "top_loaded": top}


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

    sw_ans = answer.get("threshold_sweep") or []
    sw_exp = expected["threshold_sweep"]
    checks: dict[str, bool] = {
        "sweep_count": len(sw_ans) == len(sw_exp),
        "sweep_order": [s.get("T") for s in sw_ans if isinstance(s, dict)] ==
                       [s["T"] for s in sw_exp],
    }
    sweep_ok = True
    exp_by_t = {s["T"]: s for s in sw_exp}
    for s in sw_ans:
        if not isinstance(s, dict):
            sweep_ok = False; break
        e = exp_by_t.get(s.get("T"))
        if e is None:
            sweep_ok = False; break
        if s.get("qualifying_count") != e["qualifying_count"]:
            sweep_ok = False; break
    checks["sweep_counts_match"] = sweep_ok

    tl_ans = answer.get("top_loaded")
    tl_exp = expected["top_loaded"]
    if tl_exp is None:
        checks["top_loaded"] = tl_ans is None
    else:
        checks["top_loaded"] = (
            isinstance(tl_ans, dict)
            and tl_ans.get("employee_id") == tl_exp["employee_id"]
            and tl_ans.get("accepted_event_count") == tl_exp["accepted_event_count"]
        )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
