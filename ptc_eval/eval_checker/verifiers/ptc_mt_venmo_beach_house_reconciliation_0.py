"""Verifier for ptc_mt_venmo_beach_house_reconciliation_0.

Q1: per-attendee paid_so_far / still_owes (8 attendees, $300 share each,
    incoming-only since Feb 1 2026, note must contain shore/airbnb/trip
    case-insensitive).
Q2: reassign Sam's still_owes onto Megan (Sam -> 0, Megan += Sam's old).
    paid_so_far values stay unchanged. Recompute totals.
"""
from __future__ import annotations

from datetime import datetime

from .core import VerificationResult, extract_json_from_text, score_from_checks

ATTENDEES = ["jake", "priya", "sam", "chloe", "ravi", "megan", "diego", "aisha"]
KEYWORDS = ("shore", "airbnb", "trip")
SHARE = 300.0


def _me(cfg: dict) -> str:
    return cfg.get("current_user") or cfg.get("user_id") or "user1"


def _q1_per_person(cfg: dict) -> dict[str, float]:
    me = _me(cfg)
    cutoff = datetime(2026, 2, 1).date()
    paid: dict[str, float] = {a: 0.0 for a in ATTENDEES}
    for t in cfg["transactions"]:
        if t.get("receiver") != me:
            continue
        s = t.get("sender")
        if s not in paid:
            continue
        try:
            d = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if d < cutoff:
            continue
        note = (t.get("note") or "").lower()
        if not any(k in note for k in KEYWORDS):
            continue
        paid[s] += float(t["amount"])
    return {a: round(v, 2) for a, v in paid.items()}


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["VenmoAPI"]
    paid = _q1_per_person(cfg)
    still_owes_q1 = {a: round(SHARE - p, 2) for a, p in paid.items()}
    # Q2 reassignment: Megan absorbs Sam.
    new_still = dict(still_owes_q1)
    sam_old = new_still["sam"]
    new_still["megan"] = round(new_still["megan"] + sam_old, 2)
    new_still["sam"] = 0.0

    per_person = []
    for a in ATTENDEES:
        per_person.append({
            "name": a,
            "paid_so_far": paid[a],
            "still_owes": new_still[a],
        })
    per_person.sort(key=lambda x: (-x["still_owes"], x["name"]))
    total_collected = round(sum(paid.values()), 2)
    total_still_owed = round(sum(new_still.values()), 2)
    return {
        "per_person": per_person,
        "total_collected": total_collected,
        "total_still_owed": total_still_owed,
    }


def _approx(a, b, tol=0.05) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


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

    pp = answer.get("per_person") or []
    exp_pp = expected["per_person"]

    checks: dict[str, bool] = {
        "per_person_count": len(pp) == 8,
        "name_set": (
            {x.get("name") for x in pp if isinstance(x, dict)} == set(ATTENDEES)
        ),
        "name_order": (
            [x.get("name") for x in pp if isinstance(x, dict)]
            == [e["name"] for e in exp_pp]
        ),
        "total_collected_close": _approx(
            answer.get("total_collected"), expected["total_collected"], tol=0.10),
        "total_still_owed_close": _approx(
            answer.get("total_still_owed"), expected["total_still_owed"], tol=0.10),
    }

    exp_by_name = {e["name"]: e for e in exp_pp}
    paid_ok = True
    owes_ok = True
    for x in pp:
        if not isinstance(x, dict):
            paid_ok = owes_ok = False; break
        e = exp_by_name.get(x.get("name"))
        if e is None:
            continue
        if not _approx(x.get("paid_so_far"), e["paid_so_far"], tol=0.05):
            paid_ok = False
        if not _approx(x.get("still_owes"), e["still_owes"], tol=0.05):
            owes_ok = False
    checks["paid_so_far_match"] = paid_ok
    checks["still_owes_match"] = owes_ok
    sam_ans = next((x for x in pp if isinstance(x, dict) and x.get("name") == "sam"), None)
    megan_ans = next((x for x in pp if isinstance(x, dict) and x.get("name") == "megan"), None)
    checks["sam_zeroed"] = bool(sam_ans and _approx(sam_ans.get("still_owes"), 0.0, tol=0.05))
    checks["megan_absorbed_sam"] = bool(
        megan_ans and _approx(megan_ans.get("still_owes"), exp_by_name["megan"]["still_owes"], tol=0.05)
    )

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
