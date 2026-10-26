"""Verifier for ptc_mt_venmo_reciprocity_audit_0.

Q1: out-of-balance friends (sent_to >= 80 AND received_from / sent_to < 0.30)
    over Oct 1 2025 — Mar 31 2026.
Q2: rebalance_amount per out-of-balance friend = max(0, 0.5 * sent_to - received_from);
    total_recoupable = sum of rebalance_amounts.
"""
from __future__ import annotations

from datetime import datetime

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _me(cfg: dict) -> str:
    return cfg.get("current_user") or cfg.get("user_id") or "user1"


def _q1_out_of_balance(cfg: dict) -> list[dict]:
    me = _me(cfg)
    lo = datetime(2025, 10, 1).date()
    hi = datetime(2026, 3, 31).date()
    per_friend: dict[str, dict] = {}
    for t in cfg["transactions"]:
        try:
            d = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if d < lo or d > hi:
            continue
        amt = float(t["amount"])
        s, r = t["sender"], t["receiver"]
        if s == me and r != me:
            per_friend.setdefault(r, {"sent_to": 0.0, "received_from": 0.0})
            per_friend[r]["sent_to"] += amt
        elif r == me and s != me:
            per_friend.setdefault(s, {"sent_to": 0.0, "received_from": 0.0})
            per_friend[s]["received_from"] += amt

    out = []
    for fr, v in per_friend.items():
        sent = round(v["sent_to"], 2)
        recv = round(v["received_from"], 2)
        if sent >= 80 and (recv / sent if sent else 0) < 0.30:
            ratio = round(recv / sent, 2) if sent else 0.0
            out.append({
                "friend": fr,
                "sent_to": sent,
                "received_from": recv,
                "reciprocity_ratio": ratio,
            })
    out.sort(key=lambda x: (-x["sent_to"], x["friend"]))
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["VenmoAPI"]
    oob = _q1_out_of_balance(cfg)
    rebalances = []
    for r in oob:
        amt = max(0.0, 0.5 * r["sent_to"] - r["received_from"])
        rebalances.append({
            "friend": r["friend"],
            "sent_to": r["sent_to"],
            "received_from": r["received_from"],
            "rebalance_amount": round(amt, 2),
        })
    rebalances.sort(key=lambda x: (-x["rebalance_amount"], x["friend"]))
    total = round(sum(x["rebalance_amount"] for x in rebalances), 2)
    return {"rebalances": rebalances, "total_recoupable": total}


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

    rebalances = answer.get("rebalances") or []
    exp_rebalances = expected["rebalances"]

    checks: dict[str, bool] = {
        "rebalance_count": len(rebalances) == len(exp_rebalances),
        "total_recoupable_close": _approx(
            answer.get("total_recoupable"), expected["total_recoupable"], tol=0.10),
    }

    ans_friends = [r.get("friend") for r in rebalances if isinstance(r, dict)]
    exp_friends = [r["friend"] for r in exp_rebalances]
    checks["friend_set"] = set(ans_friends) == set(exp_friends)
    checks["friend_order"] = ans_friends == exp_friends

    exp_by_friend = {r["friend"]: r for r in exp_rebalances}
    amts_ok = True
    for r in rebalances:
        if not isinstance(r, dict):
            amts_ok = False; break
        e = exp_by_friend.get(r.get("friend"))
        if e is None:
            continue
        if not _approx(r.get("rebalance_amount"), e["rebalance_amount"], tol=0.05):
            amts_ok = False; break
    checks["rebalance_amounts_match"] = amts_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
