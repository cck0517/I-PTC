"""Verifier for ptc_mt_venmo_debt_ranking_0.

Q1: Per friend, net_balance = money_they_sent_me - money_I_sent_them.
Q2: From those, keep only friends with net_balance < 0 ("I owe them"),
    rank by |net_balance| desc.
"""
from __future__ import annotations

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _compute_expected(config: dict) -> dict:
    cfg = config["VenmoAPI"]
    me = cfg.get("current_user") or cfg.get("user_id") or "user1"
    per_friend: dict[str, dict] = {}
    last_tx_by_friend: dict[str, str] = {}
    for t in cfg["transactions"]:
        s, r, amt, d = t["sender"], t["receiver"], float(t["amount"]), t.get("date", "")
        if s == me and r != me:
            fr = r
            per_friend.setdefault(fr, {"sent_to_me": 0.0, "sent_by_me": 0.0})
            per_friend[fr]["sent_by_me"] += amt
        elif r == me and s != me:
            fr = s
            per_friend.setdefault(fr, {"sent_to_me": 0.0, "sent_by_me": 0.0})
            per_friend[fr]["sent_to_me"] += amt
        else:
            continue
        if d > last_tx_by_friend.get(fr, ""):
            last_tx_by_friend[fr] = d

    creditors = []
    for fr, v in per_friend.items():
        net = round(v["sent_to_me"] - v["sent_by_me"], 2)
        if net < 0:
            creditors.append({
                "friend_username": fr,
                "amount_owed": round(-net, 2),
                "last_tx_date": last_tx_by_friend.get(fr, ""),
            })
    creditors.sort(key=lambda c: (-c["amount_owed"], c["friend_username"]))
    total_debt = round(sum(c["amount_owed"] for c in creditors), 2)
    return {"total_debt": total_debt, "creditors": creditors}


def _approx_eq(a, b, tol=0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


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
        "total_debt_close": _approx_eq(answer.get("total_debt"), expected["total_debt"], tol=0.05),
    }
    creditors_ans = answer.get("creditors") or []
    ans_names = [c.get("friend_username") for c in creditors_ans if isinstance(c, dict)]
    exp_names = [c["friend_username"] for c in expected["creditors"]]
    checks["creditor_count"] = len(creditors_ans) == len(expected["creditors"])
    checks["creditor_set"] = set(ans_names) == set(exp_names)
    checks["creditor_order"] = ans_names == exp_names

    exp_by_name = {c["friend_username"]: c for c in expected["creditors"]}
    amt_ok = True
    for c in creditors_ans:
        if not isinstance(c, dict):
            amt_ok = False; break
        e = exp_by_name.get(c.get("friend_username"))
        if e is None:
            continue
        if not _approx_eq(c.get("amount_owed"), e["amount_owed"], tol=0.05):
            amt_ok = False; break
    checks["amounts_match"] = amt_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
