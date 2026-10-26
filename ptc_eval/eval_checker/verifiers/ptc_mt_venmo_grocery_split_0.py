"""Verifier for ptc_mt_venmo_grocery_split_0.

Q2: Only split with friends whose venmo_net_balance >= 0. N = (qualified + 1).
"""
from __future__ import annotations

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _cheapest_grocery(config: dict) -> tuple[str, str, float]:
    cfg = config["GroceryDeliveryAPI"]
    sl = cfg["shopping_list"]
    best = None
    for sid, store in cfg["stores"].items():
        inv = store["inventory"]
        all_ok = True
        item_total = 0.0
        for it in sl:
            row = inv.get(it)
            if not row or not row.get("in_stock"):
                all_ok = False; break
            item_total += float(row["price"])
        if not all_ok:
            continue
        tot = round(item_total + float(store["delivery_fee"]), 2)
        if best is None or tot < best[2]:
            best = (sid, store["name"], tot)
    return best  # may be None if no store has all items


def _friend_nets(config: dict) -> dict[str, float]:
    cfg = config["VenmoAPI"]
    me = cfg.get("current_user") or cfg.get("user_id") or "user1"
    nets: dict[str, float] = {}
    for fr in cfg.get("friends", []):
        nets[fr] = 0.0
    for t in cfg["transactions"]:
        s, r, amt = t["sender"], t["receiver"], float(t["amount"])
        if s == me and r in nets:
            nets[r] -= amt
        elif r == me and s in nets:
            nets[s] += amt
    return {k: round(v, 2) for k, v in nets.items()}


def _compute_expected(config: dict) -> dict:
    sid, sname, grocery_total = _cheapest_grocery(config)
    nets = _friend_nets(config)
    qualified = [(fr, net) for fr, net in nets.items() if net >= 0]
    qualified.sort(key=lambda x: (-x[1], x[0]))
    N = len(qualified) + 1
    share = round(grocery_total / N, 2) if N else grocery_total
    return {
        "grocery_total": grocery_total,
        "cheapest_store_id": sid,
        "cheapest_store_name": sname,
        "qualified_count": len(qualified),
        "N": N,
        "my_share": share,
        "qualified_friends": [
            {"friend_username": fr, "venmo_net_balance": net, "share": share}
            for fr, net in qualified
        ],
    }


def _approx_eq(a, b, tol=0.05) -> bool:
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
            errors=["Expected JSON object"],
            details={"raw_answer": final_answer},
        )
    expected = _compute_expected(initial_config)
    q_ans = answer.get("qualified_friends") or []
    ans_names = [q.get("friend_username") for q in q_ans if isinstance(q, dict)]
    exp_names = [q["friend_username"] for q in expected["qualified_friends"]]
    checks = {
        "grocery_total_close":  _approx_eq(answer.get("grocery_total"), expected["grocery_total"], tol=0.1),
        "qualified_count":      answer.get("qualified_count") == expected["qualified_count"],
        "N_correct":            answer.get("N") == expected["N"],
        "my_share_close":       _approx_eq(answer.get("my_share"), expected["my_share"], tol=0.05),
        "qualified_set":        set(ans_names) == set(exp_names),
    }
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
