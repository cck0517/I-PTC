"""Verifier for ptc_intermediate_venmo_debt_settle_0.

Builds a net-balance graph from the raw transaction history and computes
the minimum-cardinality settlement plan (greedy largest-creditor-vs-
largest-debtor), which is optimal for the metric "number of transfers".
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "VenmoAPIAlt", initial_config.get("VenmoAPI", {}),
    )


def _compute_net(cfg: dict) -> dict:
    net: dict[str, float] = {}
    for t in cfg.get("transactions", []):
        if t.get("type") != "payment":
            continue
        amt = float(t["amount"])
        s, r = t["sender"], t["receiver"]
        net[s] = net.get(s, 0.0) - amt
        net[r] = net.get(r, 0.0) + amt
    return {k: round(v, 2) for k, v in net.items()}


def _greedy_settle(net: dict) -> list:
    creditors = sorted([(u, v) for u, v in net.items() if v > 0.005],
                       key=lambda kv: (-kv[1], kv[0]))
    debtors = sorted([(u, -v) for u, v in net.items() if v < -0.005],
                     key=lambda kv: (-kv[1], kv[0]))
    c_list = [list(x) for x in creditors]
    d_list = [list(x) for x in debtors]
    plan: list[dict] = []
    i = j = 0
    while i < len(c_list) and j < len(d_list):
        cred_name, cred_amt = c_list[i]
        debt_name, debt_amt = d_list[j]
        pay = round(min(cred_amt, debt_amt), 2)
        if pay <= 0.005:
            break
        plan.append({"from": debt_name, "to": cred_name, "amount": pay})
        c_list[i][1] = round(cred_amt - pay, 2)
        d_list[j][1] = round(debt_amt - pay, 2)
        if c_list[i][1] < 0.005:
            i += 1
        if d_list[j][1] < 0.005:
            j += 1
    plan.sort(key=lambda p: (-p["amount"], p["from"], p["to"]))
    return plan


def _compute_ground_truth(cfg: dict) -> dict:
    net = _compute_net(cfg)
    plan = _greedy_settle(net)
    return {
        "num_settlements": len(plan),
        "settlements": plan,
        "net_by_user": {k: net[k] for k in sorted(net)},
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected"] = expected
    try:
        checks["num_settlements_correct"] = int(answer.get("num_settlements")) == expected["num_settlements"]
    except (TypeError, ValueError):
        checks["num_settlements_correct"] = False
    # Net balance dict
    got_net = answer.get("net_by_user") or {}
    if not isinstance(got_net, dict):
        got_net = {}
    net_ok = set(got_net) == set(expected["net_by_user"])
    for k, v in expected["net_by_user"].items():
        try:
            if abs(float(got_net.get(k)) - v) > 0.05:
                net_ok = False
        except (TypeError, ValueError):
            net_ok = False
    checks["net_by_user_correct"] = net_ok
    # Settlements — each payment from a debtor to a creditor with the expected
    # amount total.  We don't require the greedy ordering the verifier chose,
    # only that the plan actually zeros out net balances and uses
    # ``num_settlements`` transfers.
    got_plan = answer.get("settlements") or []
    if not isinstance(got_plan, list):
        got_plan = []
    apply_ok = len(got_plan) == expected["num_settlements"]
    sim = dict(expected["net_by_user"])
    for p in got_plan:
        if not isinstance(p, dict):
            apply_ok = False
            continue
        try:
            a = float(p.get("amount"))
            fr = p.get("from")
            to = p.get("to")
            sim[fr] = round(sim.get(fr, 0.0) + a, 2)
            sim[to] = round(sim.get(to, 0.0) - a, 2)
        except (TypeError, ValueError):
            apply_ok = False
    if apply_ok:
        apply_ok = all(abs(v) < 0.05 for v in sim.values())
    checks["settlements_apply_correct"] = apply_ok
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
