"""Verifier for ptc_mt_venmo_core_financial_circle_0.

Q1: per quarter (Q1..Q4 of 2025) rank friends by total_flow = sent_to+received_from,
    take top-5; tiebreak inside a quarter: |net| desc then name asc, where
    net = received_from - sent_to. core_friends = top-5 in all 4 quarters.
Q2: for each strict-core friend, find the worst quarter (lowest net for me)
    across all 4 quarters of 2025, plus annual_net summed across all 4.
    Quarter tiebreak: Q1 < Q2 < Q3 < Q4.
"""
from __future__ import annotations

from datetime import date, datetime

from .core import VerificationResult, extract_json_from_text, score_from_checks

QUARTERS = {
    "Q1": (date(2025, 1, 1), date(2025, 3, 31)),
    "Q2": (date(2025, 4, 1), date(2025, 6, 30)),
    "Q3": (date(2025, 7, 1), date(2025, 9, 30)),
    "Q4": (date(2025, 10, 1), date(2025, 12, 31)),
}
QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]


def _me(cfg: dict) -> str:
    return cfg.get("current_user") or cfg.get("user_id") or "user1"


def _per_friend_quarter(cfg: dict) -> dict[str, dict[str, dict[str, float]]]:
    """{friend: {quarter: {sent_to, received_from}}} across all friends."""
    me = _me(cfg)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for t in cfg["transactions"]:
        try:
            d = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        q = None
        for qname, (lo, hi) in QUARTERS.items():
            if lo <= d <= hi:
                q = qname; break
        if q is None:
            continue
        amt = float(t["amount"])
        s, r = t["sender"], t["receiver"]
        if s == me and r != me:
            fr = r
            sent = amt; recv = 0.0
        elif r == me and s != me:
            fr = s
            sent = 0.0; recv = amt
        else:
            continue
        out.setdefault(fr, {q: {"sent_to": 0.0, "received_from": 0.0} for q in QUARTERS})
        out[fr][q]["sent_to"] += sent
        out[fr][q]["received_from"] += recv
    return out


def _strict_core(per_fq: dict) -> list[str]:
    """Top-5 per quarter (ranking by total_flow desc, |net| desc, name asc)."""
    by_q: dict[str, list[dict]] = {q: [] for q in QUARTERS}
    for fr, qmap in per_fq.items():
        for q, v in qmap.items():
            total_flow = v["sent_to"] + v["received_from"]
            net = v["received_from"] - v["sent_to"]
            if total_flow > 0:
                by_q[q].append({
                    "friend": fr,
                    "total_flow": round(total_flow, 2),
                    "net": round(net, 2),
                })
    in_q: dict[str, set[str]] = {}
    for q, entries in by_q.items():
        entries.sort(key=lambda x: (-x["total_flow"], -abs(x["net"]), x["friend"]))
        in_q[q] = {e["friend"] for e in entries[:5]}
    if not all(in_q.values()):
        return []
    return sorted(set.intersection(*in_q.values()))


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["VenmoAPI"]
    per_fq = _per_friend_quarter(cfg)
    core = _strict_core(per_fq)

    summaries = []
    for fr in core:
        nets_by_q = {}
        for q in QUARTER_ORDER:
            v = per_fq[fr][q]
            nets_by_q[q] = round(v["received_from"] - v["sent_to"], 2)
        # Worst quarter = min net; tiebreak by quarter order (Q1 < Q2 < Q3 < Q4).
        worst_q = min(QUARTER_ORDER, key=lambda q: (nets_by_q[q], QUARTER_ORDER.index(q)))
        annual_net = round(sum(nets_by_q.values()), 2)
        summaries.append({
            "friend": fr,
            "worst_quarter": worst_q,
            "worst_quarter_net": nets_by_q[worst_q],
            "annual_net": annual_net,
        })
    summaries.sort(key=lambda x: (x["annual_net"], x["friend"]))
    return {"core_summaries": summaries}


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

    cs = answer.get("core_summaries") or []
    exp_cs = expected["core_summaries"]

    checks: dict[str, bool] = {
        "core_summaries_count": len(cs) == len(exp_cs),
    }

    ans_friends = [e.get("friend") for e in cs if isinstance(e, dict)]
    exp_friends = [e["friend"] for e in exp_cs]
    checks["friend_set"] = set(ans_friends) == set(exp_friends)
    checks["friend_order"] = ans_friends == exp_friends

    exp_by_friend = {e["friend"]: e for e in exp_cs}
    worst_ok = True
    worst_net_ok = True
    annual_ok = True
    for e in cs:
        if not isinstance(e, dict):
            worst_ok = worst_net_ok = annual_ok = False; break
        ee = exp_by_friend.get(e.get("friend"))
        if ee is None:
            continue
        if e.get("worst_quarter") != ee["worst_quarter"]:
            worst_ok = False
        if not _approx(e.get("worst_quarter_net"), ee["worst_quarter_net"], tol=0.10):
            worst_net_ok = False
        if not _approx(e.get("annual_net"), ee["annual_net"], tol=0.10):
            annual_ok = False
    checks["worst_quarter_match"] = worst_ok
    checks["worst_quarter_net_match"] = worst_net_ok
    checks["annual_net_match"] = annual_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
