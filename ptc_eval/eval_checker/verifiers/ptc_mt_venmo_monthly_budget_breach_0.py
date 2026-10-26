"""Verifier for ptc_mt_venmo_monthly_budget_breach_0.

Q1: per-(friend, month) outgoing total over Jan/Feb/Mar 2026; flag
    breaches where monthly total > $50.
Q2: for every friend who breached at least once: q1_total = sum across
    Jan+Feb+Mar 2026 outgoing; suggested_cap = ceil(max monthly total
    across the three months).
"""
from __future__ import annotations

import math
from datetime import datetime

from .core import VerificationResult, extract_json_from_text, score_from_checks

MONTHS = ["2026-01", "2026-02", "2026-03"]
CAP = 50.0


def _me(cfg: dict) -> str:
    return cfg.get("current_user") or cfg.get("user_id") or "user1"


def _per_friend_monthly(cfg: dict) -> dict[str, dict[str, float]]:
    me = _me(cfg)
    out: dict[str, dict[str, float]] = {}
    for t in cfg["transactions"]:
        if t.get("sender") != me or t.get("receiver") == me:
            continue
        try:
            d = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        ym = f"{d.year:04d}-{d.month:02d}"
        if ym not in MONTHS:
            continue
        fr = t["receiver"]
        out.setdefault(fr, {m: 0.0 for m in MONTHS})
        out[fr][ym] += float(t["amount"])
    # Round per (friend,month) to avoid float drift.
    for fr in out:
        for m in MONTHS:
            out[fr][m] = round(out[fr][m], 2)
    return out


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["VenmoAPI"]
    per = _per_friend_monthly(cfg)

    breached_friends = [
        fr for fr, by_m in per.items()
        if any(v > CAP for v in by_m.values())
    ]
    adjusted_caps = []
    for fr in breached_friends:
        monthly = per[fr]
        q1_total = round(sum(monthly.values()), 2)
        suggested = int(math.ceil(max(monthly.values())))
        adjusted_caps.append({
            "friend": fr,
            "q1_total": q1_total,
            "suggested_cap": suggested,
        })
    adjusted_caps.sort(key=lambda x: (-x["suggested_cap"], x["friend"]))
    return {"adjusted_caps": adjusted_caps}


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

    caps = answer.get("adjusted_caps") or []
    exp_caps = expected["adjusted_caps"]

    checks: dict[str, bool] = {
        "caps_count": len(caps) == len(exp_caps),
    }
    ans_friends = [c.get("friend") for c in caps if isinstance(c, dict)]
    exp_friends = [c["friend"] for c in exp_caps]
    checks["friend_set"] = set(ans_friends) == set(exp_friends)
    checks["friend_order"] = ans_friends == exp_friends

    exp_by_friend = {c["friend"]: c for c in exp_caps}
    cap_ok = True
    total_ok = True
    for c in caps:
        if not isinstance(c, dict):
            cap_ok = total_ok = False; break
        e = exp_by_friend.get(c.get("friend"))
        if e is None:
            continue
        # suggested_cap is an integer; tolerate int vs float.
        try:
            ans_cap = int(c.get("suggested_cap"))
        except (TypeError, ValueError):
            ans_cap = None
        if ans_cap != e["suggested_cap"]:
            cap_ok = False
        if not _approx(c.get("q1_total"), e["q1_total"], tol=0.10):
            total_ok = False
    checks["suggested_cap_match"] = cap_ok
    checks["q1_total_match"] = total_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
