"""Verifier for ptc_mt_venmo_subscription_hunt_0.

Q1: find recurring monthly outgoing groups (>=3 charges, gap 25-35 days).
Q2: top-3 by total_paid (tiebreak recipient asc); per entry compute
    avg_monthly = total_paid / charge_count and projected_annual = avg_monthly * 12.
"""
from __future__ import annotations

from datetime import datetime

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _me(cfg: dict) -> str:
    return cfg.get("current_user") or cfg.get("user_id") or "user1"


def _q1_subscriptions(cfg: dict) -> list[dict]:
    me = _me(cfg)
    start = datetime(2025, 7, 1).date()
    end = datetime(2026, 3, 31).date()
    by_recipient_amount: dict[tuple, list[tuple]] = {}
    for t in cfg["transactions"]:
        if t.get("sender") != me or t.get("receiver") == me:
            continue
        try:
            d = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if d < start or d > end:
            continue
        amt = round(float(t["amount"]), 2)
        by_recipient_amount.setdefault((t["receiver"], amt), []).append(d)

    subs = []
    for (recipient, amt), dates in by_recipient_amount.items():
        dates_sorted = sorted(dates)
        # Walk dates and split into runs where consecutive gaps fall in [25,35].
        run = [dates_sorted[0]]
        runs = []
        for prev, cur in zip(dates_sorted, dates_sorted[1:]):
            gap = (cur - prev).days
            if 25 <= gap <= 35:
                run.append(cur)
            else:
                runs.append(run)
                run = [cur]
        runs.append(run)
        for r in runs:
            if len(r) >= 3:
                subs.append({
                    "recipient": recipient,
                    "amount": amt,
                    "charge_count": len(r),
                    "total_paid": round(amt * len(r), 2),
                    "first_date": r[0].strftime("%Y-%m-%d"),
                    "last_date": r[-1].strftime("%Y-%m-%d"),
                })
    subs.sort(key=lambda s: (-s["total_paid"], s["recipient"]))
    return subs


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["VenmoAPI"]
    subs = _q1_subscriptions(cfg)
    top3 = subs[:3]
    cancelled = []
    for s in top3:
        avg_monthly = round(s["total_paid"] / s["charge_count"], 2)
        cancelled.append({
            "recipient": s["recipient"],
            "amount": s["amount"],
            "avg_monthly": avg_monthly,
            "projected_annual": round(avg_monthly * 12, 2),
        })
    total_annual_savings = round(sum(c["projected_annual"] for c in cancelled), 2)
    return {"cancelled": cancelled, "total_annual_savings": total_annual_savings}


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

    cancelled = answer.get("cancelled") or []
    exp_cancelled = expected["cancelled"]

    checks: dict[str, bool] = {
        "cancelled_count": len(cancelled) == len(exp_cancelled),
        "total_annual_savings_close": _approx(
            answer.get("total_annual_savings"), expected["total_annual_savings"], tol=0.10),
    }

    ans_recipients = [c.get("recipient") for c in cancelled if isinstance(c, dict)]
    exp_recipients = [c["recipient"] for c in exp_cancelled]
    checks["recipient_set"] = set(ans_recipients) == set(exp_recipients)
    checks["recipient_order"] = ans_recipients == exp_recipients

    exp_by_recipient = {c["recipient"]: c for c in exp_cancelled}
    fields_ok = True
    for c in cancelled:
        if not isinstance(c, dict):
            fields_ok = False; break
        e = exp_by_recipient.get(c.get("recipient"))
        if e is None:
            continue
        if not (_approx(c.get("avg_monthly"), e["avg_monthly"], tol=0.05)
                and _approx(c.get("projected_annual"), e["projected_annual"], tol=0.10)
                and _approx(c.get("amount"), e["amount"], tol=0.01)):
            fields_ok = False; break
    checks["entry_fields_match"] = fields_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
