"""Verifier for ptc_mt_grocery_single_vs_split_0 (multi-turn).

Q1 (parity, 3-store variant): cheapest single-store option vs best 3-store
split.  Q2: top-3 best 3-store splits ranked by combined_total ascending.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from ._grocery_helpers import shopping_list, store_cost, stores
from .core import VerificationResult, extract_json_from_text, score_from_checks


def _greedy_3split(s_a: dict, s_b: dict, s_c: dict, sids: tuple[str, str, str],
                   shopping: list[str]) -> dict | None:
    """Assign each item to whichever of {a,b,c} has it cheapest in stock.
    Returns per-store breakdown + combined_total, or None if any item is
    uncovered or any store gets zero items (degenerate)."""
    stores_in_split = [(sids[0], s_a), (sids[1], s_b), (sids[2], s_c)]
    items_by_sid: dict[str, list[str]] = {sid: [] for sid, _ in stores_in_split}
    for item in shopping:
        candidates = []
        for sid, s in stores_in_split:
            inv = s.get("inventory", {})
            if item in inv and inv[item].get("in_stock", True):
                candidates.append((inv[item]["price"], sid))
        if not candidates:
            return None
        candidates.sort()
        items_by_sid[candidates[0][1]].append(item)
    if any(len(v) == 0 for v in items_by_sid.values()):
        return None
    out_stores = []
    combined = 0.0
    for sid, s in stores_in_split:
        c = store_cost(s, items_by_sid[sid])
        if c is None:
            return None
        out_stores.append({
            "store_id": sid,
            "store_name": s.get("name", sid),
            **c,
        })
        combined += c["subtotal"]
    return {"stores": out_stores, "combined_total": round(combined, 2)}


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["GroceryDeliveryAPI"]
    shopping = shopping_list(cfg)
    sts = stores(cfg)
    sids = sorted(sts.keys())

    candidates = []
    for combo in combinations(sids, 3):
        sid_a, sid_b, sid_c = combo
        result = _greedy_3split(sts[sid_a], sts[sid_b], sts[sid_c], combo, shopping)
        if result is None:
            continue
        # Map the three sorted slots to store_a (smallest sid) ... store_c (largest)
        s_by_sid = {s["store_id"]: s for s in result["stores"]}
        candidates.append({
            "store_a": s_by_sid[combo[0]],
            "store_b": s_by_sid[combo[1]],
            "store_c": s_by_sid[combo[2]],
            "combined_total": result["combined_total"],
        })

    candidates.sort(key=lambda r: (
        r["combined_total"],
        r["store_a"]["store_id"],
        r["store_b"]["store_id"],
        r["store_c"]["store_id"],
    ))
    return {"top3_splits": candidates[:3]}


def _store_entry_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    if got.get("store_id") != exp["store_id"]:
        return False
    if got.get("store_name") != exp["store_name"]:
        return False
    if list(got.get("items") or []) != list(exp["items"]):
        return False
    for k in ("item_total", "service_fee", "delivery_fee", "subtotal"):
        gv = got.get(k)
        if not isinstance(gv, (int, float)) or abs(gv - exp[k]) > 0.01:
            return False
    return True


def _split_match(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for slot in ("store_a", "store_b", "store_c"):
        if not _store_entry_match(got.get(slot), exp[slot]):
            return False
    gt = got.get("combined_total")
    if not isinstance(gt, (int, float)) or abs(gt - exp["combined_total"]) > 0.02:
        return False
    return True


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
    got = answer.get("top3_splits")
    exp = expected["top3_splits"]
    checks = {"top3_count_correct": isinstance(got, list) and len(got) == len(exp)}
    if not checks["top3_count_correct"]:
        got = []
    entries_ok = checks["top3_count_correct"]
    if entries_ok:
        for g, e in zip(got, exp):
            if not _split_match(g, e):
                entries_ok = False; break
    checks["top3_entries_correct"] = entries_ok
    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected_summary": [
            {"a": s["store_a"]["store_id"], "b": s["store_b"]["store_id"],
             "c": s["store_c"]["store_id"], "total": s["combined_total"]}
            for s in expected["top3_splits"]
        ], "answer": answer},
    )
