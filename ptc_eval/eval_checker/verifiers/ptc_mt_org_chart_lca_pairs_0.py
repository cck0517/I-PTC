"""Verifier for ptc_mt_org_chart_lca_pairs_0.

Q1: LCA for 5 fixed pairs on 2026-01-15.
Q2: pair_distance = (chain_a_len - lca_depth) + (chain_b_len - lca_depth)
    ranking, descending; tiebreak: pair lex asc.
"""
from __future__ import annotations

from datetime import date

from ._orgchart_helpers import (
    active_manager_on, chain_to_root, name_by_id,
)
from .core import VerificationResult, extract_json_from_text, score_from_checks


SNAPSHOT = date(2026, 1, 15)
PAIRS = [("e50", "e52"), ("e55", "e56"), ("e57", "e58"), ("e50", "e59"), ("e51", "e54")]


def _compute_expected(initial_config: dict) -> dict:
    cfg = initial_config["OrgChartAPI"]
    nbi = name_by_id(cfg)
    am = active_manager_on(cfg["reporting_history"], SNAPSHOT)

    out = []
    for a, b in PAIRS:
        ca = chain_to_root(a, am)
        cb = chain_to_root(b, am)
        i = 0
        while i < min(len(ca), len(cb)) and ca[i] == cb[i]:
            i += 1
        lca_idx = i - 1
        if lca_idx < 0:
            # No common ancestor (shouldn't happen if both have an active row).
            continue
        lca_id = ca[lca_idx]
        chain_a_len = len(ca) - 1
        chain_b_len = len(cb) - 1
        pair_distance = (chain_a_len - lca_idx) + (chain_b_len - lca_idx)
        out.append({
            "pair": [a, b],
            "pair_distance": pair_distance,
            "lca_manager_name": nbi.get(lca_id, ""),
        })
    out.sort(key=lambda x: (-x["pair_distance"], x["pair"][0], x["pair"][1]))
    return {"ranked_pairs": out}


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

    rp_ans = answer.get("ranked_pairs") or []
    rp_exp = expected["ranked_pairs"]
    checks: dict[str, bool] = {
        "ranked_count": len(rp_ans) == len(rp_exp),
    }
    ans_pairs = [tuple(r.get("pair") or []) for r in rp_ans if isinstance(r, dict)]
    exp_pairs = [tuple(r["pair"]) for r in rp_exp]
    checks["pair_set"] = set(ans_pairs) == set(exp_pairs)
    checks["pair_order"] = ans_pairs == exp_pairs

    exp_by_pair = {tuple(r["pair"]): r for r in rp_exp}
    dist_ok = True
    name_ok = True
    for r in rp_ans:
        if not isinstance(r, dict):
            dist_ok = name_ok = False; break
        e = exp_by_pair.get(tuple(r.get("pair") or []))
        if e is None:
            continue
        if r.get("pair_distance") != e["pair_distance"]:
            dist_ok = False
        if r.get("lca_manager_name") != e["lca_manager_name"]:
            name_ok = False
    checks["pair_distance_match"] = dist_ok
    checks["lca_manager_name_match"] = name_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"expected": expected, "answer": answer},
    )
