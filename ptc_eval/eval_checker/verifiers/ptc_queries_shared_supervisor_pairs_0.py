"""Verifier for ptc_queries_shared_supervisor_pairs_0.

Check that all employee pairs sharing a supervisor are found, with correct
supervisor info and concurrent-service determination. Pairs are order-independent.
"""

from __future__ import annotations
from typing import Any
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)


def _pair_key(pair: dict) -> tuple:
    eids = pair.get("employee_ids", [])
    if not eids:
        eids = [pair.get("employee_id_1", ""), pair.get("employee_id_2", "")]
    return tuple(sorted(str(e) for e in eids))


def _supervisor(pair: dict) -> str:
    return str(pair.get("supervisor_id", pair.get("shared_supervisor_id", "")))


def _concurrent(pair: dict) -> bool | None:
    v = pair.get("concurrent_service", pair.get("concurrent", None))
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes")
    return None


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    gt_pairs = ground_truth.get("pairs", [])
    answer = extract_json_from_text(final_answer)
    if answer is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=["Could not extract JSON from final answer"],
            details={"raw_answer": final_answer},
        )

    if isinstance(answer, dict):
        answer_pairs = answer.get("pairs", answer.get("results", []))
        if not isinstance(answer_pairs, list):
            answer_pairs = [answer]
    elif isinstance(answer, list):
        answer_pairs = answer
    else:
        answer_pairs = []

    answer_index: dict[tuple, dict] = {}
    for p in answer_pairs:
        if isinstance(p, dict):
            answer_index[_pair_key(p)] = p

    checks: dict[str, bool] = {}
    errors: list[str] = []

    for gt_pair in gt_pairs:
        key = _pair_key(gt_pair)
        label = f"pair_{'-'.join(key)}"
        matched = key in answer_index
        checks[f"{label}_found"] = matched
        if not matched:
            errors.append(f"Missing pair {key}")
            continue
        ap = answer_index[key]
        gt_sup = _supervisor(gt_pair)
        checks[f"{label}_supervisor"] = _supervisor(ap) == gt_sup
        if _supervisor(ap) != gt_sup:
            errors.append(f"Pair {key}: wrong supervisor (got {_supervisor(ap)}, expected {gt_sup})")
        gt_conc = gt_pair.get("concurrent_service")
        a_conc = _concurrent(ap)
        if gt_conc is not None and a_conc is not None:
            checks[f"{label}_concurrent"] = a_conc == gt_conc
            if a_conc != gt_conc:
                errors.append(f"Pair {key}: wrong concurrent_service (got {a_conc}, expected {gt_conc})")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"gt_pair_count": len(gt_pairs), "answer_pair_count": len(answer_pairs)},
        errors=errors,
    )
