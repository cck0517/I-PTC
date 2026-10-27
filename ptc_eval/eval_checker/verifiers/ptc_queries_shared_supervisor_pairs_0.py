"""Verifier for ptc_queries_shared_supervisor_pairs_0.

Check that all employee pairs sharing a supervisor are found, with correct
supervisor info and concurrent-service determination. Pairs and entries are
order-independent; a pair that shared several supervisors may be listed once
per supervisor or with any one of them.
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

    # Some pairs shared more than one supervisor. Index both the expected and
    # the answered entries by (pair, supervisor) so the result does not depend
    # on output order. The prompt says "for each pair, return who that
    # supervisor was", so a pair counts as found if it is listed with at least
    # one supervisor it really shared; every supervisor listed for a pair must
    # be a real shared supervisor, with the right concurrency.
    gt_by_pair: dict[tuple, dict[str, dict]] = {}
    for gt_pair in gt_pairs:
        gt_by_pair.setdefault(_pair_key(gt_pair), {})[_supervisor(gt_pair)] = gt_pair

    answer_by_pair: dict[tuple, dict[str, dict]] = {}
    for p in answer_pairs:
        if isinstance(p, dict):
            answer_by_pair.setdefault(_pair_key(p), {})[_supervisor(p)] = p

    checks: dict[str, bool] = {}
    errors: list[str] = []

    for key, gt_sups in gt_by_pair.items():
        label = f"pair_{'-'.join(key)}"
        listed = answer_by_pair.get(key, {})
        checks[f"{label}_found"] = bool(listed)
        if not listed:
            errors.append(f"Missing pair {key}")
            continue
        for sup, ap in listed.items():
            sup_label = f"{label}_{sup}"
            real = sup in gt_sups
            checks[f"{sup_label}_supervisor"] = real
            if not real:
                errors.append(f"Pair {key}: {sup} is not a supervisor they shared "
                              f"(shared: {sorted(gt_sups)})")
                continue
            gt_conc = gt_sups[sup].get("concurrent_service")
            a_conc = _concurrent(ap)
            if gt_conc is not None and a_conc is not None:
                checks[f"{sup_label}_concurrent"] = a_conc == gt_conc
                if a_conc != gt_conc:
                    errors.append(f"Pair {key} under {sup}: wrong concurrent_service "
                                  f"(got {a_conc}, expected {gt_conc})")

    passed, score = score_from_checks(checks)
    return VerificationResult(
        passed=passed, score=score, checks=checks,
        details={"gt_pair_count": len(gt_pairs), "answer_pair_count": len(answer_pairs)},
        errors=errors,
    )
