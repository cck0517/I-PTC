"""Verifier for ptc_yelp_chinese_near_boba_0.

Check that the model returned the correct set of (Chinese restaurant, boba shop) pairs.
Order doesn't matter; we compare sets of (chinese_id, boba_id) tuples.
"""

from __future__ import annotations
from typing import Any
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    try:
        answer = extract_json_from_text(final_answer)
        if answer is None:
            return VerificationResult(
                passed=False, score=0.0,
                errors=["Could not extract JSON from final answer"],
                details={"raw_answer": final_answer},
            )
        if isinstance(answer, dict):
            answer = [answer]

        gt_pairs = ground_truth["pairs"]
        gt_set = {
            (p["chinese_restaurant_id"], p["boba_shop_id"]) for p in gt_pairs
        }

        ans_set = set()
        for item in answer:
            cid = item.get("chinese_restaurant_id")
            bid = item.get("boba_shop_id")
            if cid and bid:
                ans_set.add((cid, bid))

        checks: dict[str, bool] = {}
        checks["pair_count"] = len(ans_set) == len(gt_set)
        checks["all_pairs_correct"] = ans_set == gt_set

        # Check individual pairs
        for p in gt_pairs:
            key = (p["chinese_restaurant_id"], p["boba_shop_id"])
            label = f"{p['chinese_restaurant_name']}+{p['boba_shop_name']}"
            checks[label] = key in ans_set

        # Check no extra pairs
        extra = ans_set - gt_set
        checks["no_extra_pairs"] = len(extra) == 0

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={
                "extracted_pairs": [list(p) for p in ans_set],
                "expected_pairs": [list(p) for p in gt_set],
            },
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
