"""Verifier for ptc_yelp_top_rated_reviews_0.

Check that the model returned the correct top-3 restaurants (by rating)
with the 2 most recent reviews for each.
"""

from __future__ import annotations
from typing import Any
from ptc_eval.eval_checker.verifiers.core import (
    VerificationResult, extract_json_from_text, score_from_checks,
)


def _normalize_text(t: str) -> str:
    """Lowercase, strip, collapse whitespace for fuzzy text comparison."""
    return " ".join(t.lower().split())


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

        gt_top3 = ground_truth["top_3"]
        checks: dict[str, bool] = {}

        checks["returned_3_restaurants"] = len(answer) == 3

        for i, gt_restaurant in enumerate(gt_top3):
            prefix = f"rank_{i+1}"
            if i < len(answer):
                ans_r = answer[i]
                checks[f"{prefix}_name"] = ans_r.get("name") == gt_restaurant["name"]
                checks[f"{prefix}_rating"] = ans_r.get("rating") == gt_restaurant["rating"]

                # Check reviews. The prompt doesn't fix their order, and two
                # reviews of one business can share the same text, so match
                # each expected review to a distinct returned review on both
                # rating and text.
                ans_reviews = ans_r.get("reviews", []) or []
                gt_reviews = gt_restaurant["reviews"]
                checks[f"{prefix}_review_count"] = len(ans_reviews) == len(gt_reviews)

                unused = list(range(len(ans_reviews)))
                for j, gt_rev in enumerate(gt_reviews):
                    rev_prefix = f"{prefix}_review_{j+1}"
                    gt_text = _normalize_text(gt_rev["text"])
                    match = None
                    for k in unused:
                        ans_rev = ans_reviews[k] if isinstance(ans_reviews[k], dict) else {}
                        ans_text = _normalize_text(str(ans_rev.get("text") or ""))
                        # Allow partial match (first 50 chars) since models may truncate
                        text_ok = bool(ans_text) and (
                            ans_text == gt_text
                            or gt_text[:50] in ans_text
                            or ans_text[:50] in gt_text
                        )
                        if text_ok and ans_rev.get("rating") == gt_rev["rating"]:
                            match = k
                            break
                    if match is not None:
                        unused.remove(match)
                    checks[f"{rev_prefix}_rating"] = match is not None
                    checks[f"{rev_prefix}_text"] = match is not None
            else:
                checks[f"{prefix}_name"] = False
                checks[f"{prefix}_rating"] = False

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={"extracted_answer": answer, "ground_truth": gt_top3},
        )

    except (KeyError, TypeError, IndexError, AttributeError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
