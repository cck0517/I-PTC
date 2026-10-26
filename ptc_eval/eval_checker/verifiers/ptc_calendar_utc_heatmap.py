"""Verifier for ptc_calendar_utc_heatmap_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


def _peak_matches(got: Any, exp: dict) -> bool:
    if not isinstance(got, dict):
        return False
    return got.get("utc_hour") == exp["utc_hour"] and got.get("meeting_count") == exp["meeting_count"]


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    details["expected"] = ground_truth

    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])

    hist = answer.get("histogram")
    exp_hist = ground_truth["histogram"]
    checks["has_histogram_key"] = isinstance(hist, dict)
    if isinstance(hist, dict):
        # Accept either string or int keys; normalize to string keys for comparison.
        norm = {str(k): v for k, v in hist.items()}
        checks["histogram_correct"] = norm == exp_hist
    else:
        checks["histogram_correct"] = False

    peaks = answer.get("top3_peak_hours")
    exp_peaks = ground_truth["top3_peak_hours"]
    checks["has_top3_peak_hours_key"] = isinstance(peaks, list)
    if not isinstance(peaks, list):
        peaks = []
    checks["top3_count_correct"] = len(peaks) == len(exp_peaks)
    entries_ok = checks["top3_count_correct"]
    if entries_ok:
        for g, e in zip(peaks, exp_peaks):
            if not _peak_matches(g, e):
                entries_ok = False; break
    checks["top3_entries_correct"] = entries_ok

    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
