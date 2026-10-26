"""Verifier for ptc_intermediate_spotify_deep_cuts_0."""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, extract_json_from_text, score_from_checks


SOURCE_PL = "pl_source"


def _cfg(initial_config: dict) -> dict:
    return initial_config.get(
        "SpotifyAPIAlt", initial_config.get("SpotifyAPI", {}),
    )


def _pop(t: dict) -> float:
    pc = int(t.get("play_count", 0) or 0)
    lr = float(t.get("listener_rating", 0) or 0)
    return round(min(pc, 500_000) / 10_000 + lr * 10, 1)


def _percentile_linear(values: list, p: float) -> float:
    """numpy.percentile's 'linear' interpolation on a sorted list."""
    if not values:
        return 0.0
    vs = sorted(values)
    k = (len(vs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return float(vs[f])
    d0 = vs[f] * (c - k)
    d1 = vs[c] * (k - f)
    return d0 + d1


def _compute_ground_truth(cfg: dict) -> dict:
    tracks = cfg["tracks"]
    tids = cfg["playlists"][SOURCE_PL]["tracks"]
    pops = {tid: _pop(tracks[tid]) for tid in tids}
    threshold = round(_percentile_linear(list(pops.values()), 20.0), 2)
    deep = []
    for tid, p in pops.items():
        if p <= threshold:
            deep.append({
                "track_id": tid,
                "artist": tracks[tid]["artist"],
                "popularity_index": p,
            })
    deep.sort(key=lambda r: (r["popularity_index"], r["track_id"]))
    return {
        "threshold": threshold,
        "deep_cuts": deep,
        "deep_cut_count": len(deep),
        "deep_cut_ids": [r["track_id"] for r in deep],
    }


def verify(final_answer, final_state, initial_config, ground_truth) -> VerificationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    answer = extract_json_from_text(final_answer)
    details["raw_answer"] = (final_answer or "")[:500]
    if not isinstance(answer, dict):
        return VerificationResult(passed=False, score=0.0, details=details,
                                  errors=["Could not extract JSON object"])
    expected = _compute_ground_truth(_cfg(initial_config))
    details["expected_ids"] = expected["deep_cut_ids"]
    details["expected_threshold"] = expected["threshold"]
    try:
        checks["threshold_correct"] = abs(float(answer.get("threshold")) - expected["threshold"]) < 0.15
    except (TypeError, ValueError):
        checks["threshold_correct"] = False
    try:
        checks["count_correct"] = int(answer.get("deep_cut_count")) == expected["deep_cut_count"]
    except (TypeError, ValueError):
        checks["count_correct"] = False
    got = answer.get("deep_cuts") or []
    if not isinstance(got, list):
        got = []
    got_ids = [str(x.get("track_id")) for x in got if isinstance(x, dict)]
    checks["deep_cuts_order_correct"] = got_ids == expected["deep_cut_ids"]
    passed, score = score_from_checks(checks)
    return VerificationResult(passed=passed, score=score, checks=checks, details=details)
