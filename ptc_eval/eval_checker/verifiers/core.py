"""
Verifier framework for PTC-Bench tasks.

Shared helpers live here. Each task has its own module under this package
with a single ``verify()`` function.  Dispatch is by ``task_id``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    passed: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "checks": self.checks,
            "details": self.details,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json_from_text(text: str | None) -> dict | list | None:
    """Best-effort extraction of a JSON object/array from model text."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        idx = text.find(open_ch)
        if idx < 0:
            continue
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return None


def extract_utc_timestamps(text: str) -> list[str]:
    """Extract ISO 8601 UTC timestamps (ending with Z) from text."""
    return re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text or "")


# ---------------------------------------------------------------------------
# Calendar state helpers
# ---------------------------------------------------------------------------

def _event_key(ev: dict) -> tuple:
    return (ev["employee_id"], ev["start"], ev["end"], ev["title"])


def extract_new_events(
    initial_config: dict, final_state: dict, api_class: str = "EnterpriseCalendarAPI"
) -> list[dict]:
    """Return events present in final_state but not in initial_config."""
    initial = initial_config.get(api_class, {}).get("events", [])
    final = final_state.get(api_class, {}).get("events", [])
    initial_keys = {_event_key(e) for e in initial}
    return [e for e in final if _event_key(e) not in initial_keys]


def group_into_bookings(
    new_events: list[dict], employees: list[dict]
) -> dict[tuple[str, str], list[dict]]:
    """Group new events into logical bookings by (utc_start, title)."""
    emp_tz = {e["id"]: e["timezone"] for e in employees}
    bookings: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ev in new_events:
        tz_name = emp_tz.get(ev["employee_id"], "UTC")
        naive = datetime.fromisoformat(ev["start"])
        utc = naive.replace(tzinfo=ZoneInfo(tz_name)).astimezone(ZoneInfo("UTC"))
        key = (utc.strftime("%Y-%m-%dT%H:%M:%SZ"), ev["title"])
        bookings[key].append(ev)
    return dict(bookings)


def times_overlap(s1: str, e1: str, s2: str, e2: str) -> bool:
    return datetime.fromisoformat(s1) < datetime.fromisoformat(e2) and \
           datetime.fromisoformat(s2) < datetime.fromisoformat(e1)


def local_to_utc(local_iso: str, tz_name: str) -> datetime:
    naive = datetime.fromisoformat(local_iso)
    return naive.replace(tzinfo=ZoneInfo(tz_name)).astimezone(ZoneInfo("UTC"))


def utc_to_local(utc_dt: datetime, tz_name: str) -> datetime:
    return utc_dt.astimezone(ZoneInfo(tz_name))


def check_no_overlap_all_employees(all_events: list[dict]) -> list[str]:
    """Return overlap error strings across all employees."""
    by_emp: dict[str, list[dict]] = defaultdict(list)
    for ev in all_events:
        by_emp[ev["employee_id"]].append(ev)
    errors = []
    for emp_id, evs in by_emp.items():
        evs_sorted = sorted(evs, key=lambda e: e["start"])
        for i in range(len(evs_sorted)):
            for j in range(i + 1, len(evs_sorted)):
                if times_overlap(
                    evs_sorted[i]["start"], evs_sorted[i]["end"],
                    evs_sorted[j]["start"], evs_sorted[j]["end"],
                ):
                    errors.append(
                        f"Overlap for {emp_id}: "
                        f"'{evs_sorted[i]['title']}' ({evs_sorted[i]['start']}–{evs_sorted[i]['end']}) "
                        f"vs '{evs_sorted[j]['title']}' ({evs_sorted[j]['start']}–{evs_sorted[j]['end']})"
                    )
    return errors


def free_blocks(
    events: list[dict], day: date, workday_start: time, workday_end: time,
) -> list[tuple[datetime, datetime]]:
    """Compute free blocks for one employee on a single day."""
    ws = datetime.combine(day, workday_start)
    we = datetime.combine(day, workday_end)
    busy = []
    for ev in events:
        s = datetime.fromisoformat(ev["start"])
        e = datetime.fromisoformat(ev["end"])
        if s < we and e > ws:
            busy.append((max(s, ws), min(e, we)))
    busy.sort()
    blocks = []
    cursor = ws
    for bs, be in busy:
        if bs > cursor:
            blocks.append((cursor, bs))
        cursor = max(cursor, be)
    if cursor < we:
        blocks.append((cursor, we))
    return blocks


def score_from_checks(checks: dict[str, bool]) -> tuple[bool, float]:
    total = len(checks)
    n = sum(checks.values())
    return (n == total and total > 0), (n / total if total else 0.0)


# ---------------------------------------------------------------------------
# Dispatch by task_id
# ---------------------------------------------------------------------------

_TASK_MODULE_MAP = {
    "ptc_org_chart_largest_manager_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_largest_manager_0",
    "ptc_queries_shared_supervisor_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_shared_supervisor_pairs_0",
    "ptc_queries_calendar_five_attendees_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_five_attendees_0",
    "ptc_queries_calendar_interview_scheduling_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_interview_scheduling_0",
    "ptc_queries_calendar_free_block_rescheduling_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_free_block_rescheduling_0",
    "ptc_queries_calendar_minimize_dead_time_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_minimize_dead_time_0",
    "ptc_queries_calendar_recurring_weekly_sync_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_recurring_weekly_sync_0",
    "ptc_org_chart_most_teammates_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_most_teammates_0",
    "ptc_queries_calendar_fragmented_weekday_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_fragmented_weekday_0",
    "ptc_queries_calendar_interview_matching_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_interview_matching_0",
    "ptc_queries_calendar_compact_oneononones_0":
        "ptc_eval.eval_checker.verifiers.ptc_queries_calendar_compact_oneononones_0",
    "ptc_yelp_highest_rated_filtered_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_highest_rated_filtered_0",
    "ptc_yelp_chinese_near_boba_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_chinese_near_boba_0",
    "ptc_yelp_top_rated_reviews_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_top_rated_reviews_0",
    "ptc_yelp_cascading_search_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_cascading_search_0",
    "ptc_yelp_category_best_nearby_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_category_best_nearby_0",
    "ptc_multi_meeting_point_0":
        "ptc_eval.eval_checker.verifiers.ptc_multi_meeting_point_0",
    "ptc_yelp_uber_supermarket_route_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_uber_supermarket_route_0",
    "ptc_yelp_min_stops_coverage_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_min_stops_coverage_0",
    "ptc_multi_group_dinner_0":
        "ptc_eval.eval_checker.verifiers.ptc_multi_group_dinner_0",
    "ptc_yelp_uber_food_boba_route_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_uber_food_boba_route_0",
    "ptc_yelp_uber_pareto_chinese_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_uber_pareto_chinese_0",
    "ptc_grocery_set_cover_0":
        "ptc_eval.eval_checker.verifiers.ptc_grocery_set_cover_0",
    "ptc_grocery_single_vs_split_0":
        "ptc_eval.eval_checker.verifiers.ptc_grocery_single_vs_split_0",
    "ptc_grocery_gift_card_0":
        "ptc_eval.eval_checker.verifiers.ptc_grocery_gift_card_0",
    # shuo_ptc tasks (loop, chain, code, combined)
    "ptc_loop_0": "ptc_eval.eval_checker.verifiers.ptc_loop_0",
    "ptc_loop_1": "ptc_eval.eval_checker.verifiers.ptc_loop_1",
    "ptc_loop_2": "ptc_eval.eval_checker.verifiers.ptc_loop_2",
    "ptc_loop_3": "ptc_eval.eval_checker.verifiers.ptc_loop_3",
    "ptc_loop_4": "ptc_eval.eval_checker.verifiers.ptc_loop_4",
    "ptc_loop_5": "ptc_eval.eval_checker.verifiers.ptc_loop_5",
    "ptc_loop_6": "ptc_eval.eval_checker.verifiers.ptc_loop_6",
    "ptc_loop_7": "ptc_eval.eval_checker.verifiers.ptc_loop_7",
    "ptc_loop_8": "ptc_eval.eval_checker.verifiers.ptc_loop_8",
    "ptc_loop_9": "ptc_eval.eval_checker.verifiers.ptc_loop_9",
    "ptc_chain_0": "ptc_eval.eval_checker.verifiers.ptc_chain_0",
    "ptc_chain_1": "ptc_eval.eval_checker.verifiers.ptc_chain_1",
    "ptc_chain_2": "ptc_eval.eval_checker.verifiers.ptc_chain_2",
    "ptc_chain_3": "ptc_eval.eval_checker.verifiers.ptc_chain_3",
    "ptc_chain_4": "ptc_eval.eval_checker.verifiers.ptc_chain_4",
    "ptc_chain_5": "ptc_eval.eval_checker.verifiers.ptc_chain_5",
    "ptc_chain_6": "ptc_eval.eval_checker.verifiers.ptc_chain_6",
    "ptc_chain_7": "ptc_eval.eval_checker.verifiers.ptc_chain_7",
    "ptc_chain_8": "ptc_eval.eval_checker.verifiers.ptc_chain_8",
    "ptc_chain_9": "ptc_eval.eval_checker.verifiers.ptc_chain_9",
    "ptc_code_0": "ptc_eval.eval_checker.verifiers.ptc_code_0",
    "ptc_code_1": "ptc_eval.eval_checker.verifiers.ptc_code_1",
    "ptc_code_2": "ptc_eval.eval_checker.verifiers.ptc_code_2",
    "ptc_code_3": "ptc_eval.eval_checker.verifiers.ptc_code_3",
    "ptc_code_4": "ptc_eval.eval_checker.verifiers.ptc_code_4",
    "ptc_combined_0": "ptc_eval.eval_checker.verifiers.ptc_combined_0",
    "ptc_combined_1": "ptc_eval.eval_checker.verifiers.ptc_combined_1",
    "ptc_combined_2": "ptc_eval.eval_checker.verifiers.ptc_combined_2",
    "ptc_combined_3": "ptc_eval.eval_checker.verifiers.ptc_combined_3",
    "ptc_combined_4": "ptc_eval.eval_checker.verifiers.ptc_combined_4",
    # Multi-turn queries (intermediate result loss)
    "ptc_mt_yelp_top_k_refinement_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_top_k_refinement_0",
    "ptc_mt_yelp_filter_relaxation_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_filter_relaxation_0",
    "ptc_mt_calendar_slot_expansion_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_slot_expansion_0",
    "ptc_mt_grocery_store_comparison_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_grocery_store_comparison_0",
    "ptc_mt_yelp_uber_second_best_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_uber_second_best_0",
    "ptc_mt_org_chart_drill_down_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_drill_down_0",
    "ptc_mt_calendar_constraint_addition_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_constraint_addition_0",
    "ptc_mt_yelp_category_aggregation_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_category_aggregation_0",
    "ptc_mt_grocery_gift_card_reopt_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_grocery_gift_card_reopt_0",
    "ptc_mt_yelp_uber_route_reopt_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_uber_route_reopt_0",
    # Multi-turn v2: cell-advantage queries (generate_multi_turn_v2.py).
    "ptc_mt_venmo_debt_ranking_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_debt_ranking_0",
    "ptc_mt_doordash_menu_drilldown_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_doordash_menu_drilldown_0",
    "ptc_mt_spotify_playlist_overlap_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_playlist_overlap_0",
    "ptc_mt_grocery_two_store_coverage_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_grocery_two_store_coverage_0",
    "ptc_mt_calendar_offsite_subset_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_offsite_subset_0",
    "ptc_mt_yelp_uber_commute_detour_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_uber_commute_detour_0",
    "ptc_mt_org_chart_manager_churn_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_manager_churn_0",
    "ptc_mt_calendar_deep_work_slot_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_deep_work_slot_0",
    "ptc_mt_venmo_grocery_split_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_grocery_split_0",
    "ptc_mt_spotify_deep_cuts_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_deep_cuts_0",
    # new_multiturn_queries.json — Venmo cell-advantage multi-turn batch.
    "ptc_mt_venmo_subscription_hunt_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_subscription_hunt_0",
    "ptc_mt_venmo_reciprocity_audit_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_reciprocity_audit_0",
    "ptc_mt_venmo_beach_house_reconciliation_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_beach_house_reconciliation_0",
    "ptc_mt_venmo_core_financial_circle_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_core_financial_circle_0",
    "ptc_mt_venmo_monthly_budget_breach_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_venmo_monthly_budget_breach_0",
    # new_multiturn_queries.json — OrgChart cell-advantage multi-turn batch.
    "ptc_mt_org_chart_deepest_chain_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_deepest_chain_0",
    "ptc_mt_org_chart_span_of_control_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_span_of_control_0",
    "ptc_mt_org_chart_levels_histogram_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_levels_histogram_0",
    "ptc_mt_org_chart_reorg_churn_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_reorg_churn_0",
    "ptc_mt_org_chart_lca_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_lca_pairs_0",
    "ptc_mt_org_chart_boomerangs_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_boomerangs_0",
    "ptc_mt_org_chart_continuous_tenure_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_continuous_tenure_0",
    "ptc_mt_org_chart_reorg_impact_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_reorg_impact_0",
    "ptc_mt_org_chart_matrix_reports_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_matrix_reports_0",
    "ptc_mt_org_chart_span_deltas_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_org_chart_span_deltas_0",
    # new_multiturn_queries.json — Spotify cell-advantage multi-turn batch.
    "ptc_mt_spotify_track_ubiquity_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_track_ubiquity_0",
    "ptc_mt_spotify_top_artists_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_top_artists_0",
    "ptc_mt_spotify_playlist_stats_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_playlist_stats_0",
    "ptc_mt_spotify_artist_reach_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_artist_reach_0",
    "ptc_mt_spotify_most_isolated_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_most_isolated_0",
    "ptc_mt_spotify_core_repertoire_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_core_repertoire_0",
    "ptc_mt_spotify_recently_played_orphans_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_recently_played_orphans_0",
    "ptc_mt_spotify_similarity_extremes_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_similarity_extremes_0",
    "ptc_mt_spotify_artist_concentration_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_artist_concentration_0",
    "ptc_mt_spotify_duplicate_detector_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_spotify_duplicate_detector_0",
    # new_multiturn_queries.json — Calendar cell-advantage multi-turn batch.
    "ptc_mt_calendar_meeting_load_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_meeting_load_0",
    "ptc_mt_calendar_utc_heatmap_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_utc_heatmap_0",
    "ptc_mt_calendar_back_to_back_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_back_to_back_0",
    "ptc_mt_calendar_rsvp_audit_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_rsvp_audit_0",
    "ptc_mt_calendar_team_free_block_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_team_free_block_0",
    "ptc_mt_calendar_60min_slots_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_60min_slots_0",
    "ptc_mt_calendar_weekly_recurring_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_weekly_recurring_0",
    "ptc_mt_calendar_self_conflict_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_self_conflict_0",
    "ptc_mt_calendar_pairing_frequency_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_pairing_frequency_0",
    "ptc_mt_calendar_fragmentation_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_calendar_fragmentation_0",
    # Intermediate experiment (see ptc_eval/data/error_recovery.jsonl).
    # Uses alt API variants with injected transient errors.
    "ptc_intermediate_yelp_chinese_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_yelp_rank",
    "ptc_intermediate_yelp_uber_dinner_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_yelp_uber_dinner",
    "ptc_intermediate_grocery_coverage_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_grocery_coverage",
    "ptc_intermediate_org_shared_managers_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_org_shared_managers",
    "ptc_intermediate_calendar_cross_tz_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_calendar_cross_tz",
    "ptc_intermediate_spotify_consolidate_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_spotify_consolidate",
    # Intermediate experiment — 14 additional scaled tasks.
    "ptc_intermediate_org_manager_coverage_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_org_manager_coverage",
    "ptc_intermediate_org_teammate_degree_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_org_teammate_degree",
    "ptc_intermediate_calendar_recurring_sync_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_calendar_recurring_sync",
    "ptc_intermediate_calendar_fragmented_weekday_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_calendar_fragmented_weekday",
    "ptc_intermediate_calendar_interview_booking_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_calendar_interview_booking",
    "ptc_intermediate_calendar_reschedule_freeblock_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_calendar_reschedule_freeblock",
    "ptc_intermediate_calendar_dead_time_1on1s_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_calendar_dead_time_1on1s",
    "ptc_intermediate_yelp_uber_tsp_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_yelp_uber_tsp",
    "ptc_intermediate_yelp_uber_pareto_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_yelp_uber_pareto",
    "ptc_intermediate_grocery_bill_optimize_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_grocery_bill_optimize",
    "ptc_intermediate_doordash_value_menu_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_doordash_value_menu",
    "ptc_intermediate_venmo_debt_settle_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_venmo_debt_settle",
    "ptc_intermediate_spotify_artist_diversity_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_spotify_artist_diversity",
    "ptc_intermediate_spotify_deep_cuts_0":
        "ptc_eval.eval_checker.verifiers.ptc_intermediate_spotify_deep_cuts",
    # new_intermediate_queries.json — flaky-VenmoAPI variants of the
    # working_batch venmo queries.  flakiness_config fails the first (only)
    # get_transaction_history call to test transient-failure recovery.
    "ptc_intermediate_venmo_subscription_hunt_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_subscription_hunt",
    "ptc_intermediate_venmo_reciprocity_audit_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_reciprocity_audit",
    "ptc_intermediate_venmo_beach_house_reconciliation_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_beach_house_reconciliation",
    "ptc_intermediate_venmo_core_financial_circle_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_core_financial_circle",
    "ptc_intermediate_venmo_monthly_budget_breach_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_monthly_budget_breach",
    # new_intermediate_queries.json — flaky-OrgChartAPI variants of the
    # working_batch org_chart queries. flakiness_config fails the 68th call
    # of get_reporting_history (or 28th for lca_pairs which iterates over
    # ~38 employees instead of all 90).
    "ptc_intermediate_org_chart_deepest_chain_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_deepest_chain",
    "ptc_intermediate_org_chart_span_of_control_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_span_of_control",
    "ptc_intermediate_org_chart_levels_histogram_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_levels_histogram",
    "ptc_intermediate_org_chart_reorg_churn_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_reorg_churn",
    "ptc_intermediate_org_chart_lca_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_lca_pairs",
    "ptc_intermediate_org_chart_boomerangs_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_boomerangs",
    "ptc_intermediate_org_chart_continuous_tenure_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_continuous_tenure",
    "ptc_intermediate_org_chart_reorg_impact_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_reorg_impact",
    "ptc_intermediate_org_chart_matrix_reports_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_matrix_reports",
    "ptc_intermediate_org_chart_span_deltas_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_span_deltas",
    # new_intermediate_queries.json — flaky-SpotifyAPI variants of the
    # working_batch spotify queries.  Mixed strategy:
    #   - playlist_stats / track_ubiquity: get_playlist N=14 (mid-loop)
    #   - core_repertoire: get_track_details N=8 (mid-loop)
    #   - 7 collapsed queries: get_user_playlists N=1 (first-call retry)
    "ptc_intermediate_spotify_track_ubiquity_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_track_ubiquity",
    "ptc_intermediate_spotify_top_artists_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_top_artists",
    "ptc_intermediate_spotify_playlist_stats_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_playlist_stats",
    "ptc_intermediate_spotify_artist_reach_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_artist_reach",
    "ptc_intermediate_spotify_most_isolated_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_most_isolated",
    "ptc_intermediate_spotify_core_repertoire_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_core_repertoire",
    "ptc_intermediate_spotify_recently_played_orphans_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_recently_played_orphans",
    "ptc_intermediate_spotify_similarity_extremes_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_similarity_extremes",
    "ptc_intermediate_spotify_artist_concentration_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_artist_concentration",
    "ptc_intermediate_spotify_duplicate_detector_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_duplicate_detector",
    # new_intermediate_queries.json — flaky-EnterpriseCalendarAPI variants
    # of the working_batch calendar queries.  Per-query N matches ~75% of
    # observed get_schedule_by_employee_name calls in the dryrun (N=9 for
    # 12-employee queries, N=8 for pairing_frequency, N=4 for the 5-attendee
    # queries team_free_block and 60min_slots).
    "ptc_intermediate_calendar_meeting_load_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_meeting_load",
    "ptc_intermediate_calendar_utc_heatmap_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_utc_heatmap",
    "ptc_intermediate_calendar_back_to_back_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_back_to_back",
    "ptc_intermediate_calendar_rsvp_audit_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_rsvp_audit",
    "ptc_intermediate_calendar_team_free_block_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_team_free_block",
    "ptc_intermediate_calendar_60min_slots_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_60min_slots",
    "ptc_intermediate_calendar_weekly_recurring_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_weekly_recurring",
    "ptc_intermediate_calendar_self_conflict_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_self_conflict",
    "ptc_intermediate_calendar_pairing_frequency_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_pairing_frequency",
    "ptc_intermediate_calendar_fragmentation_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_fragmentation",
    # new_intermediate_queries.json — flaky-YelpAPI variants of the
    # working_batch yelp queries.  Mixed strategy:
    #   - 7 review-heavy queries: get_reviews_for_business at N=2..16 (mid-loop)
    #   - 2 search-iter queries:  search_businesses at N=4 (mid-loop)
    #   - 1 single-search query:  search_businesses at N=1 (first-call retry)
    "ptc_intermediate_yelp_recent_drift_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_recent_drift",
    "ptc_intermediate_yelp_polarization_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_polarization",
    "ptc_intermediate_yelp_price_grid_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_price_grid",
    "ptc_intermediate_yelp_useful_top_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_useful_top",
    "ptc_intermediate_yelp_harsh_reviewers_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_harsh_reviewers",
    "ptc_intermediate_yelp_breadth_then_avg_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_breadth_then_avg",
    "ptc_intermediate_yelp_top_then_latest_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_top_then_latest",
    "ptc_intermediate_yelp_review_threshold_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_review_threshold",
    "ptc_intermediate_yelp_category_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_category_pairs",
    "ptc_intermediate_yelp_cascade_search_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_cascade_search",
    # new_intermediate_queries.json — flaky variants of 5 mixed queries:
    #   - 3 grocery: get_store_inventory at N=15..38 (mid-loop)
    #   - 2 multi-API: both uber methods (estimate_ride / get_ride_estimate)
    #     configured at the same N so flakiness fires regardless of which
    #     pricing method gpt-5-mini picks.
    "ptc_intermediate_grocery_gift_card_0":
        "ptc_eval.eval_checker.verifiers.ptc_grocery_gift_card_0",
    "ptc_intermediate_grocery_set_cover_0":
        "ptc_eval.eval_checker.verifiers.ptc_grocery_set_cover_0",
    "ptc_intermediate_grocery_single_vs_split_0":
        "ptc_eval.eval_checker.verifiers.ptc_grocery_single_vs_split_0",
    "ptc_intermediate_multi_group_dinner_0":
        "ptc_eval.eval_checker.verifiers.ptc_multi_group_dinner_0",
    "ptc_intermediate_multi_meeting_point_0":
        "ptc_eval.eval_checker.verifiers.ptc_multi_meeting_point_0",
    # new_queries.json — Venmo batch
    "ptc_venmo_subscription_hunt_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_subscription_hunt",
    "ptc_venmo_reciprocity_audit_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_reciprocity_audit",
    "ptc_venmo_beach_house_reconciliation_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_beach_house_reconciliation",
    "ptc_venmo_core_financial_circle_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_core_financial_circle",
    "ptc_venmo_monthly_budget_breach_0":
        "ptc_eval.eval_checker.verifiers.ptc_venmo_monthly_budget_breach",
    # new_queries.json — OrgChart batch
    "ptc_org_chart_deepest_chain_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_deepest_chain",
    "ptc_org_chart_span_of_control_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_span_of_control",
    "ptc_org_chart_levels_histogram_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_levels_histogram",
    "ptc_org_chart_reorg_churn_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_reorg_churn",
    "ptc_org_chart_lca_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_lca_pairs",
    "ptc_org_chart_boomerangs_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_boomerangs",
    "ptc_org_chart_continuous_tenure_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_continuous_tenure",
    "ptc_org_chart_reorg_impact_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_reorg_impact",
    "ptc_org_chart_matrix_reports_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_matrix_reports",
    "ptc_org_chart_span_deltas_0":
        "ptc_eval.eval_checker.verifiers.ptc_org_chart_span_deltas",
    # new_queries.json — Spotify batch
    "ptc_spotify_track_ubiquity_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_track_ubiquity",
    "ptc_spotify_top_artists_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_top_artists",
    "ptc_spotify_playlist_stats_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_playlist_stats",
    "ptc_spotify_artist_reach_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_artist_reach",
    "ptc_spotify_most_isolated_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_most_isolated",
    "ptc_spotify_core_repertoire_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_core_repertoire",
    "ptc_spotify_recently_played_orphans_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_recently_played_orphans",
    "ptc_spotify_similarity_extremes_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_similarity_extremes",
    "ptc_spotify_artist_concentration_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_artist_concentration",
    "ptc_spotify_duplicate_detector_0":
        "ptc_eval.eval_checker.verifiers.ptc_spotify_duplicate_detector",
    # new_queries.json — Yelp review-intelligence batch
    "ptc_yelp_recent_drift_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_recent_drift",
    "ptc_yelp_polarization_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_polarization",
    "ptc_yelp_price_grid_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_price_grid",
    "ptc_yelp_useful_top_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_useful_top",
    "ptc_yelp_harsh_reviewers_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_harsh_reviewers",
    "ptc_yelp_breadth_then_avg_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_breadth_then_avg",
    "ptc_yelp_top_then_latest_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_top_then_latest",
    "ptc_yelp_review_threshold_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_review_threshold",
    "ptc_yelp_category_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_category_pairs",
    "ptc_yelp_cascade_search_0":
        "ptc_eval.eval_checker.verifiers.ptc_yelp_cascade_search",
    # new_queries.json — Calendar batch
    "ptc_calendar_meeting_load_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_meeting_load",
    "ptc_calendar_utc_heatmap_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_utc_heatmap",
    "ptc_calendar_back_to_back_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_back_to_back",
    "ptc_calendar_rsvp_audit_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_rsvp_audit",
    "ptc_calendar_team_free_block_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_team_free_block",
    "ptc_calendar_60min_slots_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_60min_slots",
    "ptc_calendar_weekly_recurring_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_weekly_recurring",
    "ptc_calendar_self_conflict_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_self_conflict",
    "ptc_calendar_pairing_frequency_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_pairing_frequency",
    "ptc_calendar_fragmentation_0":
        "ptc_eval.eval_checker.verifiers.ptc_calendar_fragmentation",
    # Multi-turn yelp Q1+Q2 batch (10 tasks)
    "ptc_mt_yelp_recent_drift_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_recent_drift_0",
    "ptc_mt_yelp_polarization_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_polarization_0",
    "ptc_mt_yelp_price_grid_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_price_grid_0",
    "ptc_mt_yelp_useful_top_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_useful_top_0",
    "ptc_mt_yelp_harsh_reviewers_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_harsh_reviewers_0",
    "ptc_mt_yelp_breadth_then_avg_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_breadth_then_avg_0",
    "ptc_mt_yelp_top_then_latest_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_top_then_latest_0",
    "ptc_mt_yelp_review_threshold_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_review_threshold_0",
    "ptc_mt_yelp_category_pairs_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_category_pairs_0",
    "ptc_mt_yelp_cascade_search_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_yelp_cascade_search_0",
    # Multi-turn grocery + multi-API batch (5 tasks)
    "ptc_mt_grocery_gift_card_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_grocery_gift_card_0",
    "ptc_mt_grocery_set_cover_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_grocery_set_cover_0",
    "ptc_mt_grocery_single_vs_split_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_grocery_single_vs_split_0",
    "ptc_mt_multi_group_dinner_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_multi_group_dinner_0",
    "ptc_mt_multi_meeting_point_0":
        "ptc_eval.eval_checker.verifiers.ptc_mt_multi_meeting_point_0",
}


def verify(
    task_id: str,
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
    turn_answers: list[str | None] | None = None,
) -> VerificationResult:
    """Dispatch to the per-task verify() function.

    For multi-turn tasks, *turn_answers* carries the per-turn intermediate
    answers (excluding the final one which is in *final_answer*).  Verifiers
    that need per-turn checking accept an optional ``turn_answers`` kwarg.
    """
    mod_path = _TASK_MODULE_MAP.get(task_id)
    if mod_path is None:
        return VerificationResult(
            passed=False, score=0.0,
            errors=[f"No verifier registered for task_id: {task_id}"],
        )
    mod = importlib.import_module(mod_path)

    # Pass turn_answers to verifiers that accept it (multi-turn verifiers).
    sig = inspect.signature(mod.verify)
    if "turn_answers" in sig.parameters:
        return mod.verify(final_answer, final_state, initial_config,
                          ground_truth, turn_answers=turn_answers)
    return mod.verify(final_answer, final_state, initial_config, ground_truth)
