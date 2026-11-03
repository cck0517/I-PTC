"""Post-run cost labels for agent-callable API methods.

Labels describe real services: ``io`` reads remote data, ``compute`` is local,
and ``io_mutation`` changes remote state. Mock implementations do not affect
classification; cached geocoding, for example, is still ``io``.

Keys use ``<api_var>.<method>`` from ``_CLASS_TO_PREFIX`` in iptc/ptc_tool.py.
Base and alternate classes share prefixes, so overrides inherit labels.

Run ``python -m ptc_bench.eval_checker.multi_turn_eval.api_method_types``
to verify completeness against the API source files.
"""

from __future__ import annotations

API_METHOD_TYPES: dict[str, str] = {
    "doordash_api.search_restaurants": "io",
    "doordash_api.get_restaurant_menu": "io",
    "doordash_api.place_order": "io_mutation",
    "doordash_api.get_order_status": "io",
    "doordash_api.cancel_order": "io_mutation",
    "doordash_api.get_order_history": "io",
    "doordash_api.rate_order": "io_mutation",
    "doordash_api.get_delivery_estimate": "io",

    "enterprise_calendar_api.get_calendar_context": "io",
    "enterprise_calendar_api.get_user_schedule": "io",
    "enterprise_calendar_api.get_schedule_by_employee_name": "io",
    # Pure timezone arithmetic — no service call.
    "enterprise_calendar_api.convert_datetime_between_timezones": "compute",
    "enterprise_calendar_api.book_event": "io_mutation",

    "maps_api.geocode_address": "io",

    "grocery_api.get_shopping_list": "io",
    "grocery_api.list_stores": "io",
    "grocery_api.get_store_inventory": "io",
    "grocery_api.get_gift_card_balance": "io",

    "org_chart_api.list_employees": "io",
    "org_chart_api.get_reporting_history": "io",

    "personal_calendar_api.get_calendar_context": "io",
    "personal_calendar_api.create_event": "io_mutation",
    "personal_calendar_api.get_events": "io",
    "personal_calendar_api.get_events_range": "io",
    "personal_calendar_api.delete_event": "io_mutation",
    "personal_calendar_api.get_event_details": "io",
    "personal_calendar_api.update_event": "io_mutation",
    "personal_calendar_api.add_attendee": "io_mutation",
    # Reads the schedule then picks gaps. Dominant cost is the schedule read.
    "personal_calendar_api.find_free_slots": "io",

    "spotify_api.search_tracks": "io",
    "spotify_api.search_playlists": "io",
    "spotify_api.get_playlist": "io",
    "spotify_api.create_playlist": "io_mutation",
    "spotify_api.add_track_to_playlist": "io_mutation",
    "spotify_api.remove_track_from_playlist": "io_mutation",
    "spotify_api.get_track_details": "io",
    "spotify_api.get_recently_played": "io",
    "spotify_api.get_user_playlists": "io",
    "spotify_api.get_recommendations": "io",
    "spotify_api.save_playlist": "io_mutation",
    # Alt-only.
    "spotify_api.get_track_popularity_index": "io",

    "uber_api.get_ride_estimate": "io",
    "uber_api.estimate_ride": "io",
    "uber_api.request_ride": "io_mutation",
    "uber_api.get_ride_status": "io",
    "uber_api.cancel_ride": "io_mutation",
    "uber_api.rate_ride": "io_mutation",
    "uber_api.get_ride_history": "io",

    "venmo_api.get_balance": "io",
    "venmo_api.add_friend": "io_mutation",
    "venmo_api.get_friends": "io",
    "venmo_api.send_payment": "io_mutation",
    "venmo_api.request_payment": "io_mutation",
    "venmo_api.get_transaction_history": "io",
    "venmo_api.get_pending_requests": "io",
    "venmo_api.accept_request": "io_mutation",
    "venmo_api.decline_request": "io_mutation",

    "yelp_api.get_yelp_profile": "io",
    "yelp_api.search_businesses": "io",
    "yelp_api.get_business": "io",
    "yelp_api.get_reviews_for_business": "io",
    "yelp_api.get_review_details": "io",
    "yelp_api.get_my_yelp_reviews": "io",
    "yelp_api.compose_review": "io_mutation",
    "yelp_api.update_review": "io_mutation",
    "yelp_api.remove_review": "io_mutation",
    "yelp_api.mark_review_useful": "io_mutation",
    "yelp_api.save_business": "io_mutation",
    "yelp_api.unsave_business": "io_mutation",
    "yelp_api.list_saved_businesses": "io",
    "yelp_api.bookmark_business": "io_mutation",
    "yelp_api.remove_bookmark": "io_mutation",
    "yelp_api.get_bookmarks": "io",
    "yelp_api.search_events": "io",
    "yelp_api.get_distance_to_business": "io",
    "yelp_api.get_distance_between_businesses": "io",
    "yelp_api.check_business_open": "io",
    "yelp_api.get_wait_time": "io",
    "yelp_api.make_reservation": "io_mutation",
    # Alt-only.
    "yelp_api.get_business_popularity_index": "io",
}


VALID_TYPES: frozenset[str] = frozenset({"io", "compute", "io_mutation"})


def classify(api: str, method: str) -> str:
    """Return the cost-type tag for ``<api>.<method>``, or ``"unknown"``."""
    return API_METHOD_TYPES.get(f"{api}.{method}", "unknown")


def compute_trace_api_type_distribution(trace: dict) -> dict[str, int]:
    """Count cost types in direct tool calls and PTC result ``api_calls``.

    Returns only observed types; missing call logs produce an empty dict.
    """
    from collections import Counter
    counts: Counter[str] = Counter()
    for turn in trace.get("turns", []) or []:
        for tc in turn.get("tool_calls", []) or []:
            tool_name = tc.get("tool_name") or ""
            if tool_name in ("ptc_execute", "ptc_cell"):
                # Logs are capped at 100 entries. Scaling assumes the hidden
                # tail has the same distribution as the visible prefix.
                result = tc.get("result")
                if not isinstance(result, dict):
                    continue
                cell_counts: Counter[str] = Counter()
                for call in result.get("api_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    api = call.get("api")
                    method = call.get("method")
                    if api and method:
                        cell_counts[classify(api, method)] += 1
                visible_classified = sum(cell_counts.values())
                true_count = result.get("api_call_count", visible_classified)
                if visible_classified > 0 and true_count > visible_classified:
                    scale = true_count / visible_classified
                    for k in list(cell_counts):
                        cell_counts[k] = int(round(cell_counts[k] * scale))
                counts.update(cell_counts)
            elif "." in tool_name:
                api, method = tool_name.split(".", 1)
                counts[classify(api, method)] += 1
    return dict(counts)


def _enumerate_methods() -> dict[str, set[str]]:
    """Find public base/alt API methods via AST, avoiding runtime imports."""
    import ast
    from pathlib import Path

    # Keep in sync with _CLASS_TO_PREFIX without importing iptc.
    class_to_var = {
        "EnterpriseCalendarAPI": "enterprise_calendar_api",
        "OrgChartAPI": "org_chart_api",
        "YelpAPI": "yelp_api",
        "GoogleMapsAPI": "maps_api",
        "UberAPI": "uber_api",
        "GroceryDeliveryAPI": "grocery_api",
        "PersonalCalendarAPI": "personal_calendar_api",
        "DoorDashAPI": "doordash_api",
        "SpotifyAPI": "spotify_api",
        "VenmoAPI": "venmo_api",
    }

    NON_API_METHODS = {
        "to_dict", "from_dict",
        "load_scenario", "load_initial_config",
        "_load_scenario",
    }

    here = Path(__file__).resolve().parent
    roots = [here / "func_source_code", here / "func_source_code_alt"]

    found: dict[str, set[str]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for src_file in sorted(root.glob("*.py")):
            if src_file.name in ("__init__.py", "server_patch_mixin.py"):
                continue
            if src_file.name.startswith("_"):
                continue
            try:
                tree = ast.parse(src_file.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                var = class_to_var.get(node.name)
                if var is None:
                    continue
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    name = child.name
                    if name.startswith("_") or name in NON_API_METHODS:
                        continue
                    found.setdefault(var, set()).add(name)
    return found


def _verify() -> int:
    methods = _enumerate_methods()
    missing: list[str] = []
    for var, names in methods.items():
        for name in sorted(names):
            key = f"{var}.{name}"
            if key not in API_METHOD_TYPES:
                missing.append(key)

    extra: list[str] = []
    valid_keys = {f"{v}.{m}" for v, ms in methods.items() for m in ms}
    for key in API_METHOD_TYPES:
        if key not in valid_keys:
            extra.append(key)

    invalid_tags: list[tuple[str, str]] = [
        (k, v) for k, v in API_METHOD_TYPES.items() if v not in VALID_TYPES
    ]

    total_methods = sum(len(ms) for ms in methods.values())
    print(f"Discovered {total_methods} methods across {len(methods)} API classes.")
    print(f"Registry has {len(API_METHOD_TYPES)} entries.")

    if missing:
        print(f"\nMISSING ({len(missing)}) — methods without a label:")
        for k in missing:
            print(f"  {k}")
    if extra:
        print(f"\nORPHAN ({len(extra)}) — registry entries with no matching method:")
        for k in extra:
            print(f"  {k}")
    if invalid_tags:
        print(f"\nINVALID TAGS ({len(invalid_tags)}) — must be one of {sorted(VALID_TYPES)}:")
        for k, v in invalid_tags:
            print(f"  {k} = {v!r}")

    if not (missing or extra or invalid_tags):
        print("\nOK — registry covers every discovered method, no orphans, all tags valid.")
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(_verify())
