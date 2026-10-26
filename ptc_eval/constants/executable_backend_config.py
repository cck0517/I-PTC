"""
Maps backend class names to module paths and function-doc filenames (BFCL-style).
"""

BACKEND_PATH_PREFIX = "ptc_eval.eval_checker.multi_turn_eval.func_source_code"
BACKEND_PATH_PREFIX_ALT = "ptc_eval.eval_checker.multi_turn_eval.func_source_code_alt"

MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "OrgChartAPI": "org_chart_api.json",
    "EnterpriseCalendarAPI": "enterprise_calendar_api.json",
    "YelpAPI": "yelp_api.json",
    "GoogleMapsAPI": "google_maps_api.json",
    "UberAPI": "uber_api.json",
    "GroceryDeliveryAPI": "grocery_delivery_api.json",
    "PersonalCalendarAPI": "personal_calendar_api.json",
    "DoorDashAPI": "doordash_api.json",
    "SpotifyAPI": "spotify_api.json",
    "VenmoAPI": "venmo_api.json",
    # Added for the PTC_queries_simple cost-amplification experiment. Not part
    # of _ALL_API_CLASSES, so it is never injected into --expose-all-apis runs
    # of the main benchmark; only tasks that name it in involved_classes see it.
    # Added for the PTC_queries_simple schema-ambiguity experiment (federated
    # legacy directory with polymorphic responses). Also kept out of
    # _ALL_API_CLASSES so it never enters --expose-all-apis benchmark runs.
    # Added for the PTC_queries_simple "ambigmix" batch — five independent
    # single-API dependent-walk tasks, each on its own compact API with a
    # distinct surface (method names, fields, domain) but the same polymorphic
    # chain logic. Kept out of _ALL_API_CLASSES so they never enter
    # --expose-all-apis benchmark runs.
    # "ambigmix" batch 2 — five more independent single-API dependent-walk
    # tasks on fresh surfaces. Also kept out of _ALL_API_CLASSES.
    # "ambigmix" batch 3 — five more fresh-surface dependent-walk APIs.
    # "ambigmix" batch 4 — five more fresh-surface dependent-walk APIs.
    # "ambigmix" batch 5 — twelve more fresh-surface dependent-walk APIs.
    # Alt variants used by tasks in error_recovery.jsonl.  They
    # reference their own doc files so the main-benchmark docs stay pristine.
    "YelpAPIAlt": "yelp_api_alt.json",
    "SpotifyAPIAlt": "spotify_api_alt.json",
    "UberAPIAlt": "uber_api_alt.json",
    "GroceryDeliveryAPIAlt": "grocery_delivery_api_alt.json",
    "OrgChartAPIAlt": "org_chart_api_alt.json",
    "EnterpriseCalendarAPIAlt": "enterprise_calendar_api_alt.json",
    "DoorDashAPIAlt": "doordash_api_alt.json",
    "VenmoAPIAlt": "venmo_api_alt.json",
}

# Values are either ``module_path`` (string) or ``(module_path, class_attr)``.
# The tuple form lets an entry like ``YelpAPIAlt`` load a module whose Python
# class is literally named ``YelpAPI`` — keeping the class name that surfaces
# in tracebacks identical to the main-benchmark class.
CLASS_FILE_PATH_MAPPING = {
    "OrgChartAPI": f"{BACKEND_PATH_PREFIX}.org_chart_api",
    "EnterpriseCalendarAPI": f"{BACKEND_PATH_PREFIX}.enterprise_calendar_api",
    "YelpAPI": f"{BACKEND_PATH_PREFIX}.yelp_api",
    "GoogleMapsAPI": f"{BACKEND_PATH_PREFIX}.google_maps_api",
    "UberAPI": f"{BACKEND_PATH_PREFIX}.uber_api",
    "GroceryDeliveryAPI": f"{BACKEND_PATH_PREFIX}.grocery_delivery_api",
    "PersonalCalendarAPI": f"{BACKEND_PATH_PREFIX}.personal_calendar_api",
    "DoorDashAPI": f"{BACKEND_PATH_PREFIX}.doordash_api",
    "SpotifyAPI": f"{BACKEND_PATH_PREFIX}.spotify_api",
    "VenmoAPI": f"{BACKEND_PATH_PREFIX}.venmo_api",
    "YelpAPIAlt":                (f"{BACKEND_PATH_PREFIX_ALT}.yelp_api",                "YelpAPI"),
    "SpotifyAPIAlt":             (f"{BACKEND_PATH_PREFIX_ALT}.spotify_api",             "SpotifyAPI"),
    "UberAPIAlt":                (f"{BACKEND_PATH_PREFIX_ALT}.uber_api",                "UberAPI"),
    "GroceryDeliveryAPIAlt":     (f"{BACKEND_PATH_PREFIX_ALT}.grocery_delivery_api",    "GroceryDeliveryAPI"),
    "OrgChartAPIAlt":            (f"{BACKEND_PATH_PREFIX_ALT}.org_chart_api",           "OrgChartAPI"),
    "EnterpriseCalendarAPIAlt":  (f"{BACKEND_PATH_PREFIX_ALT}.enterprise_calendar_api", "EnterpriseCalendarAPI"),
    "DoorDashAPIAlt":            (f"{BACKEND_PATH_PREFIX_ALT}.doordash_api",            "DoorDashAPI"),
    "VenmoAPIAlt":               (f"{BACKEND_PATH_PREFIX_ALT}.venmo_api",               "VenmoAPI"),
}

STATELESS_CLASSES: list[str] = []

OMIT_STATE_INFO_CLASSES: list[str] = []
