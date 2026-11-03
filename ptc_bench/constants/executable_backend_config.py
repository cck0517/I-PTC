"""
Maps backend class names to module paths and function-doc filenames (BFCL-style).
"""

BACKEND_PATH_PREFIX = "ptc_bench.eval_checker.multi_turn_eval.func_source_code"
BACKEND_PATH_PREFIX_ALT = "ptc_bench.eval_checker.multi_turn_eval.func_source_code_alt"

ALL_API_CLASSES = [
    "EnterpriseCalendarAPI",
    "OrgChartAPI",
    "YelpAPI",
    "GoogleMapsAPI",
    "UberAPI",
    "GroceryDeliveryAPI",
    "PersonalCalendarAPI",
    "DoorDashAPI",
    "SpotifyAPI",
    "VenmoAPI",
]

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
    # Error-recovery variants use separate tool docs.
    "YelpAPIAlt": "yelp_api_alt.json",
    "SpotifyAPIAlt": "spotify_api_alt.json",
    "UberAPIAlt": "uber_api_alt.json",
    "GroceryDeliveryAPIAlt": "grocery_delivery_api_alt.json",
    "OrgChartAPIAlt": "org_chart_api_alt.json",
    "EnterpriseCalendarAPIAlt": "enterprise_calendar_api_alt.json",
    "DoorDashAPIAlt": "doordash_api_alt.json",
    "VenmoAPIAlt": "venmo_api_alt.json",
}

# Tuple entries preserve base class names in model-visible tracebacks.
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
