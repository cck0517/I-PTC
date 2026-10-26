"""
Google Maps Dummy API (in-memory, deterministic, benchmark-friendly)

Provides geocoding from a pre-loaded cache.  No real network requests.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

DEFAULT_STATE: Dict[str, Any] = {
    "geocode_cache": {},
}


class GoogleMapsAPI:
    """
    In-memory mock of a Google Maps geocoding service.

    State variables:
    - geocode_cache: Dict mapping address strings to
      {latitude, longitude, formatted_address}.
    """

    def __init__(self) -> None:
        self.geocode_cache: Dict[str, Dict[str, Any]] = {}
        self._api_description = (
            "This tool belongs to the GoogleMapsAPI mock, which provides "
            "address geocoding (converting addresses to latitude/longitude "
            "coordinates)."
        )

    def _load_scenario(
        self, scenario: Dict[str, Any], long_context: bool = False
    ) -> None:
        d = deepcopy(DEFAULT_STATE)
        self.geocode_cache = scenario.get("geocode_cache", d["geocode_cache"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GoogleMapsAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------

    def geocode_address(self, address: str) -> Dict[str, Any]:
        """
        Convert a street address to geographic coordinates.

        The result is looked up from a pre-loaded cache.  Matching is
        case-insensitive and ignores leading/trailing whitespace.

        Args:
            address (str): The street address to geocode
                (e.g. "123 Main St, San Francisco, CA").

        Returns:
            Dict[str, Any]:
                latitude (float): The latitude of the address.
                longitude (float): The longitude of the address.
                formatted_address (str): The canonical formatted address.
        """
        key = address.strip().lower()
        for cached_addr, data in self.geocode_cache.items():
            if cached_addr.strip().lower() == key:
                return deepcopy(data)
        return {
            "error": f"Could not geocode address: {address!r}",
            "suggestion": "Check the address for typos or try a more specific address.",
        }
