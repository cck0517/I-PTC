"""
Uber Dummy API (in-memory, deterministic, benchmark-friendly)

Provides ride-cost estimation based on haversine distance and a
configurable pricing model, plus ride-hailing lifecycle (request,
cancel, rate, history).  No real network requests.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

DEFAULT_STATE: Dict[str, Any] = {
    "pricing": {
        "base_fare": 2.50,
        "per_mile_rate": 1.75,
        "per_minute_rate": 0.35,
        "minimum_fare": 8.00,
        "surge_multiplier": 1.0,
        "average_speed_mph": 25,
    },
    "rides": {},
    "current_ride": None,
    "ride_counter": 0,
}


class UberAPI:
    """
    In-memory mock of an Uber ride-estimation service.

    State variables:
    - pricing: Dict with base_fare, per_mile_rate, per_minute_rate,
      minimum_fare, surge_multiplier, average_speed_mph.
    - rides: List of booked rides (for state tracking).
    """

    def __init__(self) -> None:
        self.pricing: Dict[str, Any] = {}
        self.rides: Dict[str, Dict[str, Any]] = {}
        self.current_ride: Dict[str, Any] | None = None
        self.ride_counter: int = 0
        self._api_description = (
            "This tool belongs to the UberAPI mock, which provides ride "
            "cost estimation and ride-hailing services."
        )

    def _load_scenario(
        self, scenario: Dict[str, Any], long_context: bool = False
    ) -> None:
        d = deepcopy(DEFAULT_STATE)
        self.pricing = scenario.get("pricing", d["pricing"])
        # Handle rides as dict (shuo_ptc) or list (original)
        raw_rides = scenario.get("rides", d["rides"])
        if isinstance(raw_rides, list):
            self.rides = {r.get("id", f"ride_{i}"): r for i, r in enumerate(raw_rides)}
        else:
            self.rides = raw_rides
        self.current_ride = scenario.get("current_ride", d["current_ride"])
        self.ride_counter = scenario.get("ride_counter", d["ride_counter"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, UberAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in miles."""
        R = 3958.8
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return round(2 * R * math.asin(math.sqrt(a)), 2)

    # ------------------------------------------------------------------
    # Ride estimation
    # ------------------------------------------------------------------

    def get_ride_estimate(
        self,
        start_latitude: float,
        start_longitude: float,
        end_latitude: float,
        end_longitude: float,
        ride_type: str = "UberX",
    ) -> Dict[str, Any]:
        """
        Estimate the cost of an Uber ride between two GPS coordinates.

        The estimate is computed from the straight-line (haversine)
        distance and the pricing parameters loaded in the scenario.

        Args:
            start_latitude (float): Pickup latitude.
            start_longitude (float): Pickup longitude.
            end_latitude (float): Destination latitude.
            end_longitude (float): Destination longitude.
            ride_type (str): Ride tier. Defaults to "UberX".

        Returns:
            Dict[str, Any]:
                ride_type (str), estimated_cost_dollars (float),
                estimated_duration_minutes (int), distance_miles (float),
                surge_multiplier (float).
        """
        dist = self._haversine(
            start_latitude, start_longitude, end_latitude, end_longitude
        )
        speed = self.pricing.get("average_speed_mph", 25)
        duration_min = dist / speed * 60

        base = self.pricing.get("base_fare", 2.50)
        per_mile = self.pricing.get("per_mile_rate", 1.75)
        per_min = self.pricing.get("per_minute_rate", 0.35)
        minimum = self.pricing.get("minimum_fare", 8.00)
        surge = self.pricing.get("surge_multiplier", 1.0)

        raw_cost = base + dist * per_mile + duration_min * per_min
        cost = round(max(minimum, raw_cost) * surge, 2)

        return {
            "ride_type": ride_type,
            "estimated_cost_dollars": cost,
            "estimated_duration_minutes": round(duration_min),
            "distance_miles": dist,
            "surge_multiplier": surge,
        }

    # ------------------------------------------------------------------
    # String-address helpers (for shuo_ptc-style tasks)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fare(start_location: str, end_location: str, ride_type: str = "UberX") -> float:
        """Compute a deterministic fare based on location strings."""
        # md5 instead of built-in hash() — Python's hash() is salted per-process
        # via PYTHONHASHSEED, so the same trip yielded different fares each run.
        combined = f"{start_location}|{end_location}"
        h = int(hashlib.md5(combined.encode("utf-8")).hexdigest()[:8], 16) % 3000
        base_fare = 5.0 + h / 100.0
        multipliers = {"UberX": 1.0, "UberXL": 1.5, "Comfort": 1.3, "UberBlack": 2.0}
        multiplier = multipliers.get(ride_type, 1.0)
        return round(base_fare * multiplier, 2)

    # ------------------------------------------------------------------
    # Ride lifecycle (string-address based)
    # ------------------------------------------------------------------

    def estimate_ride(
        self, start_location: str, end_location: str, ride_type: str = "UberX"
    ) -> Dict[str, Union[float, int, str]]:
        """
        Estimate the fare for a ride between two addresses.

        Args:
            start_location (str): Starting address or location.
            end_location (str): Destination address or location.
            ride_type (str): Type of ride. Defaults to "UberX".

        Returns:
            Dict[str, Any]:
                estimated_fare (float), estimated_duration (int),
                distance (float), ride_type (str).
        """
        fare = self._compute_fare(start_location, end_location, ride_type)
        duration = max(5, int(fare / 1.5))
        distance = round(fare / 3.0, 1)
        return {
            "estimated_fare": fare,
            "estimated_duration": duration,
            "distance": distance,
            "ride_type": ride_type,
        }

    def request_ride(
        self, start_location: str, end_location: str, ride_type: str = "UberX"
    ) -> Dict[str, Union[str, float]]:
        """
        Request an Uber ride.

        Args:
            start_location (str): Starting address or location.
            end_location (str): Destination address or location.
            ride_type (str): Type of ride. Defaults to "UberX".

        Returns:
            Dict[str, Any]:
                ride_id (str), status (str), start_location (str),
                end_location (str), fare (float), ride_type (str).
        """
        if self.current_ride is not None:
            return {"error": "A ride is already in progress. Cancel it before requesting a new one."}

        fare = self._compute_fare(start_location, end_location, ride_type)
        self.ride_counter += 1
        ride_id = f"ride_{self.ride_counter}"
        ride = {
            "id": ride_id,
            "start_location": start_location,
            "end_location": end_location,
            "fare": fare,
            "status": "requested",
            "driver": "Assigned Driver",
            "rating": None,
            "date": "2026-03-31",
            "ride_type": ride_type,
        }
        self.rides[ride_id] = ride
        self.current_ride = ride
        return {
            "ride_id": ride_id,
            "status": "requested",
            "start_location": start_location,
            "end_location": end_location,
            "fare": fare,
            "ride_type": ride_type,
        }

    def get_ride_status(self, ride_id: str) -> Dict:
        """
        Get the status of a specific ride.

        Args:
            ride_id (str): ID of the ride to check.

        Returns:
            Dict: Ride details including status, fare, driver, etc.
        """
        if ride_id not in self.rides:
            return {"error": f"Ride '{ride_id}' not found."}
        return self.rides[ride_id]

    def cancel_ride(self, ride_id: str) -> Dict[str, bool]:
        """
        Cancel an active ride.

        Args:
            ride_id (str): ID of the ride to cancel.

        Returns:
            Dict[str, bool]: cancel_status True if cancelled.
        """
        if ride_id not in self.rides:
            return {"error": f"Ride '{ride_id}' not found."}
        ride = self.rides[ride_id]
        if ride["status"] not in ("requested", "in_progress"):
            return {"error": f"Cannot cancel ride with status '{ride['status']}'."}
        ride["status"] = "cancelled"
        if self.current_ride and self.current_ride["id"] == ride_id:
            self.current_ride = None
        return {"cancel_status": True}

    def rate_ride(self, ride_id: str, rating: int, comment: str = "") -> Dict[str, bool]:
        """
        Rate a completed ride.

        Args:
            ride_id (str): ID of the ride to rate.
            rating (int): Rating from 1 to 5.
            comment (str): Optional comment for the driver.

        Returns:
            Dict[str, bool]: rating_status True if rated.
        """
        if ride_id not in self.rides:
            return {"error": f"Ride '{ride_id}' not found."}
        ride = self.rides[ride_id]
        if ride["status"] != "completed":
            return {"error": "Can only rate completed rides."}
        if not 1 <= rating <= 5:
            return {"error": "Rating must be between 1 and 5."}
        ride["rating"] = rating
        return {"rating_status": True}

    def get_ride_history(
        self, start_date: str = None, end_date: str = None
    ) -> Dict[str, List[Dict]]:
        """
        Get ride history, optionally filtered by date range.

        Args:
            start_date (str): Optional start date (YYYY-MM-DD).
            end_date (str): Optional end date (YYYY-MM-DD).

        Returns:
            Dict with 'rides' list of ride records.
        """
        result = list(self.rides.values())
        if start_date:
            result = [r for r in result if r.get("date", "") >= start_date]
        if end_date:
            result = [r for r in result if r.get("date", "") <= end_date]
        return {"rides": result}
