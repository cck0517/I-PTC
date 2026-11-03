"""Uber variant with configurable transient pricing failures.

Only get_ride_estimate uses the legacy 13th-call failure default.
"""

from __future__ import annotations

from typing import Any, Dict

from ..func_source_code.uber_api import UberAPI as _BaseUberAPI
from ._flaky import check_flaky


class UberError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:
        return f"UberError: {self.args[0]}"


class UberAPI(_BaseUberAPI):
    def get_ride_estimate(
        self,
        start_latitude: float,
        start_longitude: float,
        end_latitude: float,
        end_longitude: float,
        ride_type: str = "UberX",
    ) -> Dict[str, Any]:
        if check_flaky(self, "get_ride_estimate"):
            raise UberError(
                "INTERNAL_ERROR",
                "Upstream pricing service returned an internal error. Please try again.",
                start_latitude=start_latitude,
                start_longitude=start_longitude,
                end_latitude=end_latitude,
                end_longitude=end_longitude,
            )
        return super().get_ride_estimate(
            start_latitude, start_longitude, end_latitude, end_longitude, ride_type,
        )

    def estimate_ride(
        self,
        start_location: str,
        end_location: str,
        ride_type: str = "UberX",
    ) -> Dict[str, Any]:
        if check_flaky(self, "estimate_ride", use_legacy_default=False):
            raise UberError(
                "INTERNAL_ERROR",
                "Upstream pricing service returned an internal error. Please try again.",
                start_location=start_location,
                end_location=end_location,
            )
        return super().estimate_ride(start_location, end_location, ride_type)
