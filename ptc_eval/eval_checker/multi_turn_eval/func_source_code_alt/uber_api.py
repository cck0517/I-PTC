"""Variant of UberAPI used by the intermediate experiment.

Two transient-failure injection points:

  - ``get_ride_estimate`` — the lat/lon-based pricing method.  Falls back
    to legacy default-13 when no flakiness_config is set.
  - ``estimate_ride`` — the address-based pricing method.  Same semantic
    purpose, different signature; gpt-5-mini sometimes picks this one
    instead.  Uses ``use_legacy_default=False`` so legacy queries don't
    fire on this newly-added method.

The class is named ``UberAPI`` so tracebacks surfaced to the model do not
broadcast that this is a variant; the module lives at
``func_source_code_alt/uber_api.py`` and is selected via the ``UberAPIAlt``
key in CLASS_FILE_PATH_MAPPING.
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
