"""DoorDash variant with configurable transient menu failures."""

from __future__ import annotations

from typing import Any, Dict, List

from ..func_source_code.doordash_api import DoorDashAPI as _BaseDoorDashAPI
from ._flaky import check_flaky


class DoorDashError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:
        return f"DoorDashError: {self.args[0]}"


class DoorDashAPI(_BaseDoorDashAPI):
    def get_restaurant_menu(self, restaurant_id: str) -> Dict[str, List[Dict]]:
        if check_flaky(self, "get_restaurant_menu"):
            raise DoorDashError(
                "INTERNAL_ERROR",
                "Upstream menu service returned an internal error. Please try again.",
                restaurant_id=restaurant_id,
            )
        return super().get_restaurant_menu(restaurant_id)
