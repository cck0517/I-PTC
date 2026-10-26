"""Variant of DoorDashAPI used by the intermediate experiment.

Injects a single transient failure on the 13th invocation of
``get_restaurant_menu``.  The class is named ``DoorDashAPI`` so
tracebacks surfaced to the model do not broadcast that this is a
variant; the module lives at ``func_source_code_alt/doordash_api.py``
and is selected via the ``DoorDashAPIAlt`` key in
CLASS_FILE_PATH_MAPPING.
"""

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
