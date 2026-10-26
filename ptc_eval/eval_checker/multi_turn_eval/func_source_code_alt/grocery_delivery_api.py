"""Variant of GroceryDeliveryAPI used by the intermediate experiment.

Injects a single transient failure on the 13th invocation of
``get_store_inventory``.  The class is named ``GroceryDeliveryAPI`` so
tracebacks surfaced to the model do not broadcast that this is a
variant; the module lives at ``func_source_code_alt/grocery_delivery_api.py``
and is selected via the ``GroceryDeliveryAPIAlt`` key in
CLASS_FILE_PATH_MAPPING.
"""

from __future__ import annotations

from typing import Any, Dict

from ..func_source_code.grocery_delivery_api import (
    GroceryDeliveryAPI as _BaseGroceryDeliveryAPI,
)
from ._flaky import check_flaky


class GroceryDeliveryError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:
        return f"GroceryDeliveryError: {self.args[0]}"


class GroceryDeliveryAPI(_BaseGroceryDeliveryAPI):
    def get_store_inventory(self, store_id: str) -> Dict[str, Any]:
        if check_flaky(self, "get_store_inventory"):
            raise GroceryDeliveryError(
                "INTERNAL_ERROR",
                "Upstream inventory service returned an internal error. Please try again.",
                store_id=store_id,
            )
        return super().get_store_inventory(store_id)
