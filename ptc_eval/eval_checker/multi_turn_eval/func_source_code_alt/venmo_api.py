"""Variant of VenmoAPI used by the intermediate experiment.

Injects a single transient failure on the 13th invocation of
``get_transaction_history``.  The class is named ``VenmoAPI`` so
tracebacks surfaced to the model do not broadcast that this is a
variant; the module lives at ``func_source_code_alt/venmo_api.py``
and is selected via the ``VenmoAPIAlt`` key in
CLASS_FILE_PATH_MAPPING.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..func_source_code.venmo_api import VenmoAPI as _BaseVenmoAPI
from ._flaky import check_flaky


class VenmoError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:
        return f"VenmoError: {self.args[0]}"


class VenmoAPI(_BaseVenmoAPI):
    def get_transaction_history(
        self, start_date: str = None, end_date: str = None
    ) -> Dict[str, List[Dict]]:
        if check_flaky(self, "get_transaction_history"):
            raise VenmoError(
                "INTERNAL_ERROR",
                "Upstream transaction service returned an internal error. Please try again.",
                start_date=start_date,
                end_date=end_date,
            )
        return super().get_transaction_history(start_date=start_date, end_date=end_date)
