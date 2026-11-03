"""Venmo variant with configurable transient transaction-history failures."""

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
