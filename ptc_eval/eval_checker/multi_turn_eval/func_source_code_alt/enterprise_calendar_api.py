"""Variant of EnterpriseCalendarAPI used by the intermediate experiment.

Injects a single transient failure on the 13th invocation of
``get_schedule_by_employee_name``.  The class is named
``EnterpriseCalendarAPI`` so tracebacks surfaced to the model do not
broadcast that this is a variant; the module lives at
``func_source_code_alt/enterprise_calendar_api.py`` and is selected via
the ``EnterpriseCalendarAPIAlt`` key in CLASS_FILE_PATH_MAPPING.
"""

from __future__ import annotations

from typing import Any, Dict

from ..func_source_code.enterprise_calendar_api import (
    EnterpriseCalendarAPI as _BaseEnterpriseCalendarAPI,
)
from ._flaky import check_flaky


class EnterpriseCalendarError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:
        return f"EnterpriseCalendarError: {self.args[0]}"


class EnterpriseCalendarAPI(_BaseEnterpriseCalendarAPI):
    def get_schedule_by_employee_name(
        self,
        employee_name: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        if check_flaky(self, "get_schedule_by_employee_name"):
            raise EnterpriseCalendarError(
                "INTERNAL_ERROR",
                "Upstream schedule service returned an internal error. Please try again.",
                employee_name=employee_name,
            )
        return super().get_schedule_by_employee_name(employee_name, start_date, end_date)
