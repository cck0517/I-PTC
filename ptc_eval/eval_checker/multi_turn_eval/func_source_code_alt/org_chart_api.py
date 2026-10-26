"""Variant of OrgChartAPI used by the intermediate experiment.

Injects a single transient failure on the 13th invocation of
``get_reporting_history``.  The class is named ``OrgChartAPI`` so
tracebacks surfaced to the model do not broadcast that this is a
variant; the module lives at ``func_source_code_alt/org_chart_api.py``
and is selected via the ``OrgChartAPIAlt`` key in
CLASS_FILE_PATH_MAPPING.
"""

from __future__ import annotations

from typing import Any, Dict

from ..func_source_code.org_chart_api import OrgChartAPI as _BaseOrgChartAPI
from ._flaky import check_flaky


class OrgChartError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:
        return f"OrgChartError: {self.args[0]}"


class OrgChartAPI(_BaseOrgChartAPI):
    def get_reporting_history(self, employee_id: str) -> Dict[str, Any]:
        if check_flaky(self, "get_reporting_history"):
            raise OrgChartError(
                "INTERNAL_ERROR",
                "Upstream reporting-history service returned an internal error. Please try again.",
                employee_id=employee_id,
            )
        return super().get_reporting_history(employee_id)
