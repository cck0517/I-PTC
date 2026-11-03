from copy import deepcopy
from typing import Any, Dict, List

DEFAULT_STATE: Dict[str, Any] = {
    "employees": [],
    "reporting_history": [],
}


class OrgChartAPI:
    """
    In-memory org chart backend. Scenario-loaded state; no external database.

    ``list_employees`` plus ``get_reporting_history`` (including ``start_date`` /
    ``end_date`` on each row) is enough for callers to compute shared supervisors,
    unique employee counts, and concurrent reporting intervals without extra tools.
    """

    def __init__(self) -> None:
        self.employees: List[Dict[str, str]]
        self.reporting_history: List[Dict[str, str]]
        self._api_description = (
            "This tool belongs to the OrgChartAPI, which exposes read-only access "
            "to employees and their historical reporting relationships."
        )

    def _load_scenario(self, scenario: dict, long_context: bool = False) -> None:
        default_copy = deepcopy(DEFAULT_STATE)
        self.employees = scenario.get("employees", default_copy["employees"])
        self.reporting_history = scenario.get(
            "reporting_history", default_copy["reporting_history"]
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, OrgChartAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    def list_employees(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Return all employees in the org chart dataset.

        Returns:
            employees: List of dicts with employee_id and employee_name.
        """
        return {"employees": list(self.employees)}

    def get_reporting_history(self, employee_id: str) -> Dict[str, Any]:
        """
        Return all historical reporting rows where the given employee was the report.

        Args:
            employee_id: The subordinate employee id.

        Returns:
            employee_id: Same as input.
            reporting_history: Matching rows (may be empty if unknown id).
        """
        history = [r for r in self.reporting_history if r["employee_id"] == employee_id]
        return {"employee_id": employee_id, "reporting_history": history}
