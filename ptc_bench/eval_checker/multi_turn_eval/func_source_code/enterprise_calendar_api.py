from copy import deepcopy
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

DEFAULT_STATE: Dict[str, Any] = {
    "metadata": {},
    "employees": [],
    "events": [],
}


class EnterpriseCalendarAPI:
    """
    Mock calendar: employees live in their own timezones; each event's start/end are
    wall times in that employee's timezone. RSVP is stored per event. Supports booking
    new events for multiple attendees with conflict detection.
    """

    def __init__(self) -> None:
        self.metadata: Dict[str, Any]
        self.employees: List[Dict[str, str]]
        self.events: List[Dict[str, Any]]
        self._api_description = (
            "This tool belongs to the EnterpriseCalendarAPI mock: simulated clock/current week, "
            "per-employee timezones, local events with RSVP, schedule lookup by name or current "
            "user, timezone conversion, and multi-attendee event booking."
        )

    def _load_scenario(self, scenario: dict, long_context: bool = False) -> None:
        d = deepcopy(DEFAULT_STATE)
        self.metadata = dict(scenario.get("metadata", d["metadata"]))
        self.employees = list(scenario.get("employees", d["employees"]))
        self.events = list(scenario.get("events", d["events"]))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, EnterpriseCalendarAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    def _employee_by_name(self, name: str) -> Optional[Dict[str, str]]:
        n = name.strip().casefold()
        for row in self.employees:
            if row.get("display_name", "").strip().casefold() == n:
                return row
        return None

    def get_calendar_context(self) -> Dict[str, Any]:
        """
        Return simulated "now" and the current calendar week (inclusive dates).
        Meeting length, attendees, and working hours come from the user task, not here.
        Also returns current_user_name when set in metadata.
        """
        return dict(self.metadata)

    def get_user_schedule(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: return the current user's calendar events for the given
        date range.  The current user's display name is read from
        ``metadata["current_user_name"]``; returns an error dict when that key is
        absent.  The response structure is identical to
        ``get_schedule_by_employee_name``.
        """
        user_name = self.metadata.get("current_user_name")
        if not user_name:
            return {
                "error": "current_user_name is not set in metadata.",
                "events": [],
            }
        return self.get_schedule_by_employee_name(user_name, start_date, end_date)

    def get_schedule_by_employee_name(
        self,
        employee_name: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Look up one employee by display name (case-insensitive) and return their events
        whose intervals overlap [start_date, end_date] in that employee's local calendar
        (inclusive of both dates, end-of-day in local time).
        """
        emp = self._employee_by_name(employee_name)
        if emp is None:
            return {"error": f"No employee named {employee_name!r}.", "events": []}

        d0 = date.fromisoformat(start_date)
        d1 = date.fromisoformat(end_date)
        tz_name = emp["timezone"]
        tz = ZoneInfo(tz_name)
        range_start_local = datetime.combine(d0, time.min, tzinfo=tz)
        range_end_local = datetime.combine(d1, time(23, 59, 59), tzinfo=tz) + timedelta(seconds=1)

        out: List[Dict[str, Any]] = []
        for ev in self.events:
            if ev["employee_id"] != emp["id"]:
                continue
            es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
            ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
            if es < range_end_local and ee > range_start_local:
                out.append(
                    {
                        "start_local": ev["start"],
                        "end_local": ev["end"],
                        "timezone": tz_name,
                        "title": ev["title"],
                        "rsvp": ev["rsvp"],
                    }
                )

        return {
            "employee_id": emp["id"],
            "display_name": emp["display_name"],
            "timezone": tz_name,
            "start_date": start_date,
            "end_date": end_date,
            "events": out,
        }

    def convert_datetime_between_timezones(
        self,
        local_datetime_iso: str,
        source_timezone: str,
        target_timezone: str,
    ) -> Dict[str, Any]:
        """
        Interpret ``local_datetime_iso`` as a naive local wall time in ``source_timezone``,
        then return the same instant as UTC and as local time in ``target_timezone``.
        IANA zone names only (e.g. America/New_York, Europe/London).
        """
        naive = datetime.fromisoformat(local_datetime_iso)
        utc_dt = naive.replace(tzinfo=ZoneInfo(source_timezone)).astimezone(ZoneInfo("UTC"))
        target_dt = utc_dt.astimezone(ZoneInfo(target_timezone))
        return {
            "source_timezone": source_timezone,
            "target_timezone": target_timezone,
            "source_local": local_datetime_iso,
            "utc_iso": utc_dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "target_local_iso": target_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def book_event(
        self,
        attendee_names: List[str],
        start_utc: str,
        duration_minutes: int,
        title: str = "Meeting",
    ) -> Dict[str, Any]:
        """
        Book an event for two or more attendees at a given UTC start time.

        ``start_utc`` is an ISO-8601 UTC timestamp (trailing 'Z' accepted and
        stripped).  Events are created on every attendee's calendar as naive
        wall-times in their respective timezones.  Returns an error dict when
        any name is unknown or the proposed slot conflicts with an existing
        event for any attendee.  On success, returns local times for every
        attendee.
        """
        if len(attendee_names) < 2:
            return {"status": "error", "message": "at least two attendee_names are required."}

        resolved: List[Dict[str, str]] = []
        for name in attendee_names:
            emp = self._employee_by_name(name)
            if emp is None:
                return {"status": "error", "message": f"No employee named {name!r}."}
            resolved.append(emp)

        raw = start_utc.rstrip("Z")
        start_dt_utc = datetime.fromisoformat(raw).replace(tzinfo=ZoneInfo("UTC"))
        end_dt_utc = start_dt_utc + timedelta(minutes=duration_minutes)
        fmt = "%Y-%m-%dT%H:%M:%S"

        attendee_locals: List[Dict[str, Any]] = []
        for emp in resolved:
            tz = ZoneInfo(emp["timezone"])
            local_start = start_dt_utc.astimezone(tz)
            local_end = end_dt_utc.astimezone(tz)
            for ev in self.events:
                if ev["employee_id"] != emp["id"]:
                    continue
                es = datetime.fromisoformat(ev["start"]).replace(tzinfo=tz)
                ee = datetime.fromisoformat(ev["end"]).replace(tzinfo=tz)
                if local_start < ee and local_end > es:
                    return {
                        "status": "error",
                        "message": (
                            f"Conflict for {emp['display_name']}: proposed "
                            f"{local_start.strftime(fmt)}\u2013"
                            f"{local_end.strftime(fmt)} {emp['timezone']} "
                            f"overlaps with '{ev['title']}' "
                            f"({ev['start']}\u2013{ev['end']})."
                        ),
                    }
            attendee_locals.append({
                "display_name": emp["display_name"],
                "timezone": emp["timezone"],
                "start_local": local_start.strftime(fmt),
                "end_local": local_end.strftime(fmt),
            })

        for emp, local_info in zip(resolved, attendee_locals):
            self.events.append({
                "employee_id": emp["id"],
                "start": local_info["start_local"],
                "end": local_info["end_local"],
                "title": title,
                "rsvp": "accepted",
            })

        return {
            "status": "success",
            "booking": {
                "attendees": [a["display_name"] for a in attendee_locals],
                "start_utc": start_dt_utc.strftime(fmt) + "Z",
                "end_utc": end_dt_utc.strftime(fmt) + "Z",
                "duration_minutes": duration_minutes,
                "attendee_locals": attendee_locals,
                "title": title,
            },
        }
