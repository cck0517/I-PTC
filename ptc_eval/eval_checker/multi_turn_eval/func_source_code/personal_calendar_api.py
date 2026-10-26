from copy import deepcopy
from typing import Dict, List, Optional, Union

DEFAULT_STATE = {
    "events": {},
    "current_date": "2026-03-31",
    "event_counter": 0,
}


class PersonalCalendarAPI:
    """
    A class representing a Calendar API for managing events and schedules.

    This class provides methods for creating, updating, deleting events,
    managing attendees, and finding free time slots.
    """

    def __init__(self):
        self.events: Dict[str, Dict]
        self.current_date: str
        self.event_counter: int
        self._api_description = "This tool belongs to the PersonalCalendarAPI, which provides functionality for managing calendar events, scheduling, and attendee coordination."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.events = scenario.get("events", DEFAULT_STATE_COPY["events"])
        self.current_date = scenario.get("current_date", DEFAULT_STATE_COPY["current_date"])
        self.event_counter = scenario.get("event_counter", DEFAULT_STATE_COPY["event_counter"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PersonalCalendarAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    def get_calendar_context(self) -> Dict[str, str]:
        """
        Get the current calendar context including the simulated current date.

        Returns:
            Dict with current_date (str) and total_events (int).
        """
        return {
            "current_date": self.current_date,
            "total_events": len(self.events),
        }

    def create_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        attendees: List[str] = [],
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Create a new calendar event.

        Args:
            title (str): Title of the event.
            start_time (str): Start time of the event (format: 'YYYY-MM-DD HH:MM').
            end_time (str): End time of the event (format: 'YYYY-MM-DD HH:MM').
            description (str): [Optional] Description of the event.
            location (str): [Optional] Location of the event.
            attendees (List[str]): [Optional] List of attendee usernames.
        Returns:
            event_id (str): ID of the created event.
            title (str): Title of the event.
            start_time (str): Start time.
            end_time (str): End time.
            description (str): Description.
            location (str): Location.
            attendees (List[str]): List of attendees.
        """
        self.event_counter += 1
        event_id = f"evt_{self.event_counter}"
        event = {
            "id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "location": location,
            "attendees": list(attendees),
        }
        self.events[event_id] = event
        return event

    def get_events(self, date: str) -> Dict[str, List[Dict]]:
        """
        Get all events for a specific date.

        Args:
            date (str): Date to get events for (format: 'YYYY-MM-DD').
        Returns:
            events (List[Dict]): List of events on the given date.
        """
        result = [
            evt for evt in self.events.values()
            if evt["start_time"].startswith(date)
        ]
        return {"events": result}

    def get_events_range(self, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """
        Get all events within a date range.

        Args:
            start_date (str): Start date of the range (format: 'YYYY-MM-DD').
            end_date (str): End date of the range (format: 'YYYY-MM-DD').
        Returns:
            events (List[Dict]): List of events within the date range.
        """
        result = []
        for evt in self.events.values():
            evt_date = evt["start_time"][:10]
            if start_date <= evt_date <= end_date:
                result.append(evt)
        return {"events": result}

    def delete_event(self, event_id: str) -> Dict[str, bool]:
        """
        Delete a calendar event.

        Args:
            event_id (str): ID of the event to delete.
        Returns:
            delete_status (bool): True if the event was successfully deleted.
        """
        if event_id not in self.events:
            return {"error": f"Event '{event_id}' not found."}
        del self.events[event_id]
        return {"delete_status": True}

    def get_event_details(self, event_id: str) -> Dict:
        """
        Get details of a specific event.

        Args:
            event_id (str): ID of the event to retrieve.
        Returns:
            id (str): Event ID.
            title (str): Event title.
            start_time (str): Start time.
            end_time (str): End time.
            description (str): Description.
            location (str): Location.
            attendees (List[str]): List of attendees.
        """
        if event_id not in self.events:
            return {"error": f"Event '{event_id}' not found."}
        return self.events[event_id]

    def update_event(self, event_id: str, updates: Dict) -> Dict:
        """
        Update an existing calendar event.

        Args:
            event_id (str): ID of the event to update.
            updates (Dict): Dictionary of fields to update (e.g., title, location, description).
        Returns:
            Updated event dictionary with all fields.
        """
        if event_id not in self.events:
            return {"error": f"Event '{event_id}' not found."}
        for key, value in updates.items():
            if key in self.events[event_id] and key != "id":
                self.events[event_id][key] = value
        return self.events[event_id]

    def add_attendee(self, event_id: str, attendee: str) -> Dict[str, bool]:
        """
        Add an attendee to a calendar event.

        Args:
            event_id (str): ID of the event.
            attendee (str): Username of the attendee to add.
        Returns:
            attendee_added (bool): True if the attendee was added successfully.
        """
        if event_id not in self.events:
            return {"error": f"Event '{event_id}' not found."}
        if attendee in self.events[event_id]["attendees"]:
            return {"attendee_added": False}
        self.events[event_id]["attendees"].append(attendee)
        return {"attendee_added": True}

    def find_free_slots(self, date: str, duration_minutes: int) -> Dict[str, List[Dict[str, str]]]:
        """
        Find free time slots on a given date.

        Args:
            date (str): Date to find free slots for (format: 'YYYY-MM-DD').
            duration_minutes (int): Required duration in minutes.
        Returns:
            free_slots (List[Dict]): List of free time slots, each with 'start' and 'end' times.
        """
        day_events = sorted(
            [e for e in self.events.values() if e["start_time"].startswith(date)],
            key=lambda x: x["start_time"],
        )

        free_slots = []
        day_start = f"{date} 08:00"
        day_end = f"{date} 22:00"

        current_time = day_start
        for evt in day_events:
            if evt["start_time"] > current_time:
                start_h, start_m = int(current_time[11:13]), int(current_time[14:16])
                end_h, end_m = int(evt["start_time"][11:13]), int(evt["start_time"][14:16])
                gap = (end_h - start_h) * 60 + (end_m - start_m)
                if gap >= duration_minutes:
                    free_slots.append({"start": current_time, "end": evt["start_time"]})
            if evt["end_time"] > current_time:
                current_time = evt["end_time"]

        if current_time < day_end:
            start_h, start_m = int(current_time[11:13]), int(current_time[14:16])
            end_h, end_m = int(day_end[11:13]), int(day_end[14:16])
            gap = (end_h - start_h) * 60 + (end_m - start_m)
            if gap >= duration_minutes:
                free_slots.append({"start": current_time, "end": day_end})

        return {"free_slots": free_slots}
