"""Shared helpers for ptc_mt_calendar_*_0 verifiers."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


def employees_by_id(cfg: dict) -> dict[str, dict]:
    return {e["id"]: e for e in cfg["employees"]}


def events_by_employee(cfg: dict) -> dict[str, list[dict]]:
    """{employee_id: [event, ...] in input order}."""
    out: dict[str, list[dict]] = defaultdict(list)
    for ev in cfg["events"]:
        out[ev["employee_id"]].append(ev)
    return dict(out)


def event_duration_minutes(ev: dict) -> int:
    s = datetime.fromisoformat(ev["start"])
    e = datetime.fromisoformat(ev["end"])
    return int((e - s).total_seconds() // 60)


def to_utc(local_iso: str, tz: ZoneInfo) -> datetime:
    return datetime.fromisoformat(local_iso).replace(tzinfo=tz).astimezone(UTC)


def fmt_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_window(cfg: dict) -> tuple[date, date]:
    md = cfg.get("metadata", {})
    return (
        date.fromisoformat(md["current_window_start"]),
        date.fromisoformat(md["current_window_end"]),
    )


def weekdays_in(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def busy_intervals_utc_for(
    cfg: dict, employee_id: str, tz: ZoneInfo,
) -> list[tuple[datetime, datetime]]:
    """All events for an employee converted to (start_utc, end_utc) tuples."""
    out = []
    for ev in cfg["events"]:
        if ev["employee_id"] != employee_id:
            continue
        s = to_utc(ev["start"], tz)
        e = to_utc(ev["end"], tz)
        out.append((s, e))
    return out


def merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping/touching intervals."""
    if not intervals:
        return []
    s = sorted(intervals)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def free_blocks_in_window(
    busy: list[tuple[datetime, datetime]],
    win_start: datetime,
    win_end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Subtract merged-busy from [win_start, win_end). Returns the free chunks."""
    clipped = []
    for a, b in busy:
        a2 = max(a, win_start)
        b2 = min(b, win_end)
        if a2 < b2:
            clipped.append((a2, b2))
    merged = merge_intervals(clipped)
    free = []
    cursor = win_start
    for a, b in merged:
        if cursor < a:
            free.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < win_end:
        free.append((cursor, win_end))
    return free


def employee_id_by_display_name(cfg: dict, name: str) -> str | None:
    n = name.strip().casefold()
    for e in cfg["employees"]:
        if e["display_name"].strip().casefold() == n:
            return e["id"]
    return None


def approx(a, b, tol: float = 0.0005) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False
