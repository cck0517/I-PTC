"""Shared helpers for ptc_mt_org_chart_*_0 verifiers."""
from __future__ import annotations

from collections import defaultdict
from datetime import date


OPEN_END = date(9999, 12, 31)


def parse_date(s: str) -> date:
    """Parse YYYY-MM-DD; empty string treated as open-ended (9999-12-31)."""
    if not s:
        return OPEN_END
    return date.fromisoformat(s)


def name_by_id(cfg: dict) -> dict[str, str]:
    return {e["employee_id"]: e["employee_name"] for e in cfg["employees"]}


def active_manager_on(history: list[dict], snapshot: date) -> dict[str, tuple[date, str]]:
    """Return {employee_id: (start_date, manager_id)} on the snapshot,
    using the latest-start_date disambiguation rule for matrix reporting."""
    out: dict[str, tuple[date, str]] = {}
    for r in history:
        s = parse_date(r["start_date"])
        e = parse_date(r["end_date"])
        if s <= snapshot <= e:
            cur = out.get(r["employee_id"])
            if cur is None or s > cur[0]:
                out[r["employee_id"]] = (s, r["manager_id"])
    return out


def chain_to_root(eid: str, active_mgr: dict[str, tuple[date, str]]) -> list[str]:
    """Root-first list of employee_ids from CEO to *eid*."""
    out = []
    cur = eid
    seen = set()
    while cur in active_mgr and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = active_mgr[cur][1]
    out.append(cur)  # the root (no active reporting row of its own)
    return list(reversed(out))


def chain_lengths(active_mgr: dict[str, tuple[date, str]]) -> dict[str, int]:
    """Return {employee_id: number_of_edges_to_root}."""
    cache: dict[str, int] = {}

    def rec(eid: str) -> int:
        if eid in cache:
            return cache[eid]
        if eid not in active_mgr:
            cache[eid] = 0
            return 0
        cache[eid] = 1 + rec(active_mgr[eid][1])
        return cache[eid]

    for e in list(active_mgr):
        rec(e)
    return cache


def direct_report_counts(active_mgr: dict[str, tuple[date, str]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for _, (_, mid) in active_mgr.items():
        out[mid] += 1
    return dict(out)


def approx(a, b, tol=0.05) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False
