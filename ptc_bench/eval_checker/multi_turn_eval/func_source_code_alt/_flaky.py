"""Per-instance failure injection for alternate APIs.

``_instantiate_api`` sets ``_flaky_thresholds`` from task config: each method
maps to one call number or a list of call numbers on which to fail. An explicit
dict disables the legacy 13th-call default, including for unlisted methods.
"""
from __future__ import annotations

from typing import Any

_DEFAULT_FAIL_AT = 13


def check_flaky(
    api: Any,
    method_name: str,
    *,
    use_legacy_default: bool = True,
) -> bool:
    """Count this call and check its failure threshold.

    Set ``use_legacy_default=False`` on added overrides so unconfigured legacy
    tasks still fail only on the original method's 13th call.
    """
    counts = getattr(api, "_call_counts", None)
    if counts is None:
        counts = {}
        api._call_counts = counts
    counts[method_name] = counts.get(method_name, 0) + 1
    n = counts[method_name]

    thresholds = getattr(api, "_flaky_thresholds", None)
    if thresholds is None:
        return use_legacy_default and n == _DEFAULT_FAIL_AT

    spec = thresholds.get(method_name)
    if spec is None:
        return False
    if isinstance(spec, int):
        return n == spec
    return n in set(spec)
