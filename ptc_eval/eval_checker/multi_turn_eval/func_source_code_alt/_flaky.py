"""Shared per-instance flakiness helper for alt API variants.

Each alt subclass overrides one method and inserts a single call:

    if check_flaky(self, "method_name"):
        raise SomeError(...)

The trigger reads ``self._flaky_thresholds`` (a per-instance dict stamped by
``_instantiate_api`` from the task's ``flakiness_config`` field).  When the
attribute is absent, the legacy default (fail on the 13th call) is used so
the original 5 intermediate queries keep working without explicit config.

A threshold value may be:
  - an ``int``  — fail on that single call number
  - a ``list[int]`` — fail on each listed call number
  - missing (key not in dict) — never fail this method

When ``_flaky_thresholds`` is set to a dict (even an empty one), only methods
explicitly listed there fail; the legacy default is *not* applied.
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
    """Increment per-method counter and return True if this call should fail.

    ``use_legacy_default`` controls behavior when the task has no explicit
    ``flakiness_config``.  Set to ``False`` for method overrides added to an
    alt class *after* the original override, so legacy tasks (which expect
    fail-on-13th only on the *first* override) don't accidentally start
    failing on the newly-added method too.  Defaults to ``True`` to preserve
    behavior of methods that have always been overridden.
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
