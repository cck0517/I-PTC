"""
PTC code interpreter tool + individual API tool definitions.

Exposes:
  1. A code-execution tool ``ptc_execute`` — the model writes Python snippets
     that call methods on pre-loaded API objects.
  2. Individual API method tools (e.g. ``get_calendar_context``,
     ``list_employees``) loaded from the JSON func-doc files, so every API
     method is also available as a standalone tool the model can call directly.

This mirrors the Anthropic PTC format where all dummy-API methods are
presented as tools alongside the code-execution tool.
"""

from __future__ import annotations

import copy
import ctypes
import io
import json
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any

from agent_eval.agent import ToolDef


# ---------------------------------------------------------------------------
# Thread-safe stdout/stderr capture for concurrent cell execution
# ---------------------------------------------------------------------------
#
# ``contextlib.redirect_stdout`` mutates global ``sys.stdout``. When multiple
# workers run ``ptc_execute`` / ``ptc_cell`` concurrently (as with
# ``--workers > 1``), those redirects race and stomp on each other: one
# worker's captured output can end up in another worker's buffer, and the
# restored-global fallback can briefly point at an arbitrary worker's
# buffer instead of the real stdout. This is not just sloppy capture —
# interleaved writes to a closed/moved ``sys.stdout`` from C extensions
# can produce undefined behavior.
#
# Fix: install ONE global proxy at module import that routes every write
# to a per-thread ``threading.local()`` buffer when set, else to the
# original stdout/stderr. ``_run_in`` then sets/clears its thread's buffer
# rather than mutating the global. No more cross-thread pollution.

_stdout_local = threading.local()
_stderr_local = threading.local()


class _ThreadLocalStream:
    """File-like proxy that routes writes to a per-thread buffer if set."""

    __slots__ = ("_local", "_fallback")

    def __init__(self, local_state: threading.local, fallback: Any) -> None:
        self._local = local_state
        self._fallback = fallback

    def _target(self) -> Any:
        buf = getattr(self._local, "buf", None)
        return buf if buf is not None else self._fallback

    def write(self, s: str) -> int:
        return self._target().write(s)

    def flush(self) -> None:
        try:
            self._target().flush()
        except Exception:
            pass

    def writelines(self, lines: Any) -> None:
        self._target().writelines(lines)

    def isatty(self) -> bool:
        try:
            return self._target().isatty()
        except Exception:
            return False

    def __getattr__(self, name: str) -> Any:
        # Fallback for any attribute not defined above — always check the
        # target at the time of access, respecting the thread-local state.
        return getattr(self._target(), name)


# Install the proxies once per process. Safe against re-imports (e.g.,
# reload or repeated CLI launches within the same interpreter).
if not isinstance(sys.stdout, _ThreadLocalStream):
    sys.stdout = _ThreadLocalStream(_stdout_local, sys.stdout)
if not isinstance(sys.stderr, _ThreadLocalStream):
    sys.stderr = _ThreadLocalStream(_stderr_local, sys.stderr)


class _thread_local_redirect:
    """Context manager replacement for ``redirect_stdout``/``redirect_stderr``.

    Sets thread-local stdout/stderr buffers. Only the current thread's
    writes are captured — other threads continue to route their writes
    to their own buffers (or to the real terminal if they have none).
    """

    __slots__ = ("_stdout_buf", "_stderr_buf", "_prev_stdout", "_prev_stderr")

    def __init__(self, stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> None:
        self._stdout_buf = stdout_buf
        self._stderr_buf = stderr_buf
        self._prev_stdout: Any = None
        self._prev_stderr: Any = None

    def __enter__(self) -> "_thread_local_redirect":
        self._prev_stdout = getattr(_stdout_local, "buf", None)
        self._prev_stderr = getattr(_stderr_local, "buf", None)
        _stdout_local.buf = self._stdout_buf
        _stderr_local.buf = self._stderr_buf
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._prev_stdout is None:
            try:
                del _stdout_local.buf
            except AttributeError:
                pass
        else:
            _stdout_local.buf = self._prev_stdout
        if self._prev_stderr is None:
            try:
                del _stderr_local.buf
            except AttributeError:
                pass
        else:
            _stderr_local.buf = self._prev_stderr

# ---------------------------------------------------------------------------
# Type mapping: func-doc "dict"/"list"/etc. → JSON-Schema types
# ---------------------------------------------------------------------------

_GORILLA_TO_OPENAPI: dict[str, str] = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
}


def _normalize_schema_type(raw_type: Any) -> str:
    """Resolve a JSON-Schema ``type`` field that may be a list.

    Accepts list-form nullability (``["string", "null"]``) by picking the
    first non-``"null"`` string. Falls back to ``"string"`` for anything
    unrecognized so the lookup in ``_GORILLA_TO_OPENAPI`` can never raise
    ``TypeError: unhashable type: 'list'`` on LLM-generated schemas.
    """
    if isinstance(raw_type, list):
        for t in raw_type:
            if isinstance(t, str) and t.lower() != "null":
                return t
        return "string"
    if isinstance(raw_type, str):
        return raw_type
    return "string"


def _cast_properties(properties: dict) -> dict:
    """Recursively convert func-doc types to JSON-Schema types."""
    out = {}
    for key, value in properties.items():
        prop = dict(value)
        raw_type = _normalize_schema_type(prop.get("type", "string"))
        prop["type"] = _GORILLA_TO_OPENAPI.get(raw_type, "string")

        if prop["type"] in ("array", "object"):
            if "properties" in prop:
                prop["properties"] = _cast_properties(prop["properties"])
            if "items" in prop:
                items = dict(prop["items"])
                items_type = _normalize_schema_type(items.get("type", "string"))
                items["type"] = _GORILLA_TO_OPENAPI.get(items_type, "string")
                if items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast_properties(items["properties"])
                prop["items"] = items
            elif prop["type"] == "array":
                # OpenAI's function-calling API rejects ``type: array``
                # without an ``items`` schema. Provide a permissive
                # fallback so a schema that forgot ``items`` can't
                # crater the whole run's tool-list registration.
                prop["items"] = {"type": "object"}
        out[key] = prop
    return out


def _func_doc_to_tool_def(func_doc: dict, api_prefix: str) -> ToolDef:
    """
    Convert a single func-doc JSON entry into a provider-agnostic ToolDef.

    The tool name is prefixed with the API name (e.g. ``enterprise_calendar_api.book_event``)
    so the executor can dispatch to the right API object.
    """
    doc = copy.deepcopy(func_doc)
    name = f"{api_prefix}.{doc['name']}"
    description = doc.get("description", "")

    # Append response schema to description so the model knows return format
    if "response" in doc:
        description += f"\n\nResponse schema: {json.dumps(doc['response'])}"

    params = doc.get("parameters", {})
    properties = _cast_properties(params.get("properties", {}))
    required = params.get("required", [])

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return ToolDef(name=name, description=description, parameters=schema)


def load_api_tool_defs(involved_classes: list[str]) -> list[ToolDef]:
    """
    Load func-doc JSON files for the given API classes and return ToolDefs
    for every individual API method.
    """
    from ptc_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
    from ptc_eval.constants.executable_backend_config import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    from ptc_eval.utils import load_file

    tool_defs: list[ToolDef] = []
    for cls_name in involved_classes:
        doc_file = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls_name)
        if doc_file is None:
            continue
        doc_path = MULTI_TURN_FUNC_DOC_PATH / doc_file
        if not doc_path.exists():
            continue
        docs = load_file(doc_path)
        prefix = _CLASS_TO_PREFIX[cls_name]
        for doc in docs:
            tool_defs.append(_func_doc_to_tool_def(doc, prefix))
    return tool_defs



# ---------------------------------------------------------------------------
# Build the PTC tool description dynamically from the JSON func docs
# ---------------------------------------------------------------------------

# All 10 APIs use the same minimal placeholder so the system prompt is
# symmetric across them — no per-API method listings leak into the agent's
# context. Method discovery happens via:
#   - tool inspection ON (--tool-inspection): the agent runs ``<api>??``
#     inside a cell to list methods (see ``_TOOL_INSPECTION_SUFFIX``).
#   - tool inspection OFF: the per-method tool defs are in the agent's tool
#     schema directly (added to the OpenAI/Anthropic ``tools`` list).
# Earlier this dict had handcrafted method listings for 6 of 10 APIs and
# placeholders for the other 4 — that asymmetry gave tasks using the listed
# APIs an upfront prompt advantage and confounded ablations of inspection
# vs. no-inspection. Now uniform.
_API_DOCS: dict[str, str] = {
    "EnterpriseCalendarAPI": """\
### ``enterprise_calendar_api`` (EnterpriseCalendarAPI)
See individual tool definitions for available methods.""",

    "PersonalCalendarAPI": """\
### ``personal_calendar_api`` (PersonalCalendarAPI)
See individual tool definitions for available methods.""",

    "OrgChartAPI": """\
### ``org_chart_api`` (OrgChartAPI)
See individual tool definitions for available methods.""",

    "GroceryDeliveryAPI": """\
### ``grocery_api`` (GroceryDeliveryAPI)
See individual tool definitions for available methods.""",

    "YelpAPI": """\
### ``yelp_api`` (YelpAPI)
See individual tool definitions for available methods.""",

    "GoogleMapsAPI": """\
### ``maps_api`` (GoogleMapsAPI)
See individual tool definitions for available methods.""",

    "UberAPI": """\
### ``uber_api`` (UberAPI)
See individual tool definitions for available methods.""",

    "DoorDashAPI": """\
### ``doordash_api`` (DoorDashAPI)
See individual tool definitions for available methods.""",

    "SpotifyAPI": """\
### ``spotify_api`` (SpotifyAPI)
See individual tool definitions for available methods.""",

    "VenmoAPI": """\
### ``venmo_api`` (VenmoAPI)
See individual tool definitions for available methods.""",

}


def _build_ptc_description(
    involved_classes: list[str],
    tool_inspection: bool = False,
) -> str:
    """Build a PTC tool description that only documents the APIs in this task.

    When ``tool_inspection`` is True the per-method tool defs have been
    stripped from the agent's schema, so the API block lists only the
    variable names and their class names — the ``_API_DOCS`` entries would
    otherwise point the agent at "individual tool definitions" it cannot
    see. This mirrors ``_build_ptc_cell_description``, which has always
    rendered class-name-only sections, so the two code-execution
    conditions receive equivalent guidance under inspection.

    With ``tool_inspection`` False the returned string is unchanged.
    """
    api_sections = []
    for cls in involved_classes:
        canonical = _display_class(cls)
        if tool_inspection and canonical in _CLASS_TO_PREFIX:
            # Class name only — methods are discovered via ``??``.
            api_sections.append(
                f"### ``{_CLASS_TO_PREFIX.get(canonical, cls)}`` "
                f"({canonical})"
            )
        elif canonical in _API_DOCS:
            api_sections.append(_API_DOCS[canonical])

    api_list = "\n\n".join(api_sections)
    var_names = ", ".join(
        _CLASS_TO_PREFIX.get(cls, cls) for cls in involved_classes
        if cls in _CLASS_TO_PREFIX
    )

    has_calendar = any(c in involved_classes for c in
                       ["EnterpriseCalendarAPI", "PersonalCalendarAPI"])
    date_note = """
A string variable ``CURRENT_DATE`` is pre-loaded with the scenario's simulated \
current date (e.g. "2026-03-31"). Use this instead of ``date.today()`` or \
``datetime.now()`` for any date arithmetic.""" if has_calendar else ""

    # Mirrors the cell builder's inspection block, with "cell" renamed to
    # "snippet" and every persistence claim removed (that is the
    # statefulness treatment, not the inspection one).
    #
    # The explicit "bare statement, don't wrap in print()" line has no
    # literal counterpart in the cell block, and is deliberate. ``??`` is
    # IPython syntax; the cell description opens with "behaves like a cell
    # in a Jupyter notebook", which tells the model it is in an
    # environment where a bare ``??`` auto-displays. The stateless
    # description cannot say that without importing persistence. Without
    # the hint the model writes ``print(api??)`` -> SyntaxError: measured
    # on the first inspection attempt of 83% of stateless tasks vs 4% of
    # cell tasks. The hint restores equal usability of the inspection
    # mechanism; it conveys no information about statefulness.
    inspect_note = """
**Inspect before you call.** You do NOT know which methods each API \
exposes until you inspect it — always run the inspection first and \
read the method's description and parameters before calling it. \
``??`` is IPython-style inspection syntax: write it as a bare \
statement on its own line — it prints its own output. Do NOT wrap it \
in ``print(...)``.

```python
# snippet 1 — inspect what's available
<api>??                        # list methods on this API class
<api>.<method>??               # read its description + parameters
```

```python
# snippet 2 — only now make the real call, using the exact method name
# and parameter names you learned from inspection
result = <api>.<method>(param1=..., param2=...)
print(type(result), result)
```""" if tool_inspection else ""

    default = f"""\
Execute a Python code snippet in a sandboxed environment with the following \
API objects pre-loaded as local variables:

{api_list}

## How to use

Write a Python code snippet as the ``code`` argument. Call methods using \
``<variable>.<method>(arg1=..., arg2=...)``. The variables available in this \
task are: {var_names}.
{date_note}{inspect_note}

To see data in your next turn, **print()** it. Whatever is printed to stdout \
is returned as the tool-call output and becomes visible to you on the \
next turn. For structured values, ``print(json.dumps(obj))`` works well.

**Important**: Check the return type of each API call before using ``.get()``. \
Some methods return a list directly, others return a dict wrapping a list. \
Print the raw response first if unsure.

Standard library modules (json, collections, datetime, itertools) are available."""
    return _resolve(
        "ptc_execute_description", default,
        api_sections=api_list, var_names=var_names, date_note=date_note,
        inspect_note=inspect_note,
    )


# Map class name → variable name the executor uses (module-level for reuse)
_CLASS_TO_PREFIX: dict[str, str] = {
    "EnterpriseCalendarAPI": "enterprise_calendar_api",
    "OrgChartAPI": "org_chart_api",
    "YelpAPI": "yelp_api",
    "GoogleMapsAPI": "maps_api",
    "UberAPI": "uber_api",
    "GroceryDeliveryAPI": "grocery_api",
    "PersonalCalendarAPI": "personal_calendar_api",
    "DoorDashAPI": "doordash_api",
    "SpotifyAPI": "spotify_api",
    "VenmoAPI": "venmo_api",
    # Alt variants share the same variable names and tool-name prefixes as
    # their base classes so the model sees identical tool schemas.
    "YelpAPIAlt": "yelp_api",
    "SpotifyAPIAlt": "spotify_api",
    "UberAPIAlt": "uber_api",
    "GroceryDeliveryAPIAlt": "grocery_api",
    "OrgChartAPIAlt": "org_chart_api",
    "EnterpriseCalendarAPIAlt": "enterprise_calendar_api",
    "DoorDashAPIAlt": "doordash_api",
    "VenmoAPIAlt": "venmo_api",
}

# Render these keys as their base-class name when building model-facing text
# (system prompt, tool descriptions). Keeps the alt variants indistinguishable
# from the main classes in anything the model reads.
_CLASS_DISPLAY_NAME: dict[str, str] = {
    "YelpAPIAlt": "YelpAPI",
    "SpotifyAPIAlt": "SpotifyAPI",
    "UberAPIAlt": "UberAPI",
    "GroceryDeliveryAPIAlt": "GroceryDeliveryAPI",
    "OrgChartAPIAlt": "OrgChartAPI",
    "EnterpriseCalendarAPIAlt": "EnterpriseCalendarAPI",
    "DoorDashAPIAlt": "DoorDashAPI",
    "VenmoAPIAlt": "VenmoAPI",
}


def _display_class(cls_name: str) -> str:
    return _CLASS_DISPLAY_NAME.get(cls_name, cls_name)


# ---------------------------------------------------------------------------
# Prompt-override registry.
#
# When no overrides are registered (the default), every prompt-building
# function in this module returns exactly the same string it would have
# returned before this registry was added — the registry is a pure no-op.
#
# Optimized candidate prompts are registered by calling
# ``set_prompt_overrides(d)``. Each value in *d* is a template string that
# MUST use ``str.format``-style placeholders: single ``{name}`` slots for
# interpolation, and literal ``{``/``}`` must be doubled as ``{{``/``}}``.
# The keys supported today are listed in ``OVERRIDE_KEYS`` below. Passing
# an unknown key raises ``ValueError`` so typos surface loudly.
# ---------------------------------------------------------------------------

OVERRIDE_KEYS: frozenset[str] = frozenset({
    # System prompts — one per mode
    "system_prompt_ptc_encouraged",
    "system_prompt_ptc_cell_encouraged",
    "system_prompt_tool_only",
    "system_prompt_native_ptc",
    # Shared rules block (injected into every system prompt)
    "shared_rules",
    # PTC tool descriptions
    "ptc_execute_description",
    "ptc_cell_description",
    # ``code`` parameter descriptions on the PTC tools
    "ptc_execute_code_param",
    "ptc_cell_code_param",
})

_PROMPT_OVERRIDES: dict[str, str] | None = None


def set_prompt_overrides(overrides: dict[str, str] | None) -> None:
    """Register candidate prompt templates to replace the built-in ones.

    Pass ``None`` (or ``{}``) to reset to the built-in prompts. Unknown
    keys raise ``ValueError``.
    """
    global _PROMPT_OVERRIDES
    if not overrides:
        _PROMPT_OVERRIDES = None
        return
    unknown = set(overrides) - OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown prompt override keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(OVERRIDE_KEYS)}."
        )
    _PROMPT_OVERRIDES = dict(overrides)


def get_prompt_overrides() -> dict[str, str] | None:
    """Return the currently registered overrides (or ``None``)."""
    return None if _PROMPT_OVERRIDES is None else dict(_PROMPT_OVERRIDES)


def _resolve(key: str, default: str, **kwargs: Any) -> str:
    """Return the override template (formatted with *kwargs*) if registered,
    else the pre-built *default* string.

    ``default`` is the already-formatted string the caller would have returned
    without this registry. When no override is registered, this function is
    a straight pass-through — identical behavior to pre-registry code.
    """
    if _PROMPT_OVERRIDES is not None and key in _PROMPT_OVERRIDES:
        return _PROMPT_OVERRIDES[key].format(**kwargs)
    return default


_MINIMAL_PRINT_TOOL_SUFFIX = """

**Use ``print()`` only when you need a value on a later turn.** \
Everything you print is added to your next turn's input context and \
stays there for every subsequent turn — it is your cross-turn memory, \
not a debugging channel. Valid uses: (1) the final answer you are \
about to emit, (2) a small inspection sample to guide your next call \
(e.g. first element, a count, or specific keys — never the full \
object), or (3) a value you will reference in a follow-up call. Skip \
prints that are only useful during code writing — you have already \
seen those values. When inspecting, print the minimum needed to make \
your next decision, not the whole response. Unnecessary prints bloat \
future context without earning their keep."""

_MINIMAL_PRINT_CODE_PARAM = (
    " IMPORTANT: Print values you will read on a later turn — your "
    "answer, small inspection samples (first item, a count), or values "
    "you will reference in follow-up calls. Don't print full API "
    "responses or debug data you have already seen while writing this code."
)


def get_ptc_tool_def(
    involved_classes: list[str] | None = None,
    minimal_print: bool = False,
    tool_inspection: bool = False,
) -> ToolDef:
    """Return the ToolDef for the ptc_execute tool.

    If *involved_classes* is provided, the description only documents the
    APIs relevant to the current task.

    When ``tool_inspection`` is True, the description appends the ``??``
    discovery syntax (use alongside ``--tool-inspection`` in run_eval,
    which strips the per-method tool defs from the tool list so the agent
    must discover methods via inspection rather than read them from the
    tool schema). Mirrors ``get_ptc_cell_tool_def``'s behavior — the
    inspection mechanism is identical between the two tools, only the
    tool-name reference in the suffix changes.
    """
    if involved_classes:
        description = _build_ptc_description(
            involved_classes, tool_inspection=tool_inspection,
        )
    else:
        description = "Execute a Python code snippet in a sandboxed environment with API objects pre-loaded."

    if tool_inspection:
        description += _tool_inspection_suffix("ptc_execute")

    if minimal_print:
        description += _MINIMAL_PRINT_TOOL_SUFFIX

    var_names = ", ".join(
        _CLASS_TO_PREFIX.get(cls, cls) for cls in (involved_classes or [])
        if cls in _CLASS_TO_PREFIX
    ) or "the relevant API objects"

    code_desc = (
        f"Python code snippet to execute. The code has access to "
        f"pre-loaded API objects ({var_names}) for this task. "
        f"Use print() to surface values into the tool-call "
        f"output so you can see them on your next turn."
    )
    code_desc = _resolve("ptc_execute_code_param", code_desc, var_names=var_names)
    if minimal_print:
        code_desc += _MINIMAL_PRINT_CODE_PARAM

    return ToolDef(
        name="ptc_execute",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": code_desc,
                },
            },
            "required": ["code"],
        },
    )


def _build_ptc_cell_description(
    involved_classes: list[str],
    tool_inspection: bool = False,
) -> str:
    """Build a ``ptc_cell`` tool description (notebook-cell semantics).

    The API block lists only the class variable names and their Python
    class names. Full method signatures live in the per-method tool
    schemas (non-inspection mode) or are discovered via ``??`` (inspection
    mode). Keeping this section minimal avoids duplication and reduces
    tokens sent with every LLM call.

    Uses ``_display_class`` so alt-variant classes (e.g., ``YelpAPIAlt``)
    render under their canonical name in anything the model reads.
    """
    var_names_list = [
        _CLASS_TO_PREFIX.get(_display_class(cls), cls)
        for cls in involved_classes
        if _display_class(cls) in _CLASS_TO_PREFIX
    ]

    api_sections: list[str] = []
    for cls in involved_classes:
        canonical = _display_class(cls)
        if canonical not in _CLASS_TO_PREFIX:
            continue
        api_sections.append(
            f"### ``{_CLASS_TO_PREFIX.get(canonical, cls)}`` "
            f"({canonical})"
        )

    api_list = "\n\n".join(api_sections)
    var_names = ", ".join(var_names_list)

    has_calendar = any(c in involved_classes for c in
                       ["EnterpriseCalendarAPI", "PersonalCalendarAPI"])
    date_note = """
A string variable ``CURRENT_DATE`` is pre-loaded with the scenario's simulated \
current date (e.g. "2026-03-31"). Use this instead of ``date.today()`` or \
``datetime.now()`` for any date arithmetic.""" if has_calendar else ""

    if tool_inspection:
        # In inspection mode, examples must not name real methods.
        reuse_block = """\
**Inspect before you call.** You do NOT know which methods each API \
exposes until you inspect it — always run the inspection first and \
read the method's description and parameters before calling it.

```python
# cell 1 — inspect what's available
<api>??                        # list methods on this API class
<api>.<method>??               # read its description + parameters
```

```python
# cell 2 — only now make the real call, using the exact method name
# and parameter names you learned from inspection
result = <api>.<method>(param1=..., param2=...)
print(type(result), result)
```

**State persists across calls — reuse variables instead of refetching.** \
If you assigned API data to a variable in an earlier cell, it is still \
available. Do NOT call the same API method a second time just to get the \
same data back; reference the existing variable."""
    else:
        reuse_block = """\
**State persists across calls — reuse variables instead of refetching.** \
If you assigned API data to a variable in an earlier cell, it is still \
available. Do NOT call the same API method a second time just to get the \
same data back. Typical explore-then-reuse pattern:

```python
# cell 1 — fetch and explore the data shape
items = <api>.<list_method>(<args>)
print(type(items), items[:2] if isinstance(items, list) else items)
```

```python
# cell 2 — reuse `items` from cell 1 (do NOT call the list method again)
filtered = [x for x in items if x.get("<field>", 0) >= <threshold>]
print(f"{{len(filtered)}} matches")
```"""

    return f"""\
Execute a Python code snippet in a persistent sandboxed environment. \
**Each call to ``ptc_cell`` behaves like a cell in a Jupyter notebook**: \
variables, functions, and imports defined in one call remain available in \
every subsequent ``ptc_cell`` call within this task. Partial state from a \
cell that raises also persists — you can recover and keep going.

API objects pre-loaded as local variables:

{api_list}

## How to use

Write a Python code snippet as the ``code`` argument. Call methods using \
``<variable>.<method>(arg1=..., arg2=...)``. The variables available in this \
task are: {var_names}.
{date_note}

{reuse_block}

**Errors don't wipe state.** If a cell raises halfway through, everything \
assigned before the exception is still in scope on the next call.

To see data in your next turn, **print()** it. Whatever is printed to stdout \
is returned as the tool-call output. For structured values, use \
``print(json.dumps(obj))``. There is no magic return variable — printing is \
the only channel. Within a task, variables you define in one cell remain in \
scope in later cells, so you can reference them directly without re-printing.

**Important**: Check the return type of each API call before using ``.get()``. \
Some methods return a list directly, others return a dict wrapping a list. \
Print the raw response first if unsure.

Standard library modules (json, collections, datetime, itertools) are available."""
    return _resolve(
        "ptc_cell_description", default,
        api_sections=api_list, var_names=var_names, date_note=date_note,
    )


import re


# ---------------------------------------------------------------------------
# Tool inspection (`X??` / `X.m??` syntax, cell-mode only)
# ---------------------------------------------------------------------------

# Line-level regex for ``<identifier>??`` (with optional dots for method
# inspection). Anchored so we only rewrite standalone statements, not ``??``
# appearing inside expressions.
_INSPECT_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<target>[A-Za-z_][\w.]*)\?\?\s*$",
    re.MULTILINE,
)


def _preprocess_inspect_syntax(code: str) -> str:
    """Rewrite ``X??`` lines into ``__ptc_inspect__('X')`` calls.

    Supports multiple inspections in one cell (one per line), preserves
    indentation, and leaves non-matching lines alone.
    """
    def _sub(m: re.Match) -> str:
        indent = m.group("indent")
        target = m.group("target")
        return f'{indent}__ptc_inspect__({target!r})'
    return _INSPECT_LINE_RE.sub(_sub, code)


def _build_inspect_fn(
    tool_defs: list[ToolDef],
    involved_classes: list[str],
) -> Any:
    """Return a callable ``_inspect(name)`` that renders doc for an API
    class or method. Closed over the tool-defs available for this task.
    """
    # api_name -> [(method_name, ToolDef), ...]
    by_api: dict[str, list[tuple[str, ToolDef]]] = {}
    by_full_name: dict[str, ToolDef] = {}
    for td in tool_defs:
        if "." not in td.name:
            continue
        api, method = td.name.split(".", 1)
        by_api.setdefault(api, []).append((method, td))
        by_full_name[td.name] = td

    # APIs for this specific task. Determines "available".
    available_apis: set[str] = set()
    for cls in involved_classes:
        prefix = _CLASS_TO_PREFIX.get(cls)
        if prefix and prefix in by_api:
            available_apis.add(prefix)

    def _render_method(td: ToolDef) -> str:
        lines = [f"=== {td.name} ==="]
        lines.append("Description:")
        lines.append(td.description.strip())
        params = td.parameters or {}
        if isinstance(params, dict):
            props = params.get("properties") or {}
            required = set(params.get("required") or [])
            if props:
                lines.append("")
                lines.append("Parameters:")
                for pname, pspec in props.items():
                    ptype = (pspec or {}).get("type", "any")
                    pdesc = (pspec or {}).get("description", "").strip()
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - {pname}: {ptype}{req}")
                    if pdesc:
                        for ln in pdesc.splitlines():
                            lines.append(f"      {ln}")
            else:
                lines.append("")
                lines.append("Parameters: (none)")
        return "\n".join(lines)

    def _render_api(api: str) -> str:
        # Names-only listing for ALL APIs; the agent drills into a method
        # with ``<api>.<method>??``.
        entries = sorted(by_api.get(api, []), key=lambda mt: mt[0])
        lines = [f"=== {api} — {len(entries)} method(s) ==="]
        if entries:
            lines.append("Available methods (inspect any with e.g. "
                         f"``{api}.<method>??``):")
            for m, _td in entries:
                lines.append(f"  {api}.{m}")
        else:
            lines.append("(no methods — this API has no callable methods)")
        return "\n".join(lines)

    def _inspect(name: str) -> None:
        # name is always a string passed from the rewriter.
        if not isinstance(name, str) or not name:
            print("ERROR: __ptc_inspect__ expected a non-empty string target.")
            return

        if "." in name:
            # Method-level inspection.
            td = by_full_name.get(name)
            if td is not None:
                print(_render_method(td))
                return
            # Not found — give a helpful error.
            api = name.split(".", 1)[0]
            if api not in available_apis:
                print(
                    f"ERROR: api class '{api}' is not available for this task. "
                    f"Available: {sorted(available_apis)}"
                )
                return
            methods = sorted(m for m, _ in by_api.get(api, []))
            print(
                f"ERROR: method '{name}' not found. "
                f"Available methods in {api}: {methods}"
            )
            return

        # Class-level inspection.
        if name in available_apis:
            print(_render_api(name))
            return
        print(
            f"ERROR: api class '{name}' is not available for this task. "
            f"Available: {sorted(available_apis)}"
        )

    return _inspect


# Tool-inspection prompt suffix shared by both ``ptc_cell`` and
# ``ptc_execute``. Templated on ``{tool_name}`` so the same content can be
# attached to either tool's description without per-mode duplication.
# "Snippet" is used as the generic noun (works for both stateless and
# cell-mode invocations).
_TOOL_INSPECTION_SUFFIX_TEMPLATE = """

**Tool inspection (discover API methods on demand):**

You do NOT have upfront descriptions of each API's methods — only the \
class names (listed above). You MUST inspect an API before calling any \
of its methods.

Inspection syntax (write these inside a ``{tool_name}`` code snippet):

- Put ``<api_name>??`` on its own line to list all methods of that API \
class.
- Put ``<api_name>.<method_name>??`` on its own line to show that \
method's full description and parameters.
- Chain multiple inspections in a single snippet — put each on its own \
line and they all print to the snippet's stdout.

**Inspect the class first — in its own snippet — so you can SEE which \
methods exist before you try to inspect any method.** Method names \
are easy to mis-guess from intuition: a name that sounds right may \
not exist, while the real method has a different verb or noun order. \
Do NOT write ``<api>.<method>??`` for a method name you haven't \
confirmed from the class listing — you'll just get a \
``method not found`` error.

Recommended cadence:

```python
# Snippet 1 — class inspection only. Read the output of this snippet
# on the next turn to learn the ACTUAL method names on each API.
<api_a>??
<api_b>??
```

```python
# Snippet 2 — now that you know the real method names, inspect each
# one you actually need. Parameter details (types, required flags,
# response schema) are only visible at this level.
<api_a>.<actual_method_from_snippet_1>??
<api_a>.<another_actual_method>??
<api_b>.<actual_method_from_snippet_1>??
```

This gives you the full signature for every method before you call \
it, without wasting turns on method names that don't exist.

**Once you've inspected a method, call it inside a ``{tool_name}`` using \
the exact name and parameters you learned.** You can keep each call \
simple (one line, print the result) or combine it with other logic — \
loops, list comprehensions, calls to other APIs, data processing — \
whatever fits the task step. Either way, every call goes through \
``{tool_name}``.

```python
# Simple single call — good when you just need the raw return value
result = <api>.<method>(param1=..., param2=...)
print(result)
```

```python
# Combined with processing — good when you want to chain calls,
# filter/transform results, or combine multiple APIs in one step.
items = <api>.<method>(param1=...)
filtered = [x for x in items if ...]
enriched = [<api2>.<method2>(id=x["id"]) for x in filtered]
print(json.dumps(enriched))
```

Inspecting a non-existent class or method returns a clear error \
(``ERROR: api class '...' is not available`` or ``ERROR: method '...' \
not found. Available methods in <api>: [...]``). Use these errors to \
correct yourself.
"""


def _tool_inspection_suffix(tool_name: str) -> str:
    """Render the tool-inspection prompt suffix for a given tool name."""
    return _TOOL_INSPECTION_SUFFIX_TEMPLATE.format(tool_name=tool_name)


def get_ptc_cell_tool_def(
    involved_classes: list[str] | None = None,
    minimal_print: bool = False,
    tool_inspection: bool = False,
) -> ToolDef:
    """Return the ToolDef for the ``ptc_cell`` tool (persistent-namespace variant).

    Like ``get_ptc_tool_def`` but exposes a tool whose executions share a
    namespace across calls within the same task — Jupyter-notebook-cell
    semantics.

    When ``tool_inspection`` is True, the description explains the ``??``
    discovery syntax (use alongside ``--tool-inspection`` in run_eval,
    which strips the per-method tool defs from the tool list so the agent
    must discover methods via inspection rather than read them from the
    tool schema).
    """
    if involved_classes:
        description = _build_ptc_cell_description(
            involved_classes, tool_inspection=tool_inspection,
        )
    else:
        description = (
            "Execute a Python code snippet in a persistent sandboxed environment. "
            "Each call behaves like a Jupyter notebook cell — variables and "
            "imports persist across calls within this task."
        )

    if tool_inspection:
        description += _tool_inspection_suffix("ptc_cell")

    if minimal_print:
        description += _MINIMAL_PRINT_TOOL_SUFFIX

    var_names = ", ".join(
        _CLASS_TO_PREFIX.get(cls, cls) for cls in (involved_classes or [])
        if cls in _CLASS_TO_PREFIX
    ) or "the relevant API objects"

    code_desc = (
        f"Python code snippet to execute as the next notebook "
        f"cell. State from earlier ptc_cell calls (variables, "
        f"functions, imports) is still available. Pre-loaded "
        f"APIs: {var_names}. Use print() to surface values "
        f"into the tool-call output."
    )
    if tool_inspection:
        code_desc += (
            " Discover methods with ``<api>??`` or "
            "``<api>.<method>??`` on a line by itself; see the tool "
            "description for details."
        )
    if minimal_print:
        code_desc += _MINIMAL_PRINT_CODE_PARAM

    return ToolDef(
        name="ptc_cell",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": code_desc,
                },
            },
            "required": ["code"],
        },
    )


# ---------------------------------------------------------------------------
# API call-counting proxy
# ---------------------------------------------------------------------------

class _APICallProxy:
    """Transparent proxy that logs every public method call on an API object.

    Usage::

        log: list[dict] = []
        proxy = _APICallProxy(real_api, "yelp_api", log)
        proxy.search_businesses(category="Chinese")
        # log == [{"api": "yelp_api", "method": "search_businesses", ...}]

    Attribute access and non-callable attributes are forwarded unchanged.
    """

    def __init__(
        self,
        wrapped: Any,
        api_name: str,
        call_log: list[dict],
        error_log: list[dict] | None = None,
    ):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_api_name", api_name)
        object.__setattr__(self, "_call_log", call_log)
        object.__setattr__(self, "_error_log", error_log)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_wrapped"), name)
        if not callable(attr) or name.startswith("_"):
            return attr
        api_name = object.__getattribute__(self, "_api_name")
        call_log = object.__getattribute__(self, "_call_log")
        error_log = object.__getattribute__(self, "_error_log")

        def _logging_wrapper(*args: Any, **kwargs: Any) -> Any:
            call_log.append({"api": api_name, "method": name})
            result = attr(*args, **kwargs)
            # Auto-capture API-returned error dicts so the feedback loop
            # works even when the agent forgets to print the return value.
            if (
                error_log is not None
                and isinstance(result, dict)
                and result.get("error")
            ):
                error_log.append({
                    "api": api_name,
                    "method": name,
                    "args": args,
                    "kwargs": kwargs,
                    "error": str(result["error"])[:200],
                })
            return result
        return _logging_wrapper

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_wrapped"))


# ---------------------------------------------------------------------------
# Code execution sandbox
# ---------------------------------------------------------------------------

class PTCExecutor:
    """
    Executes Python code snippets against pre-loaded PTC API instances.

    Instantiate with the API objects for a given task, then call ``execute(code)``
    to run a snippet and get the output.
    """

    def __init__(
        self,
        enterprise_calendar_api: Any | None = None,
        org_chart_api: Any | None = None,
        yelp_api: Any | None = None,
        maps_api: Any | None = None,
        uber_api: Any | None = None,
        grocery_api: Any | None = None,
        personal_calendar_api: Any | None = None,
        doordash_api: Any | None = None,
        spotify_api: Any | None = None,
        venmo_api: Any | None = None,
        record_metrics: bool = False,
    ):
        self.enterprise_calendar_api = enterprise_calendar_api
        self.org_chart_api = org_chart_api
        self.yelp_api = yelp_api
        self.maps_api = maps_api
        self.uber_api = uber_api
        self.grocery_api = grocery_api
        self.personal_calendar_api = personal_calendar_api
        self.doordash_api = doordash_api
        self.spotify_api = spotify_api
        self.venmo_api = venmo_api
        # Optional tool-inspection context. When set, ``_inspect`` (a
        # callable rendering API/method docs) is bound into the cell
        # namespace and ``<X>??`` syntax in the code is rewritten to call
        # it. Attach via :meth:`set_inspector` after construction.
        self._inspector: Any = None
        # Persistent namespace for ``execute_cell`` (notebook-cell semantics).
        # Lazily initialized on first call; reset only by creating a new executor.
        self._cell_namespace: dict[str, Any] | None = None
        # Log of API calls made during execute()/execute_cell().
        # Cleared at the start of each call; read after call returns.
        self._call_log: list[dict] = []
        # Log of API-returned error dicts, surfaced automatically to the
        # agent via stderr so silent error-returns don't disappear.
        self._error_log: list[dict] = []
        # Runtime-overhead instrumentation. Default OFF: when
        # False, execute()/execute_cell() behave exactly as before — no
        # timing, no sizing, no extra fields anywhere. When True, a side-
        # channel record is appended per invocation to
        # ``self._runtime_metrics``; the tool RESULT returned to the model
        # is never modified, so model-visible content is identical either way.
        self._record_metrics = bool(record_metrics)
        self._runtime_metrics: list[dict] = []
        self._psutil: Any = None  # lazily resolved; False = unavailable

    # ------------------------------------------------------------------
    # Namespace helpers (shared by stateless ``execute`` and stateful
    # ``execute_cell``)
    # ------------------------------------------------------------------

    def _bind_apis(self, namespace: dict[str, Any]) -> None:
        """Bind/refresh API handles (and the ``calendar_api`` alias) into ``namespace``.

        Called both when building a fresh namespace and at the start of each
        ``execute_cell`` call so a cell that accidentally overwrites
        e.g. ``calendar_api = None`` gets the real handle back on the next call.
        Only the pre-loaded API names are restored; any user-defined variables
        are left untouched.
        """
        log = self._call_log
        errs = self._error_log
        if self.enterprise_calendar_api is not None:
            proxy = _APICallProxy(self.enterprise_calendar_api, "enterprise_calendar_api", log, errs)
            namespace["enterprise_calendar_api"] = proxy
            namespace["calendar_api"] = proxy
        if self.org_chart_api is not None:
            namespace["org_chart_api"] = _APICallProxy(self.org_chart_api, "org_chart_api", log, errs)
        if self.yelp_api is not None:
            namespace["yelp_api"] = _APICallProxy(self.yelp_api, "yelp_api", log, errs)
        if self.maps_api is not None:
            namespace["maps_api"] = _APICallProxy(self.maps_api, "maps_api", log, errs)
        if self.uber_api is not None:
            namespace["uber_api"] = _APICallProxy(self.uber_api, "uber_api", log, errs)
        if self.grocery_api is not None:
            namespace["grocery_api"] = _APICallProxy(self.grocery_api, "grocery_api", log, errs)
        if self.personal_calendar_api is not None:
            proxy = _APICallProxy(self.personal_calendar_api, "personal_calendar_api", log, errs)
            namespace["personal_calendar_api"] = proxy
            namespace["calendar_api"] = proxy
        if self.doordash_api is not None:
            namespace["doordash_api"] = _APICallProxy(self.doordash_api, "doordash_api", log, errs)
        if self.spotify_api is not None:
            namespace["spotify_api"] = _APICallProxy(self.spotify_api, "spotify_api", log, errs)
        if self.venmo_api is not None:
            namespace["venmo_api"] = _APICallProxy(self.venmo_api, "venmo_api", log, errs)

    def _build_base_namespace(self) -> dict[str, Any]:
        """Build a fresh namespace with stdlib aliases, CURRENT_DATE, and APIs."""
        import collections
        import datetime
        import itertools

        # Discover simulated current_date from whichever calendar API is loaded
        _current_date = None
        for api in [self.enterprise_calendar_api, self.personal_calendar_api]:
            if api is not None:
                try:
                    ctx = api.get_calendar_context()
                    _current_date = ctx.get("current_date") or ctx.get("simulated_now_iso", "")[:10]
                except Exception:
                    pass
                break

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "json": json,
            "collections": collections,
            "defaultdict": collections.defaultdict,
            "datetime": datetime,
            "itertools": itertools,
        }
        if _current_date:
            namespace["CURRENT_DATE"] = _current_date
        self._bind_apis(namespace)
        # Tool-inspection hook. Bound under a namespaced name so agent
        # code can't accidentally shadow it with ``_inspect = ...``.
        if self._inspector is not None:
            namespace["__ptc_inspect__"] = self._inspector
        return namespace

    def set_inspector(self, inspector: Any) -> None:
        """Attach a tool-inspection callable. Called by the runner when
        ``--tool-inspection`` is active. See :func:`_build_inspect_fn`.
        """
        self._inspector = inspector

    # Default per-cell execution budget. Bounds a single ``ptc_execute`` or
    # ``ptc_cell`` call; overrides via ``PTC_CELL_TIMEOUT_S`` env var. Set
    # generously enough that legitimate tasks aren't cut off, but tightly
    # enough that a runaway agent script (infinite loop, exponential DFS)
    # can't monopolize the whole run's timeout budget.
    _DEFAULT_CELL_TIMEOUT_S = 180

    @classmethod
    def _cell_timeout_s(cls) -> int:
        import os
        raw = os.environ.get("PTC_CELL_TIMEOUT_S")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return cls._DEFAULT_CELL_TIMEOUT_S

    def _run_in(self, namespace: dict[str, Any], code: str) -> dict[str, Any]:
        """Run ``code`` in ``namespace`` and return the stdout/stderr/error dict.

        A per-cell wall-clock timeout is enforced via
        ``PyThreadState_SetAsyncExc``: a watchdog ``threading.Timer`` raises
        ``TimeoutError`` in the executing thread after the budget elapses.
        This protects the run from runaway agent code (infinite loops,
        exponential search) that would otherwise burn the whole
        ``overall_timeout`` on a single task.

        To surface any value to the conversation, the code must ``print()`` it
        (possibly via ``json.dumps`` for structured data). There is no magic
        ``result`` variable.
        """
        timeout_s = self._cell_timeout_s()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        output: dict[str, Any] = {"stdout": "", "stderr": "", "error": None}

        worker_tid = threading.get_ident()
        fired = [False]

        def _interrupt() -> None:
            # Raise TimeoutError in the worker thread. Uses CPython's C API;
            # exception is delivered at the next bytecode boundary in that
            # thread (so pure-Python infinite loops are interrupted quickly).
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(worker_tid),
                ctypes.py_object(TimeoutError),
            )
            if res == 1:
                fired[0] = True
            elif res > 1:
                # More than one thread got the exception — shouldn't happen;
                # clear to be safe.
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(worker_tid), 0
                )

        timer = threading.Timer(timeout_s, _interrupt)
        timer.daemon = True
        timer.start()

        try:
            with _thread_local_redirect(stdout_buf, stderr_buf):
                try:
                    exec(code, namespace)
                except TimeoutError:
                    output["error"] = (
                        f"Cell execution timed out after {timeout_s}s "
                        f"(PTC_CELL_TIMEOUT_S). The agent likely wrote a "
                        f"runaway loop or unbounded search; rewrite the "
                        f"algorithm to finish within the budget."
                    )
                except SystemExit:
                    # SystemExit is a BaseException: without this branch an
                    # agent-written ``exit()``/``sys.exit()`` escapes the
                    # worker thread via Future.result() and terminates the
                    # whole eval process with status 0 (silent truncation).
                    output["error"] = (
                        "SystemExit: exit()/sys.exit() is not available in "
                        "this sandbox. End the cell normally; use print() to "
                        "emit results."
                    )
                except Exception as exc:
                    # API-shaped errors (those carrying an ``.error`` dict with
                    # ``error_code``/``message``) are surfaced as a clean
                    # one-liner — same shape the tool_only dispatcher uses —
                    # so the agent sees an opaque "internal error, retry"
                    # response indistinguishable from a real REST API failure.
                    # Genuine Python errors keep their full traceback so the
                    # agent can debug its own code.
                    api_err = getattr(exc, "error", None)
                    if isinstance(api_err, dict) and "message" in api_err:
                        output["error"] = (
                            f"{type(exc).__name__}: {api_err['message']}"
                        )
                    else:
                        output["error"] = traceback.format_exc()
        finally:
            timer.cancel()
            # If the timer fired but its TimeoutError hasn't been delivered
            # yet (race between ``fired[0] = True`` and our return here),
            # clear the pending async exception so it doesn't poison code
            # that runs after _run_in returns.
            if fired[0]:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(worker_tid), 0
                )

        output["stdout"] = stdout_buf.getvalue()
        output["stderr"] = stderr_buf.getvalue()
        return output

    @staticmethod
    def _format_call_signature(
        args: tuple, kwargs: dict, max_len: int = 150,
    ) -> str:
        """Render ``(arg1, kw=val, ...)`` with repr-safe values, truncated."""
        parts = [repr(a) for a in args]
        parts += [f"{k}={v!r}" for k, v in kwargs.items()]
        sig = ", ".join(parts)
        if len(sig) > max_len:
            sig = sig[: max_len - 3] + "..."
        return f"({sig})"

    # Safety caps on fields returned in the tool-call result. OpenAI caps
    # each message at 10 MiB; if a single ptc_execute/ptc_cell call has a
    # runaway loop it can easily blow past that (e.g., 440,000 API calls ×
    # ~50 bytes/entry = 22 MB in ``api_calls`` alone). Keep these generous
    # enough to not bite normal runs, strict enough to stay well under the
    # per-message limit.
    _MAX_API_CALLS_IN_RESULT = 100
    _MAX_STDOUT_BYTES = 200_000  # ~200 KB

    @staticmethod
    def _truncate_call_log(call_log: list[dict], cap: int) -> tuple[list, int]:
        """Return a bounded view of ``call_log`` plus the true count.

        When over ``cap``, returns first half + a truncation sentinel +
        last half. The true count is always preserved as a separate field.
        """
        total = len(call_log)
        if total <= cap:
            return list(call_log), total
        half = cap // 2
        head = list(call_log[:half])
        tail = list(call_log[-half:])
        marker = {"_truncated_": f"{total - cap} more calls omitted"}
        return head + [marker] + tail, total

    @staticmethod
    def _truncate_stdout(s: str, cap: int) -> str:
        if len(s) <= cap:
            return s
        head = s[: cap // 2]
        tail = s[-cap // 2 :]
        return f"{head}\n…[stdout truncated: {len(s) - cap} bytes omitted]…\n{tail}"

    def _surface_api_errors(self, output: dict[str, Any]) -> None:
        """Append ``api.method(args): error`` lines to stderr for every
        error-dict the APIs returned this call, unless that error text
        already appears in stdout (i.e. the agent printed it). The args
        signature disambiguates repeated calls to the same method.
        """
        if not self._error_log:
            return
        stdout = output.get("stdout", "") or ""
        lines: list[str] = []
        for entry in self._error_log:
            err = entry["error"]
            # Skip if the agent already surfaced it via print.
            if err and err in stdout:
                continue
            sig = self._format_call_signature(
                entry.get("args", ()), entry.get("kwargs", {})
            )
            lines.append(f"{entry['api']}.{entry['method']}{sig}: {err}")
        if not lines:
            return
        existing = output.get("stderr", "") or ""
        addition = "\n".join(lines)
        output["stderr"] = (existing + ("\n" if existing else "") + addition)

    # ------------------------------------------------------------------
    # Runtime-overhead instrumentation (all no-ops unless
    # ``record_metrics=True`` was passed at construction).
    # ------------------------------------------------------------------

    def _estimate_namespace_bytes(self, namespace: dict[str, Any]) -> dict[str, Any]:
        """Deep-size estimate of user state in ``namespace``. Never raises.

        Environment handles are excluded (modules, ``_APICallProxy`` API
        handles, the inspector binding, dunder entries) — we want the size of
        *notebook state the agent accumulated*, not the simulated backend.
        Uses ``pympler.asizeof`` when available; otherwise a bounded
        seen-set traversal over containers with ``sys.getsizeof``
        (documented as an underestimate for exotic objects).
        """
        def _is_env_handle(v: Any) -> bool:
            return isinstance(v, types.ModuleType) or isinstance(v, _APICallProxy)

        items = {
            k: v for k, v in namespace.items()
            if not k.startswith("__") and not _is_env_handle(v)
        }
        # Sampling-based deep size: exact for small containers; for large
        # containers, extrapolate from a bounded element sample. Fast
        # (O(sample) per container) and accurate to within sampling error
        # for homogeneous data; shared sub-objects across sampled elements
        # may be double-counted (documented estimate, not an exact figure).
        _SAMPLE = 1024
        _MAX_DEPTH = 8
        seen: set[int] = set()

        def _sized(obj: Any, depth: int) -> float:
            if _is_env_handle(obj) or depth > _MAX_DEPTH:
                return 0.0
            oid = id(obj)
            if oid in seen:
                return 0.0
            seen.add(oid)
            try:
                total = float(sys.getsizeof(obj))
            except Exception:
                return 0.0
            if isinstance(obj, (str, bytes, bytearray, int, float, bool, complex)):
                return total
            if isinstance(obj, dict):
                n = len(obj)
                if n:
                    kv = list(obj.items())[:_SAMPLE]
                    s = sum(
                        _sized(k, depth + 1) + _sized(v, depth + 1) for k, v in kv
                    )
                    total += s * (n / len(kv))
                return total
            if isinstance(obj, (list, tuple, set, frozenset)):
                n = len(obj)
                if n:
                    elems = list(obj)[:_SAMPLE]
                    s = sum(_sized(e, depth + 1) for e in elems)
                    total += s * (n / len(elems))
                return total
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                total += _sized(d, depth + 1)
            return total

        try:
            return {
                "namespace_bytes": int(sum(_sized(v, 0) for v in items.values())),
                "sizing_method": "sampled_traversal",
            }
        except Exception:
            return {"namespace_bytes": None, "sizing_method": "failed"}

    def _process_rss_bytes(self) -> int | None:
        """Current process RSS via psutil, or None if unavailable."""
        if self._psutil is None:
            try:
                import psutil  # type: ignore
                self._psutil = psutil
            except Exception:
                self._psutil = False
        if not self._psutil:
            return None
        try:
            return int(self._psutil.Process().memory_info().rss)
        except Exception:
            return None

    def _append_runtime_record(
        self, surface: str, namespace: dict[str, Any],
        setup_time_s: float, exec_time_s: float,
    ) -> None:
        """Record one invocation's runtime metrics (side channel only)."""
        try:
            t0 = time.perf_counter()
            size_info = self._estimate_namespace_bytes(namespace)
            sizing_time_s = time.perf_counter() - t0
            self._runtime_metrics.append({
                "surface": surface,
                "index": len(self._runtime_metrics),
                "setup_time_s": round(setup_time_s, 6),
                "exec_time_s": round(exec_time_s, 6),
                "sizing_time_s": round(sizing_time_s, 6),
                "namespace_vars": sum(
                    1 for k in namespace if not k.startswith("__")
                ),
                "process_rss_bytes": self._process_rss_bytes(),
                **size_info,
            })
        except Exception:
            # Instrumentation must never affect the run.
            pass

    def runtime_metrics_summary(self) -> dict[str, Any]:
        """Aggregate view of recorded per-invocation runtime metrics."""
        per = list(self._runtime_metrics)
        def _tot(field: str) -> float:
            return round(sum(r.get(field) or 0.0 for r in per), 6)
        peak_bytes = max(
            (r.get("namespace_bytes") or 0 for r in per), default=0
        )
        return {
            "n_code_invocations": len(per),
            "total_setup_time_s": _tot("setup_time_s"),
            "total_exec_time_s": _tot("exec_time_s"),
            "total_sizing_time_s": _tot("sizing_time_s"),
            "peak_namespace_bytes": peak_bytes,
            "final_namespace_vars": per[-1]["namespace_vars"] if per else 0,
            "per_invocation": per,
        }

    def execute(self, code: str) -> dict[str, Any]:
        """
        Execute ``code`` in a *fresh* namespace with the API objects (stateless).

        Returns a dict with:
          - ``stdout``: captured print output
          - ``stderr``: captured stderr (if any) + auto-surfaced API error dicts
          - ``error``: traceback string if execution raised
          - ``api_calls``: list of ``{"api": ..., "method": ...}`` dicts
        """
        self._call_log.clear()
        self._error_log.clear()
        _t_setup = time.perf_counter() if self._record_metrics else None
        namespace = self._build_base_namespace()
        _setup_s = (
            time.perf_counter() - _t_setup if _t_setup is not None else 0.0
        )
        # If tool-inspection is active, rewrite ``<x>??`` lines to a
        # function call. Otherwise the exec would raise SyntaxError.
        if self._inspector is not None:
            code = _preprocess_inspect_syntax(code)
        _t_exec = time.perf_counter() if self._record_metrics else None
        output = self._run_in(namespace, code)
        if _t_exec is not None:
            self._append_runtime_record(
                "ptc_execute", namespace,
                _setup_s, time.perf_counter() - _t_exec,
            )
        self._surface_api_errors(output)
        calls, call_count = self._truncate_call_log(
            self._call_log, self._MAX_API_CALLS_IN_RESULT
        )
        output["api_calls"] = calls
        output["api_call_count"] = call_count
        output["stdout"] = self._truncate_stdout(
            output.get("stdout", "") or "", self._MAX_STDOUT_BYTES
        )
        # Serializable view of API-returned error dicts (omit args objects
        # which may not be JSON-safe; the agent already sees a rendered form
        # in stderr, and this field is consumed by DocLearner).
        output["api_errors"] = [
            {"api": e["api"], "method": e["method"], "error": e["error"]}
            for e in self._error_log
        ]
        return output

    def execute_cell(self, code: str) -> dict[str, Any]:
        """
        Execute ``code`` in a *persistent* namespace — Jupyter-notebook-cell semantics.

        Within a single ``PTCExecutor`` instance, variables, functions, and
        imports defined in one ``execute_cell`` call remain available in every
        subsequent call. Partial state from a cell that raises also persists
        (``exec`` binds every statement that ran before the exception).

        API handles (``calendar_api``, ``yelp_api``, …) are re-bound at the
        start of each call so a cell that overwrites them can't break
        subsequent cells.

        Return-value shape matches ``execute``.
        """
        self._call_log.clear()
        self._error_log.clear()
        _t_setup = time.perf_counter() if self._record_metrics else None
        if self._cell_namespace is None:
            self._cell_namespace = self._build_base_namespace()
        else:
            # Re-bind API handles in case the previous cell overwrote them.
            self._bind_apis(self._cell_namespace)
            # Re-bind the inspector as well — cheap, idempotent, and
            # protects against accidental shadowing in prior cells.
            if self._inspector is not None:
                self._cell_namespace["__ptc_inspect__"] = self._inspector
        _setup_s = (
            time.perf_counter() - _t_setup if _t_setup is not None else 0.0
        )
        if self._inspector is not None:
            code = _preprocess_inspect_syntax(code)
        _t_exec = time.perf_counter() if self._record_metrics else None
        output = self._run_in(self._cell_namespace, code)
        if _t_exec is not None:
            self._append_runtime_record(
                "ptc_cell", self._cell_namespace,
                _setup_s, time.perf_counter() - _t_exec,
            )
        self._surface_api_errors(output)
        calls, call_count = self._truncate_call_log(
            self._call_log, self._MAX_API_CALLS_IN_RESULT
        )
        output["api_calls"] = calls
        output["api_call_count"] = call_count
        output["stdout"] = self._truncate_stdout(
            output.get("stdout", "") or "", self._MAX_STDOUT_BYTES
        )
        output["api_errors"] = [
            {"api": e["api"], "method": e["method"], "error": e["error"]}
            for e in self._error_log
        ]
        return output


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

# Valid modes for build_system_prompt
PROMPT_MODES = (
    "ptc_encouraged",
    "tool_only",
    "ptc_cell_encouraged",
    "native_ptc",
)


def _api_list_str(involved_classes: list[str]) -> str:
    """Return a human-readable list of API variable names."""
    _MAP = {
        "EnterpriseCalendarAPI": "enterprise_calendar_api (EnterpriseCalendarAPI)",
        "OrgChartAPI": "org_chart_api (OrgChartAPI)",
        "YelpAPI": "yelp_api (YelpAPI)",
        "GoogleMapsAPI": "maps_api (GoogleMapsAPI)",
        "UberAPI": "uber_api (UberAPI)",
        "GroceryDeliveryAPI": "grocery_api (GroceryDeliveryAPI)",
        "PersonalCalendarAPI": "personal_calendar_api (PersonalCalendarAPI)",
        "DoorDashAPI": "doordash_api (DoorDashAPI)",
        "SpotifyAPI": "spotify_api (SpotifyAPI)",
        "VenmoAPI": "venmo_api (VenmoAPI)",
    }
    apis = [_MAP[_display_class(c)] for c in involved_classes
            if _display_class(c) in _MAP]
    return ", ".join(apis) if apis else "the relevant API objects"


def build_system_prompt(
    involved_classes: list[str],
    mode: str = "ptc_encouraged",
) -> str:
    """
    Build a system prompt whose tone depends on *mode*.

    Modes
    -----
    ``ptc_encouraged``
        The model is explicitly told about ``ptc_execute``, shown its
        capabilities, and encouraged to write code for multi-step tasks.
        (This is the original prompt.)

    ``tool_only``
        The model is told about the API tools only; ``ptc_execute`` is *not*
        mentioned (the tool itself is excluded upstream by the runner). The
        model must solve everything via direct API tool calls.

    ``ptc_cell_encouraged``
        Like ``ptc_encouraged`` but the code-execution tool is ``ptc_cell``,
        which has persistent-namespace (Jupyter-notebook-cell) semantics
        across calls within a task. Encourages explore-then-reuse patterns.

    ``native_ptc``
        Used with the ``claude_ptc`` backend. The bench's ``ptc_execute`` /
        ``ptc_cell`` tools are absent — Anthropic's native ``code_execution``
        server tool is added by the agent layer instead. Minimal, neutral
        wording: mention the API tools and the ``code_execution`` tool by name,
        with no encouragement, examples, or naming guidance.
    """
    if mode not in PROMPT_MODES:
        raise ValueError(
            f"Unknown prompt mode: {mode!r}. Choose from {PROMPT_MODES}."
        )

    api_list = _api_list_str(involved_classes)

    # Shared rules — identical across all modes for fair comparison.
    _shared_rules = """\
- Always use the tools to access data. Do NOT hard-code data.
- State is shared across all tool calls — side-effects persist.
- The scenario may use simulated time. Do not rely on the real system clock. \
Instead, discover dates and times from the API responses or the data itself.
- If an approach doesn't produce useful results, try alternative strategies \
before giving up.
- When referencing entities (people, places, items), use the identifiers that \
the API expects (e.g. usernames, IDs) rather than display names or free text.
- Before committing to a complex approach, make a simple probing call first to \
check what the data looks like.
- Start with the simplest approach that could work. Add complexity only if needed.
- Your **final answer** is whatever you say in your last response without making \
any tool calls. Put your answer there, in plain text. If the task asks for a \
specific structured format (e.g., JSON), include that structure in your final \
text response."""
    _shared_rules = _resolve("shared_rules", _shared_rules)

    # ---- ptc_encouraged: encourage PTC + code-specific syntax ---------------
    if mode == "ptc_encouraged":
        default = f"""\
You are a helpful assistant that solves tasks using the available tools.

You have access to tools for interacting with the following APIs: {api_list}. \
You also have access to a ``ptc_execute`` tool that runs Python code in a \
sandboxed environment with those same API objects pre-loaded.

**For any task that requires multiple API calls, data processing, or \
computation, you should write a Python program using ``ptc_execute`` rather \
than calling tools one at a time.**

**How to call API methods in PTC code:**

Inside ``ptc_execute``, each API is a Python object with the variable name \
shown above in parentheses. You call methods on it using standard Python \
syntax: ``<api_variable>.<method_name>(arg1=..., arg2=...)``.

For example, given an API loaded as ``<api_var>``, you write:
```python
results = <api_var>.<method_name>(<arg1>=<value>, <arg2>=<value>)
record  = <api_var>.<other_method>(<id_arg>=<id_value>)
```

**Do NOT** try to import the API, instantiate it yourself, or prefix with the \
class name. The variables are already loaded. Just write \
``<variable>.<method>(...)``. The exact variable names for this task are \
listed at the top of this prompt — use them verbatim.

**Naming gotcha — always use dots inside ``ptc_execute``, never underscores.** \
The direct tool-call interface (visible in the tool list) shows method names \
with underscores, e.g. ``<api_var>_<method_name>`` — but that is a \
tool-schema rendering, not a Python function. Inside ``ptc_execute`` code you \
MUST use the dotted form: ``<api_var>.<method_name>(...)``. Writing \
``<api_var>_<method_name>(...)`` in code will raise ``NameError``.

Do not write ``raise Exception(...)`` on unexpected conditions — it kills \
the whole code block. Instead, fall back gracefully.

Use ``print()`` to surface values into the tool-call output. Whatever you \
print is what you see on the next turn — there is no magic return variable. \
Standard library modules (json, collections, datetime, itertools) are available.

Use the tools to complete the task. Once you have the answer, provide it in \
plain text.

**Important rules:**
{_shared_rules}
"""
        return _resolve(
            "system_prompt_ptc_encouraged", default,
            api_list=api_list, shared_rules=_shared_rules,
        )

    # ---- ptc_cell_encouraged: encourage ptc_cell + notebook-cell semantics -
    if mode == "ptc_cell_encouraged":
        default = f"""\
You are a helpful assistant that solves tasks using the available tools.

You have access to tools for interacting with the following APIs: {api_list}. \
You also have access to a ``ptc_cell`` tool that runs Python code in a \
sandboxed environment with those same API objects pre-loaded.

**Each call to ``ptc_cell`` behaves like a cell in a Jupyter notebook.** \
Variables, functions, and imports you define in one call stay in scope for \
every subsequent call within this task. Partial state from a cell that \
raises also persists — assignments that happened before the exception are \
still there on the next call, so you can recover without starting over.

**For any task that requires multiple API calls, data processing, or \
computation, you should write Python using ``ptc_cell`` rather than calling \
tools one at a time.** Because state persists, prefer splitting work across \
cells: inspect data in one cell, then process it in the next.

**Reuse before refetching.** If you already called an API method in an \
earlier cell and assigned the result to a variable, that variable is still \
in scope — USE IT. Do not re-call the same API method just because you \
want the data again. Refetching wastes tokens and latency, and risks \
getting a different response on a stateful API. Before any API call, ask \
yourself: *"have I already fetched this in a previous cell? if yes, \
reference the existing variable."*

Typical pattern, regardless of which APIs your task loads:
```python
# cell 1 — fetch and inspect
items = <api_var>.<list_method>(<args>)
print(type(items), items[:2] if isinstance(items, list) else items)
```
```python
# cell 2 — reuse `items` from cell 1 (do NOT call the list method again)
filtered = [x for x in items if x.get("<field>", 0) >= <threshold>]
print(f"{{len(filtered)}} matches")
```

**How to call API methods in PTC code:**

Inside ``ptc_cell``, each API is a Python object with the variable name \
shown above in parentheses. You call methods on it using standard Python \
syntax: ``<api_variable>.<method_name>(arg1=..., arg2=...)``.

For example, given an API loaded as ``<api_var>``, you write:
```python
results = <api_var>.<method_name>(<arg1>=<value>, <arg2>=<value>)
record  = <api_var>.<other_method>(<id_arg>=<id_value>)
```

**Do NOT** try to import the API, instantiate it yourself, or prefix with the \
class name. The variables are already loaded. Just write \
``<variable>.<method>(...)``. The exact variable names for this task are \
listed at the top of this prompt — use them verbatim.

**Naming gotcha — always use dots inside ``ptc_cell``, never underscores.** \
The direct tool-call interface (visible in the tool list) shows method names \
with underscores, e.g. ``<api_var>_<method_name>`` — but that is a \
tool-schema rendering, not a Python function. Inside ``ptc_cell`` code you \
MUST use the dotted form: ``<api_var>.<method_name>(...)``. Writing \
``<api_var>_<method_name>(...)`` in a cell will raise ``NameError``.

Do not write ``raise Exception(...)`` on unexpected conditions — it kills \
the current cell (though earlier cells' state survives). Instead, fall back \
gracefully.

Use ``print()`` to surface values into the tool-call output. Whatever you \
print is what you see on the next turn — there is no magic return variable. \
For structured values, ``print(json.dumps(obj))`` is fine. Standard library \
modules (json, collections, datetime, itertools) are available.

Use the tools to complete the task. Once you have the answer, provide it in \
plain text.

**Important rules:**
{_shared_rules}
"""
        return _resolve(
            "system_prompt_ptc_cell_encouraged", default,
            api_list=api_list, shared_rules=_shared_rules,
        )

    # ---- native_ptc: claude_ptc backend (Anthropic's native code_execution) -
    # The benchmark's ptc_execute/ptc_cell tools are stripped in this mode;
    # the ``code_execution`` server tool is added by the agent layer. Factual
    # mention of the tools, no encouragement.
    if mode == "native_ptc":
        default = f"""\
You are a helpful assistant that solves tasks using the available tools.

You have access to tools for interacting with the following APIs: {api_list}. \
You also have access to a ``code_execution`` tool that runs Python code in a \
sandboxed environment.

Use the tools to complete the task. Once you have the answer, provide it in \
plain text.

**Important rules:**
{_shared_rules}
"""
        return _resolve(
            "system_prompt_native_ptc", default,
            api_list=api_list, shared_rules=_shared_rules,
        )

    # ---- tool_only: no mention of ptc_execute ----------------------------
    # (ptc_execute is not provided as a tool in this mode)
    default = f"""\
You are a helpful assistant that solves tasks using the available tools.

You have access to tools for interacting with the following APIs: {api_list}.

Use the tools to complete the task. Once you have the answer, provide it in \
plain text.

**Important rules:**
{_shared_rules}
"""
    return _resolve(
        "system_prompt_tool_only", default,
        api_list=api_list, shared_rules=_shared_rules,
    )
