"""Learn API documentation addenda from trace errors between task batches."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from iptc.agent import ToolDef


@dataclass
class ErrorPattern:
    """Deduplicated trace error; traceback patterns await method attribution by LLM triage."""
    method_name: str
    error_message: str
    code_snippet: str
    count: int = 1
    is_traceback: bool = False
    api_calls: list[dict] = field(default_factory=list)


class DocLearner:
    """Accumulates error patterns and generates LLM-based doc improvements.

    Thread-safe: multiple worker threads can call :meth:`collect_from_trace`
    concurrently while the main thread calls :meth:`improve_and_apply` between
    batches.
    """

    _SYSTEM_PROMPT = (
        "You are an API documentation writer. An agent encountered an error "
        "while using an API method. Based on the error and the current "
        "documentation, produce a short addendum (1-2 sentences) that would "
        "help a future agent avoid this mistake.\n\n"
        "Rules:\n"
        "- Be specific about return types, valid parameter values, or "
        "preconditions\n"
        "- Output ONLY the addendum text (1-2 sentences), not the full "
        "rewritten documentation\n"
        "- Do not repeat information already in the current documentation\n"
        "- If the error is a simple typo or syntax issue in the agent's code "
        "(not a documentation gap), output exactly: NO_DOC_CHANGE"
    )

    # Triage unattributed cell tracebacks before choosing a documentation target.
    _TRIAGE_SYSTEM_PROMPT = (
        "You are an API documentation writer. An agent's code crashed inside "
        "a Jupyter-like cell that calls our APIs. Given the code, the Python "
        "traceback, and the list of API methods that were called in the "
        "cell, decide whether any tool's documentation should be updated to "
        "help a future agent avoid this mistake. There are three possible "
        "outputs:\n\n"
        "1. API-method attribution (use when the root cause is misuse of a "
        "specific API method — wrong args, wrong return-value assumption, "
        "missing precondition, etc.). Output exactly two lines:\n"
        "    METHOD: <api_name>.<method_name>\n"
        "    ADDENDUM: <1-2 sentences that belong in that method's doc>\n"
        "The method MUST be one of the API methods listed in the user "
        "message.\n\n"
        "2. Code-execution-tool attribution (use when the root cause is a "
        "general Python code pattern that would recur across API methods — "
        "e.g., serializing non-JSON objects, infinite loops, string/int "
        "mismatches, misunderstanding stateless-vs-stateful execution). "
        "Output exactly two lines:\n"
        "    TOOL: <code_exec_tool_name>\n"
        "    ADDENDUM: <1-2 sentences about the code pattern to avoid>\n"
        "The tool name MUST be the exact code-execution tool mentioned in "
        "the user message (either ``ptc_cell`` or ``ptc_execute``).\n\n"
        "3. No change (use when it's a one-off typo or agent logic error "
        "that wouldn't generalize). Output exactly: NO_DOC_CHANGE\n\n"
        "Rules:\n"
        "- Never attribute to the code-execution tool using the METHOD: "
        "prefix. Use TOOL: for that.\n"
        "- The addendum must be specific and avoid repeating existing docs."
    )

    def __init__(self, model: str = "gpt-5-mini", backend: str | None = None):
        self.model = model
        if backend is not None:
            backend = self._BACKEND_ALIASES.get(backend, backend)
            if backend not in ("openai", "anthropic", "gemini") and backend not in self._COMPAT_BACKENDS:
                raise ValueError(f"Unknown learner backend {backend!r}")
        self.backend = backend
        self._lock = threading.Lock()
        # method_name → list of addendum strings
        self._improvements: dict[str, list[str]] = {}
        # (method_name, error_key) → already processed
        self._seen: set[tuple[str, str]] = set()
        self._version = 0
        # Create only the client required by this learner's backend.
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None
        self._qwen_client = None
        self._glm_client = None
        self._kimi_client = None
        self._openrouter_client = None
        self._log: list[dict[str, Any]] = []
        # method_name → pristine description (snapshotted on first sight)
        self._originals: dict[str, str] = {}

    # -- Backend dispatch (explicit backend, else model-name → SDK) --------

    _BACKEND_ALIASES = {
        "claude": "anthropic",
        "claude_ptc": "anthropic",
        "openai_native": "openai",
        "gemini_native": "gemini",
    }

    def _detect_backend(self) -> str:
        """Prefer the explicit backend, otherwise infer from model prefixes with an OpenAI fallback."""
        if self.backend:
            return self.backend
        name = (self.model or "").lower()
        if name.startswith("claude"):
            return "anthropic"
        if name.startswith("gemini"):
            return "gemini"
        if name.startswith("qwen"):
            return "qwen"
        if name.startswith("glm"):
            return "glm"
        if name.startswith("kimi") or name.startswith("moonshot"):
            return "kimi"
        return "openai"

    # Same endpoint tuple layout as Agent._COMPAT_BACKENDS.
    _COMPAT_BACKENDS = {
        "qwen": (
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        "glm": (
            "ZHIPUAI_API_KEY",
            "GLM_BASE_URL",
            "https://api.z.ai/api/paas/v4/",
        ),
        "kimi": (
            "MOONSHOT_API_KEY",
            "MOONSHOT_BASE_URL",
            "https://api.moonshot.ai/v1",
        ),
        "openrouter": (
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
    }

    def _chat(self, system: str, user: str, max_tokens: int = 2500) -> str:
        """Return stripped response text from the selected provider; SDK exceptions propagate.

        The token budget includes hidden reasoning as well as visible output."""
        backend = self._detect_backend()
        if backend == "openai":
            return self._chat_openai(system, user, max_tokens)
        if backend == "anthropic":
            return self._chat_anthropic(system, user, max_tokens)
        if backend == "gemini":
            return self._chat_gemini(system, user, max_tokens)
        if backend in self._COMPAT_BACKENDS:
            return self._chat_compat(backend, system, user, max_tokens)
        raise ValueError(f"Unknown backend for model {self.model!r}")

    def _chat_openai(self, system: str, user: str, max_tokens: int) -> str:
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(timeout=30)
        # Reasoning models require max_completion_tokens and reject explicit temperature.
        is_reasoning = any(
            self.model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        if not is_reasoning:
            kwargs["temperature"] = 0
        resp = self._openai_client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _chat_compat(self, backend: str, system: str, user: str, max_tokens: int) -> str:
        """Call an OpenAI-compatible endpoint configured by _COMPAT_BACKENDS environment variables."""
        attr = f"_{backend}_client"
        client = getattr(self, attr)
        if client is None:
            import openai
            import os
            key_var, url_var, default_url = self._COMPAT_BACKENDS[backend]
            api_key = os.environ.get(key_var)
            if not api_key:
                raise RuntimeError(
                    f"{key_var} env var is required for {backend} models "
                    f"(``export {key_var}=...`` in your .env)."
                )
            base_url = os.environ.get(url_var, default_url)
            client = openai.OpenAI(
                api_key=api_key, base_url=base_url, timeout=30,
            )
            setattr(self, attr, client)
        # Use temperature 1 for compatibility with thinking variants that require it.
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
            temperature=1,
        )
        return (resp.choices[0].message.content or "").strip()

    def _chat_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(timeout=30)
        resp = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for block in resp.content:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        return "".join(parts).strip()

    def _chat_gemini(self, system: str, user: str, max_tokens: int) -> str:
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client()
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            max_output_tokens=max_tokens,
        )
        resp = self._gemini_client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        return (resp.text or "").strip()

    # -- Error collection ---------------------------------------------------

    # Tracebacks remain unattributed until LLM triage.
    _TRACEBACK_SENTINEL = "__traceback__"

    def collect_from_traces(self, traces: list[dict]) -> list[ErrorPattern]:
        """Collect unseen error patterns; cell tracebacks await LLM attribution."""
        patterns: dict[tuple[str, str], ErrorPattern] = {}

        for trace in traces:
            for turn in trace.get("turns", []):
                for tc in turn.get("tool_calls", []):
                    for p in self._extract_errors(tc):
                        key_method = (
                            self._TRACEBACK_SENTINEL
                            if p.is_traceback
                            else p.method_name
                        )
                        error_key = p.error_message[:100]
                        pattern_key = (key_method, error_key)

                        with self._lock:
                            if pattern_key in self._seen:
                                continue

                        if pattern_key in patterns:
                            patterns[pattern_key].count += 1
                        else:
                            patterns[pattern_key] = p

        return list(patterns.values())

    @staticmethod
    def select_patterns(
        patterns: list[ErrorPattern], max_patterns: int | None,
    ) -> list[ErrorPattern]:
        """Keep the most frequent patterns, breaking ties by first appearance.

        Preserve original order and leave dropped patterns unseen for later batches.
        None disables the cap."""
        if max_patterns is None or len(patterns) <= max_patterns:
            return patterns
        ranked = sorted(range(len(patterns)), key=lambda i: -patterns[i].count)
        keep = sorted(ranked[:max_patterns])
        return [patterns[i] for i in keep]

    @staticmethod
    def _extract_errors(tc: dict) -> list[ErrorPattern]:
        """Extract attributed API errors or unattributed cell traceback patterns from a tool call."""
        tool_name = tc.get("tool_name", "")
        result = tc.get("result")
        args_obj = tc.get("arguments", {}) or {}
        code = args_obj.get("code", "") if isinstance(args_obj, dict) else ""

        # -- Cell / execute wrapper ------------------------------------------
        if tool_name in ("ptc_cell", "ptc_execute"):
            if not isinstance(result, dict):
                return []
            entries: list[ErrorPattern] = []
            for e in result.get("api_errors", []) or []:
                api = e.get("api", "")
                method = e.get("method", "")
                err = str(e.get("error", ""))
                if api and method and err:
                    entries.append(ErrorPattern(
                        method_name=f"{api}.{method}",
                        error_message=err[:400],
                        code_snippet=str(code)[:500],
                    ))
            tb = result.get("error")
            if tb and isinstance(tb, str):
                entries.append(ErrorPattern(
                    method_name="",  # filled in by triage
                    error_message=tb.strip()[:1200],
                    code_snippet=str(code)[:600],
                    is_traceback=True,
                    api_calls=list(result.get("api_calls") or []),
                ))
            return entries

        # -- Direct API tool call (tool_only mode) ---------------------------
        err: str | None = None
        if isinstance(result, dict):
            if result.get("error"):
                err = str(result["error"])
            elif "error" in result:
                err = str(result["error"])
        if not err and tc.get("error"):
            err = str(tc["error"])
        if err:
            return [ErrorPattern(
                method_name=tool_name,
                error_message=err[:400],
                code_snippet=str(code)[:500],
            )]
        return []

    # -- LLM-based improvement ----------------------------------------------

    def improve_docs(
        self,
        error_patterns: list[ErrorPattern],
        tool_defs: list[ToolDef],
    ) -> dict[str, str]:
        """Return {method_name: addendum} for useful documentation improvements."""
        desc_lookup = {td.name: td.description for td in tool_defs}
        available_methods = [td.name for td in tool_defs if "." in td.name]
        # Code-level mistakes can target the execution tool instead of an API method.
        code_exec_tool_names = [
            td.name for td in tool_defs
            if td.name in ("ptc_cell", "ptc_execute")
        ]
        code_exec_tool = code_exec_tool_names[0] if code_exec_tool_names else None

        # Keep every addendum when multiple patterns target the same method.
        new_improvements: dict[str, list[str]] = {}

        for pattern in error_patterns:
            key_method = (
                self._TRACEBACK_SENTINEL
                if pattern.is_traceback
                else pattern.method_name
            )
            key = (key_method, pattern.error_message[:100])

            with self._lock:
                if key in self._seen:
                    continue
                self._seen.add(key)

            if pattern.is_traceback:
                triage = self._llm_triage_traceback(
                    pattern, available_methods, desc_lookup,
                    code_exec_tool=code_exec_tool,
                )
                if triage is None:
                    continue
                target, addendum, source = triage
                # Persist now so later patterns for this method see prior addenda and avoid duplicates.
                with self._lock:
                    self._improvements.setdefault(target, []).append(addendum)
                new_improvements.setdefault(target, []).append(addendum)
                self._log.append({
                    "method": target,
                    "error": pattern.error_message[:200],
                    "addendum": addendum,
                    "version": self._version + 1,
                    "source": source,
                })
                continue

            current_desc = desc_lookup.get(
                pattern.method_name, "No documentation available."
            )
            if len(current_desc) > 1500:
                current_desc = current_desc[:1500] + " …[truncated]"

            addendum = self._llm_improve(pattern, current_desc)

            if addendum and not self._is_no_doc_change(addendum):
                with self._lock:
                    self._improvements.setdefault(
                        pattern.method_name, []
                    ).append(addendum)
                new_improvements.setdefault(
                    pattern.method_name, []
                ).append(addendum)
                self._log.append({
                    "method": pattern.method_name,
                    "error": pattern.error_message[:200],
                    "addendum": addendum,
                    "version": self._version + 1,
                })

        if new_improvements:
            with self._lock:
                self._version += 1

        # Keep the historical last-addendum return shape; state retains all addenda.
        return {m: addenda[-1] for m, addenda in new_improvements.items()}

    @staticmethod
    def _is_no_doc_change(text: str) -> bool:
        """Robust NO_DOC_CHANGE detection (case/punctuation-insensitive)."""
        if not text:
            return True
        normalized = text.strip().upper().rstrip(".!?")
        return normalized == "NO_DOC_CHANGE" or normalized.startswith(
            "NO_DOC_CHANGE "
        ) or normalized.startswith("NO_DOC_CHANGE\n")

    def _llm_improve(self, pattern: ErrorPattern, current_desc: str) -> str | None:
        """Make one LLM call to generate a doc addendum."""
        with self._lock:
            prior = list(self._improvements.get(pattern.method_name, []))

        user_msg = (
            f"## Current documentation for `{pattern.method_name}`\n"
            f"{current_desc}\n\n"
        )
        if prior:
            addenda_block = "\n".join(f"- {a}" for a in prior)
            user_msg += (
                f"## Addenda already learned for this method\n"
                f"{addenda_block}\n\n"
                f"If the lesson from the error below is already covered by any "
                f"addendum above (even if the error text, identifiers, or "
                f"values differ), output exactly NO_DOC_CHANGE. Only emit a "
                f"new addendum when it teaches something genuinely missing "
                f"from the list.\n\n"
            )
        user_msg += (
            f"## Error encountered ({pattern.count} occurrences)\n"
            f"{pattern.error_message}\n\n"
        )
        if pattern.code_snippet:
            user_msg += (
                f"## Code that triggered the error\n"
                f"{pattern.code_snippet[:300]}\n\n"
            )
        user_msg += "## Your addendum (1-2 sentences, or NO_DOC_CHANGE):"

        try:
            return self._chat(
                system=self._SYSTEM_PROMPT,
                user=user_msg,
                # Allow room for internal reasoning as well as the addendum.
                max_tokens=2000,
            )
        except Exception as exc:
            print(f"    DocLearner: LLM call failed for {pattern.method_name}: {exc}")
            return None

    def _llm_triage_traceback(
        self,
        pattern: ErrorPattern,
        available_methods: list[str],
        desc_lookup: dict[str, str],
        code_exec_tool: str | None = None,
    ) -> tuple[str, str, str] | None:
        """Return (target_name, addendum, source) for a cell traceback, or None.

        Sources distinguish API-method traceback_triage from code_exec_addendum."""
        api_call_lines = [
            f"- {c.get('api', '')}.{c.get('method', '')}"
            for c in pattern.api_calls
            if c.get("api") and c.get("method")
        ]
        api_call_summary = "\n".join(api_call_lines) if api_call_lines else "(none)"

        # Scope triage to called methods to bound prompt size; uncalled methods cannot be targets.
        cell_methods: list[str] = []
        seen_methods: set[str] = set()
        for c in pattern.api_calls:
            api = c.get("api", "")
            method = c.get("method", "")
            if api and method:
                full = f"{api}.{method}"
                if full not in seen_methods:
                    seen_methods.add(full)
                    cell_methods.append(full)
        # A crash before any API call leaves no narrower attribution scope.
        if cell_methods:
            methods_block = "\n".join(f"- {m}" for m in cell_methods)
        else:
            methods_block = "\n".join(f"- {m}" for m in sorted(set(available_methods)))
        code_tool_block = (
            f"\n## Code-execution tool available in this run (use with TOOL:)\n"
            f"- {code_exec_tool}\n"
            if code_exec_tool else
            "\n## Code-execution tool available in this run\n(none — only "
            "METHOD: or NO_DOC_CHANGE are valid outputs)\n"
        )

        # Include prior addenda for all candidate methods and the code-execution tool.
        candidate_targets: list[str] = []
        seen_targets: set[str] = set()
        for c in pattern.api_calls:
            api = c.get("api", "")
            method = c.get("method", "")
            if api and method:
                full = f"{api}.{method}"
                if full not in seen_targets:
                    seen_targets.add(full)
                    candidate_targets.append(full)
        if code_exec_tool and code_exec_tool not in seen_targets:
            seen_targets.add(code_exec_tool)
            candidate_targets.append(code_exec_tool)

        with self._lock:
            prior_blocks = []
            for target in candidate_targets:
                addenda = self._improvements.get(target)
                if addenda:
                    body = "\n".join(f"- {a}" for a in addenda)
                    prior_blocks.append(f"### {target}\n{body}")
        prior_section = ""
        if prior_blocks:
            prior_section = (
                "## Addenda already learned for candidate targets\n"
                + "\n\n".join(prior_blocks)
                + "\n\nIf the lesson from this traceback is already covered by "
                "any addendum above for the target you would pick (even if "
                "the error text, identifiers, or values differ), output "
                "exactly NO_DOC_CHANGE. Only emit a new METHOD/TOOL + "
                "ADDENDUM when it teaches something genuinely missing.\n\n"
            )

        user_msg = (
            f"## Traceback\n{pattern.error_message}\n\n"
            f"## Code that ran in the cell\n{pattern.code_snippet}\n\n"
            f"## API methods called in this cell (in order)\n{api_call_summary}\n\n"
            f"## All available API methods you may attribute to (use with METHOD:)\n{methods_block}\n"
            f"{code_tool_block}\n"
            f"{prior_section}"
            f"## Your output (METHOD/ADDENDUM, TOOL/ADDENDUM, or NO_DOC_CHANGE):"
        )

        try:
            text = self._chat(
                system=self._TRIAGE_SYSTEM_PROMPT,
                user=user_msg,
                max_tokens=2500,
            )
        except Exception as exc:
            print(f"    DocLearner: triage LLM call failed: {exc}")
            return None

        if not text or self._is_no_doc_change(text):
            return None

        parsed = self._parse_triage_response(text)
        kind, target, addendum = parsed

        if not target or not addendum:
            return None
        if self._is_no_doc_change(addendum):
            return None

        if kind == "method":
            if target not in available_methods:
                return None
            return target, addendum, "traceback_triage"
        if kind == "tool":
            if code_exec_tool is None or target != code_exec_tool:
                return None
            return target, addendum, "code_exec_addendum"
        return None

    @staticmethod
    def _parse_triage_response(text: str) -> tuple[str, str, str]:
        """Parse the triage response into ``(kind, target, addendum)``.

        ``kind`` is ``"method"``, ``"tool"``, or ``""`` if unparsable.
        """
        import re
        # Prefer METHOD: if both are present; TOOL: for code-exec attribution.
        method_match = re.search(
            r"^\s*METHOD\s*:\s*([^\n]+)$", text, re.MULTILINE | re.IGNORECASE
        )
        tool_match = re.search(
            r"^\s*TOOL\s*:\s*([^\n]+)$", text, re.MULTILINE | re.IGNORECASE
        )
        addendum_match = re.search(
            r"^\s*ADDENDUM\s*:\s*(.+)$", text,
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )
        addendum = addendum_match.group(1).strip() if addendum_match else ""

        def _clean(name: str) -> str:
            return name.strip("`\"'. ")

        if method_match:
            return "method", _clean(method_match.group(1).strip()), addendum
        if tool_match:
            return "tool", _clean(tool_match.group(1).strip()), addendum
        return "", "", addendum

    # -- Applying improvements to tool defs ---------------------------------

    def apply_improvements(self, tool_defs: list[ToolDef]) -> None:
        """Mutate *tool_defs* in place, appending accumulated addenda."""
        with self._lock:
            for td in tool_defs:
                # Snapshot descriptions before mutation for original/final serialization.
                if td.name not in self._originals:
                    self._originals[td.name] = td.description
                notes = self._improvements.get(td.name)
                if notes:
                    section = "\n\nLearned from prior errors:\n" + "\n".join(
                        f"- {note}" for note in notes
                    )
                    if section not in td.description:
                        td.description += section

    def preload(self, path: Any) -> int:
        """Load method addenda from a prior learned_docs.json and return the method count.

        Leave originals unset until apply_improvements snapshots pristine descriptions."""
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"learned docs not found: {p}")
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a dict keyed by method name")
        count = 0
        with self._lock:
            for method, info in data.items():
                if not isinstance(info, dict):
                    continue
                addenda = info.get("addenda") or []
                if not addenda:
                    continue
                self._improvements.setdefault(method, []).extend(
                    str(a) for a in addenda
                )
                count += 1
        return count

    def dump_augmented_docs(self) -> dict[str, dict[str, Any]]:
        """Return reference-only dump of augmented descriptions.

        Only includes methods that actually got addenda. Shape:
          ``{method_name: {"original": str, "addenda": [str, ...], "final": str}}``
        """
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for method, addenda in self._improvements.items():
                original = self._originals.get(method, "")
                section = "\n\nLearned from prior errors:\n" + "\n".join(
                    f"- {note}" for note in addenda
                )
                final = (original + section) if original else section.lstrip()
                result[method] = {
                    "original": original,
                    "addenda": list(addenda),
                    "final": final,
                }
            return result

    # -- State / metadata ---------------------------------------------------

    @property
    def version(self) -> int:
        """Current doc-improvement version (0 = original, unmodified)."""
        with self._lock:
            return self._version

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot for trace metadata."""
        with self._lock:
            return {
                "doc_version": self._version,
                "total_improvements": sum(
                    len(v) for v in self._improvements.values()
                ),
                "methods_improved": sorted(self._improvements.keys()),
                "log": list(self._log),
            }
