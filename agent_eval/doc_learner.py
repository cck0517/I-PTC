"""LLM-based API documentation learner.

Between task batches, this module:
  1. Collects error patterns from completed traces
  2. Calls an LLM to generate targeted doc improvements
  3. Applies improvements to tool definitions for the next batch

Usage is controlled by ``--learning-batch-size B`` on the runner:
  - B = 0 (default): disabled, no learning
  - B = 1: update docs after every task (sequential)
  - B = 5: update docs every 5 tasks
  - B = 55: update docs after all tasks (benefits next run only)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from agent_eval.agent import ToolDef


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ErrorPattern:
    """One deduplicated error pattern extracted from traces.

    When ``is_traceback`` is True, ``method_name`` is not yet known: the
    pattern represents a Python exception that crashed a cell, and the
    triage LLM must decide which API method (if any) should be updated.
    """
    method_name: str
    error_message: str
    code_snippet: str
    count: int = 1
    is_traceback: bool = False
    api_calls: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DocLearner
# ---------------------------------------------------------------------------

class DocLearner:
    """Accumulates error patterns and generates LLM-based doc improvements.

    Thread-safe: multiple worker threads can call :meth:`collect_from_trace`
    concurrently while the main thread calls :meth:`improve_and_apply` between
    batches.
    """

    # The LLM prompt for generating doc improvements on a known method.
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

    # Separate prompt for triaging Python tracebacks inside ptc_cell /
    # ptc_execute. Here the LLM must (a) decide whether any tool's doc
    # should be updated at all, and (b) identify which one.
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
        # Explicit provider for the learner calls (a run_eval --backend
        # name). None → choose from the model name (``_detect_backend``).
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
        # Lazy per-backend clients (only one will be created per learner
        # instance, depending on the model's backend).
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None
        self._qwen_client = None
        self._glm_client = None
        self._kimi_client = None
        self._openrouter_client = None
        # Log of all improvements for the trace record
        self._log: list[dict[str, Any]] = []
        # method_name → pristine description (snapshotted on first sight)
        self._originals: dict[str, str] = {}

    # -- Backend dispatch (explicit backend, else model-name → SDK) --------

    # run_eval --backend names → the learner's SDK routes.
    _BACKEND_ALIASES = {
        "claude": "anthropic",
        "claude_ptc": "anthropic",
        "openai_native": "openai",
        "gemini_native": "gemini",
    }

    def _detect_backend(self) -> str:
        """Return the explicit ``self.backend`` if set, else map
        ``self.model`` to one of: ``openai``, ``anthropic``, ``gemini``,
        ``qwen``, ``glm``, ``kimi``.

        The name rules mirror the prefixes used by ``Agent``. Falls back to
        ``openai`` for unknown prefixes (preserves prior default behavior).
        """
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
        # Kimi models from Moonshot are usually named "kimi-*" or
        # "moonshot-*" — accept both so users can pass either spelling.
        if name.startswith("kimi") or name.startswith("moonshot"):
            return "kimi"
        # gpt-*, o1/o3/o4-*, anything else → openai
        return "openai"

    # OpenAI-compatible third-party providers used by the doc-learner.
    # Same shape as Agent._COMPAT_BACKENDS (key_env, base_url_env, default_url).
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
        """Unified chat call across openai/anthropic/gemini/qwen/glm/kimi/openrouter.

        Returns the response text (stripped). Raises whatever exception the
        underlying SDK raised — callers catch and log.

        ``max_tokens`` is large by default to cover any backend that may
        emit hidden reasoning/thinking tokens (gpt-5*, o1/o3/o4*, gemini-2.5
        with thinking, claude with extended thinking, qwen3 thinking,
        glm-4.5/4.6 thinking, kimi-thinking).
        Visible-output-only models will use less than this.
        """
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
        # Reasoning models (gpt-5*, o1/o3/o4*) only accept
        # ``max_completion_tokens`` and reject explicit temperature.
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
        """OpenAI-SDK call against an OpenAI-compatible third-party endpoint.

        Handles ``qwen`` (DashScope), ``glm`` (Zhipu/Z.ai), ``kimi``
        (Moonshot) and ``openrouter``. API key + optional base URL are read
        from env vars listed in ``_COMPAT_BACKENDS``.
        """
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
        # Some thinking variants (e.g. kimi-k2.5-thinking) reject any
        # temperature other than 1, so use 1 unconditionally for compat
        # backends. Determinism is sacrificed for portability.
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
        # Concatenate any text blocks in the response (Anthropic returns a
        # list of content blocks; usually one text block for chat-style use).
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

    # Sentinel used in dedup keys for traceback patterns, which have no
    # attributed method until the triage LLM runs.
    _TRACEBACK_SENTINEL = "__traceback__"

    def collect_from_traces(self, traces: list[dict]) -> list[ErrorPattern]:
        """Extract new, deduplicated error patterns from a batch of traces.

        Only returns patterns not yet seen by this learner instance.
        API-returned errors are attributed directly to the underlying API
        method. Python tracebacks inside ``ptc_cell`` / ``ptc_execute`` are
        emitted as ``is_traceback=True`` patterns for the LLM to triage.
        """
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
        """Keep at most ``max_patterns`` patterns, preferring the most frequent.

        Each pattern costs one sequential LLM call in ``improve_docs``, and
        error messages that embed guessed values (e.g. a different name in
        every "not found" error) can produce hundreds of patterns in one
        batch. Returns *patterns* unchanged when ``max_patterns`` is None or
        the list is within the cap. Otherwise keeps the ``max_patterns``
        patterns with the highest ``count`` (ties: first seen), in their
        original order. Dropped patterns are not marked as seen, so one that
        recurs can still be learned in a later batch.
        """
        if max_patterns is None or len(patterns) <= max_patterns:
            return patterns
        ranked = sorted(range(len(patterns)), key=lambda i: -patterns[i].count)
        keep = sorted(ranked[:max_patterns])
        return [patterns[i] for i in keep]

    @staticmethod
    def _extract_errors(tc: dict) -> list[ErrorPattern]:
        """Return ``ErrorPattern`` objects for a tool-call record.

        - API-returned error dicts → patterns with the specific ``api.method``.
        - Python tracebacks inside a cell → a single ``is_traceback=True``
          pattern carrying the traceback, code, and list of APIs called.
          The triage LLM decides which method (if any) needs a doc update.
        - Direct API tool calls → one pattern named by the tool itself.
        """
        tool_name = tc.get("tool_name", "")
        result = tc.get("result")
        args_obj = tc.get("arguments", {}) or {}
        code = args_obj.get("code", "") if isinstance(args_obj, dict) else ""

        # -- Cell / execute wrapper ------------------------------------------
        if tool_name in ("ptc_cell", "ptc_execute"):
            if not isinstance(result, dict):
                return []
            entries: list[ErrorPattern] = []
            # 1. Precise: API-returned error dicts captured by the proxy.
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
            # 2. Python traceback: defer attribution to the triage LLM.
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
        """Generate LLM-based doc improvements for the given error patterns.

        Returns a dict of ``{method_name: addendum_text}`` for patterns that
        produced a useful improvement. Patterns that the LLM deems
        non-doc-related (e.g., agent typos) are skipped.
        """
        desc_lookup = {td.name: td.description for td in tool_defs}
        # Valid API-method targets for the triage LLM.
        available_methods = [td.name for td in tool_defs if "." in td.name]
        # Code-execution tool (if any) — used when the triage LLM attributes
        # a pattern to code-level mistakes rather than a specific API method.
        code_exec_tool_names = [
            td.name for td in tool_defs
            if td.name in ("ptc_cell", "ptc_execute")
        ]
        code_exec_tool = code_exec_tool_names[0] if code_exec_tool_names else None

        # method_name → list of addenda generated this batch. Using a list
        # (instead of a dict[str, str]) preserves every addendum even when
        # multiple errors in the same batch target the same method.
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
                # Write immediately so the next pattern in this batch — which
                # might target the same method — sees this addendum in its
                # "already learned" list and can bail with NO_DOC_CHANGE.
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

        # Preserve a flat {method: last_addendum} summary for callers/logs,
        # matching the historical return shape even though multiple addenda
        # per method are now fully accumulated into ``self._improvements``.
        return {m: addenda[-1] for m, addenda in new_improvements.items()}

    @staticmethod
    def _is_no_doc_change(text: str) -> bool:
        """Robust NO_DOC_CHANGE detection (case/punctuation-insensitive)."""
        if not text:
            return True
        normalized = text.strip().upper().rstrip(".!?")
        # Exact or leading token.
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
                # Budget covers both internal reasoning (gpt-5*, gemini-2.5
                # thinking, claude extended thinking) and the short addendum.
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
        """Ask the LLM which target (if any) needs a doc update for a cell
        that crashed with a Python traceback. Returns
        ``(target_name, addendum, source)`` where ``source`` is either
        ``"traceback_triage"`` (API-method attribution) or
        ``"code_exec_addendum"`` (attribution to ptc_cell/ptc_execute).
        Returns ``None`` for NO_DOC_CHANGE / parse failure.
        """
        api_call_lines = [
            f"- {c.get('api', '')}.{c.get('method', '')}"
            for c in pattern.api_calls
            if c.get("api") and c.get("method")
        ]
        api_call_summary = "\n".join(api_call_lines) if api_call_lines else "(none)"

        # Candidate attribution scope: only the methods actually called in
        # this cell. Keeps the prompt small (was the full ~78-method catalog
        # before — ~4-5k tokens of pure listing per call). Trade-off: the
        # LLM can no longer attribute to a method the agent SHOULD have
        # called but didn't. Acceptable in practice — the most common
        # triage outcome is "this method's doc needs an addendum about
        # how to call it correctly," and that method is by definition in
        # the cell's call list.
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
        # Fall back to the full catalog only if the cell crashed before
        # making any API call — in that case we have nothing else to scope to.
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

        # Gather prior addenda for every target the triage LLM might pick:
        # the methods that actually ran in this cell plus the code-exec tool.
        # Keeps the prompt bounded (cell method list is small) while still
        # letting the LLM see what's already been taught for any likely target.
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
                # Snapshot the pristine description the first time we see
                # this method, so we can serialize original/final pairs later.
                if td.name not in self._originals:
                    self._originals[td.name] = td.description
                notes = self._improvements.get(td.name)
                if notes:
                    section = "\n\nLearned from prior errors:\n" + "\n".join(
                        f"- {note}" for note in notes
                    )
                    # Only append if not already there (idempotency).
                    if section not in td.description:
                        td.description += section

    def preload(self, path: Any) -> int:
        """Seed ``_improvements`` from a prior run's ``learned_docs.json``.

        Accepts the output of :meth:`dump_augmented_docs` (a dict keyed by
        method name with an ``"addenda"`` list). Returns the number of
        methods loaded. Leaves ``_originals`` empty so the next call to
        :meth:`apply_improvements` still snapshots pristine descriptions.
        """
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
