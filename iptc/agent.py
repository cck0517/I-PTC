"""Multi-turn LLM tool-calling loops with structured traces and token accounting."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

@dataclass
class TurnRecord:
    role: str
    content: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class Trace:
    task_id: str
    model: str
    system_prompt: str
    turns: list[TurnRecord] = field(default_factory=list)
    final_answer: str | None = None
    turn_answers: list[str | None] = field(default_factory=list)  # per-turn answers for multi-turn
    stop_reason: str | None = None  # "completed", "max_turns", "token_budget", "length_cap", "api_error", "api_moderation"
    total_time_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        def _tc(tc: ToolCallRecord) -> dict:
            d: dict[str, Any] = {"tool_name": tc.tool_name, "arguments": tc.arguments}
            if tc.result is not None:
                d["result"] = tc.result
            if tc.error is not None:
                d["error"] = tc.error
            return d

        return {
            "task_id": self.task_id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "turns": [
                {
                    "role": t.role,
                    **({"content": t.content} if t.content else {}),
                    **({"tool_calls": [_tc(c) for c in t.tool_calls]} if t.tool_calls else {}),
                    "latency_s": round(t.latency_s, 3),
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                }
                for t in self.turns
            ],
            "final_answer": self.final_answer,
            **({"turn_answers": self.turn_answers} if self.turn_answers else {}),
            "stop_reason": self.stop_reason,
            "total_time_s": round(self.total_time_s, 3),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            **({"error": self.error} if self.error else {}),
        }


@dataclass
class ToolDef:
    """Provider-agnostic tool definition."""
    name: str
    description: str
    parameters: dict[str, Any]


def _gemini_token_counts(usage_metadata: Any) -> tuple[int, int]:
    """Return input/output counts, including separately reported thinking tokens in output."""
    if not usage_metadata:
        return 0, 0
    prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
    candidates = getattr(usage_metadata, "candidates_token_count", 0) or 0
    thoughts = getattr(usage_metadata, "thoughts_token_count", 0) or 0
    return prompt, candidates + thoughts


def _tool_defs_to_openai(tools: list[ToolDef]) -> tuple[list[dict], dict[str, str]]:
    """Return OpenAI tools and a sanitized-to-original name map."""
    out = []
    name_map: dict[str, str] = {}  # sanitized → original
    for t in tools:
        sanitized_name = t.name.replace(".", "_")
        name_map[sanitized_name] = t.name

        params = dict(t.parameters)
        params.setdefault("type", "object")
        params.pop("required", None)
        properties = params.get("properties", {})
        required = []
        for pname, pschema in properties.items():
            pschema.pop("description", None) if False else None
            if pschema.pop("required", False):
                required.append(pname)
        if "required" in t.parameters:
            required = t.parameters["required"]
        clean_props = {}
        for pname, pschema in properties.items():
            clean = {k: v for k, v in pschema.items() if k != "required"}
            clean_props[pname] = clean
        schema = {"type": "object", "properties": clean_props}
        if required:
            schema["required"] = required
        out.append({
            "type": "function",
            "function": {
                "name": sanitized_name,
                "description": t.description,
                "parameters": schema,
            },
        })
    return out, name_map


def _tool_defs_to_gemini(tools: list[ToolDef]) -> tuple[list[dict], dict[str, str]]:
    """Convert ToolDefs to Gemini-style function declarations.

    Returns (declarations, name_map) where name_map maps sanitized names
    back to originals (dots replaced with underscores).
    """
    declarations = []
    name_map: dict[str, str] = {}
    for t in tools:
        sanitized_name = t.name.replace(".", "_")
        name_map[sanitized_name] = t.name

        params = dict(t.parameters)
        properties = params.get("properties", {})
        required = params.get("required", [])
        gemini_props = {}
        for pname, pschema in properties.items():
            gp: dict[str, Any] = {}
            ptype = pschema.get("type", "string")
            type_map = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER",
                        "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"}
            gp["type"] = type_map.get(ptype, "STRING")
            if "description" in pschema:
                gp["description"] = pschema["description"]
            if ptype == "array" and "items" in pschema:
                items = pschema["items"]
                item_type = items.get("type", "string")
                gp["items"] = {"type": type_map.get(item_type, "STRING")}
            gemini_props[pname] = gp
        decl: dict[str, Any] = {
            "name": sanitized_name,
            "description": t.description,
        }
        if gemini_props:
            decl["parameters"] = {
                "type": "OBJECT",
                "properties": gemini_props,
            }
            if required:
                decl["parameters"]["required"] = required
        declarations.append(decl)
    return declarations, name_map


def _tool_defs_to_claude(tools: list[ToolDef]) -> tuple[list[dict], dict[str, str]]:
    """Convert ToolDefs to Anthropic tools format.

    Returns (tools_list, name_map) where name_map maps sanitized names back
    to original names. Claude requires names to match ``^[a-zA-Z0-9_-]{1,64}$``.
    """
    out = []
    name_map: dict[str, str] = {}
    for t in tools:
        sanitized_name = t.name.replace(".", "_")
        name_map[sanitized_name] = t.name

        params = dict(t.parameters)
        properties = params.get("properties", {})
        required = params.get("required", [])

        clean_props = {}
        for pname, pschema in properties.items():
            clean = {k: v for k, v in pschema.items() if k != "required"}
            clean_props[pname] = clean

        schema: dict[str, Any] = {"type": "object", "properties": clean_props}
        if required:
            schema["required"] = required

        out.append({
            "name": sanitized_name,
            "description": t.description,
            "input_schema": schema,
        })
    return out, name_map


def _tool_defs_to_claude_ptc(tools: list[ToolDef]) -> tuple[list[dict], dict[str, str]]:
    """Return native Claude PTC tools and a name map, replacing benchmark code tools."""
    out: list[dict] = []
    name_map: dict[str, str] = {}

    out.append({
        "type": "code_execution_20260120",
        "name": "code_execution",
    })

    for t in tools:
        # Skip the benchmark's code-execution tools — Claude's native PTC replaces them
        if t.name in ("ptc_execute", "ptc_cell"):
            continue

        sanitized_name = t.name.replace(".", "_")
        name_map[sanitized_name] = t.name

        params = dict(t.parameters)
        properties = params.get("properties", {})
        required = params.get("required", [])

        clean_props = {}
        for pname, pschema in properties.items():
            clean = {k: v for k, v in pschema.items() if k != "required"}
            clean_props[pname] = clean

        schema: dict[str, Any] = {"type": "object", "properties": clean_props}
        if required:
            schema["required"] = required

        out.append({
            "name": sanitized_name,
            "description": t.description,
            "input_schema": schema,
            "allowed_callers": ["code_execution_20260120"],
        })
    return out, name_map


ToolExecutor = Callable[[str, dict[str, Any]], Any]

_VALID_BACKENDS = (
    "openai", "gemini", "claude", "claude_ptc", "qwen", "glm", "kimi",
    "openrouter",
    # OpenAI/Gemini sandboxes cannot call developer tools; interleave them at the model level.
    "openai_native",
    "gemini_native",
)


class Agent:
    """
    Minimal agent loop that sends messages to an LLM, executes tool calls,
    and records a full trace.
    """

    def __init__(
        self,
        backend: str,
        model: str,
        tool_defs: list[ToolDef],
        tool_executor: ToolExecutor,
        system_prompt: str = "",
        max_turns: int = 30,
        # None leaves sampling behavior to the backend.
        temperature: float | None = None,
        max_token_budget: int | None = None,
    ):
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unknown backend: {backend!r}")
        self.backend = backend
        self.model = model
        self.tool_defs = tool_defs
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_token_budget = max_token_budget

    def _over_token_budget(self, trace: Trace) -> bool:
        """Return True if cumulative tokens exceed the budget."""
        if self.max_token_budget is None:
            return False
        return (trace.total_input_tokens + trace.total_output_tokens) >= self.max_token_budget

    # -- public entry point --------------------------------------------------

    def run(self, user_message: str, task_id: str = "",
            extra_user_messages: list[str] | None = None) -> Trace:
        if self.backend == "openai":
            return self._run_openai(user_message, task_id, extra_user_messages)
        elif self.backend == "gemini":
            return self._run_gemini(user_message, task_id, extra_user_messages)
        elif self.backend == "claude":
            return self._run_claude(user_message, task_id, extra_user_messages)
        elif self.backend == "claude_ptc":
            return self._run_claude_ptc(user_message, task_id, extra_user_messages)
        elif self.backend in ("qwen", "glm", "kimi", "openrouter"):
            return self._run_openai(user_message, task_id, extra_user_messages)
        elif self.backend == "openai_native":
            return self._run_openai_native(user_message, task_id, extra_user_messages)
        elif self.backend == "gemini_native":
            return self._run_gemini_native(user_message, task_id, extra_user_messages)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    # -- OpenAI / OpenAI-compatible backends --------------------------------

    _openai_client = None
    _qwen_client = None
    _glm_client = None
    _kimi_client = None
    _openrouter_client = None

    # Compatible endpoints: (API-key env, base-URL env, default URL).
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

    @classmethod
    def _get_openai_client(cls, backend: str = "openai"):
        """Build an OpenAI or compatible-provider client using configured endpoint environment variables."""
        import openai
        import os
        if backend in cls._COMPAT_BACKENDS:
            attr = f"_{backend}_client"
            client = getattr(cls, attr)
            if client is None:
                key_var, url_var, default_url = cls._COMPAT_BACKENDS[backend]
                api_key = os.environ.get(key_var)
                if not api_key:
                    raise RuntimeError(
                        f"{key_var} env var is required for --backend {backend} "
                        f"(``export {key_var}=...`` in your .env)."
                    )
                base_url = os.environ.get(url_var, default_url)
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    max_retries=3,
                    timeout=120,
                )
                setattr(cls, attr, client)
            return client
        if cls._openai_client is None:
            cls._openai_client = openai.OpenAI(max_retries=3, timeout=120)
        return cls._openai_client

    def _run_openai(self, user_message: str, task_id: str,
                     extra_user_messages: list[str] | None = None) -> Trace:
        import openai  # local import so `openai.BadRequestError` resolves in the except clause below
        client = self._get_openai_client(self.backend)
        trace = Trace(task_id=task_id, model=self.model, system_prompt=self.system_prompt)
        t0 = time.time()

        pending_user_messages = list(extra_user_messages) if extra_user_messages else []

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        trace.turns.append(TurnRecord(role="user", content=user_message))

        oai_tools = None
        name_map: dict[str, str] = {}
        if self.tool_defs:
            oai_tools, name_map = _tool_defs_to_openai(self.tool_defs)

        for _ in range(self.max_turns):
            try:
                t1 = time.time()
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    # Cap reasoning and visible output together across backends.
                    "max_completion_tokens": 8000,
                }
                # Only send explicit temperatures; reasoning models may reject the parameter.
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if oai_tools:
                    kwargs["tools"] = oai_tools
                try:
                    resp = client.chat.completions.create(**kwargs)
                except openai.BadRequestError as e:
                    if "temperature" in str(e):
                        kwargs.pop("temperature", None)
                        resp = client.chat.completions.create(**kwargs)
                    else:
                        raise
                latency = time.time() - t1
            except Exception as exc:
                trace.error = f"OpenAI API error: {exc}"
                # Keep moderation rejections distinct from other API errors in trace analysis.
                err_str = str(exc)
                if "invalid_prompt" in err_str or "usage policy" in err_str:
                    trace.stop_reason = "api_moderation"
                else:
                    trace.stop_reason = "api_error"
                break

            choice = resp.choices[0]
            msg = choice.message

            # Thinking providers require reasoning_content echoed on subsequent assistant messages.
            reasoning_content = getattr(msg, "reasoning_content", None)

            # Length-capped reasoning responses may have empty text or truncated tool arguments.
            hit_length_cap = choice.finish_reason == "length"

            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0

            if msg.tool_calls:
                turn = TurnRecord(
                    role="assistant", content=msg.content, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
                messages.append(assistant_msg)

                for tc in msg.tool_calls:
                    sanitized_name = tc.function.name
                    original_name = name_map.get(sanitized_name, sanitized_name)
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"_raw": tc.function.arguments}

                    record = ToolCallRecord(tool_name=original_name, arguments=args)
                    try:
                        result = self.tool_executor(original_name, args)
                        record.result = result
                    except Exception as exc:
                        record.error = f"{type(exc).__name__}: {exc}"
                        result = {"error": record.error}

                    turn.tool_calls.append(record)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok

                # Length-capped tool arguments may be malformed; stop instead of retrying the loop.
                if hit_length_cap:
                    trace.stop_reason = "length_cap"
                    break

                if self._over_token_budget(trace):
                    trace.stop_reason = "token_budget"
                    break

            else:
                turn = TurnRecord(
                    role="assistant", content=msg.content, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok
                # Some providers reject null assistant content when a multi-turn conversation continues.
                asst_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if reasoning_content:
                    asst_msg["reasoning_content"] = reasoning_content
                messages.append(asst_msg)

                if hit_length_cap:
                    trace.final_answer = msg.content
                    trace.stop_reason = "length_cap"
                    break

                if pending_user_messages:
                    trace.turn_answers.append(msg.content)
                    next_msg = pending_user_messages.pop(0)
                    messages.append({"role": "user", "content": next_msg})
                    trace.turns.append(TurnRecord(role="user", content=next_msg))
                    continue

                trace.final_answer = msg.content
                trace.stop_reason = "completed"
                break

            if choice.finish_reason == "stop":
                break
        else:
            if trace.stop_reason is None:
                trace.stop_reason = "max_turns"

        if trace.stop_reason is None:
            trace.stop_reason = "max_turns"
        trace.total_time_s = time.time() - t0
        return trace

    # -- Gemini backend ------------------------------------------------------

    def _run_gemini(self, user_message: str, task_id: str,
                     extra_user_messages: list[str] | None = None) -> Trace:
        from google import genai
        from google.genai import types

        client = genai.Client()
        trace = Trace(task_id=task_id, model=self.model, system_prompt=self.system_prompt)
        t0 = time.time()

        pending_user_messages = list(extra_user_messages) if extra_user_messages else []

        gemini_decls = []
        name_map: dict[str, str] = {}
        if self.tool_defs:
            gemini_decls, name_map = _tool_defs_to_gemini(self.tool_defs)
        tools_arg = None
        if gemini_decls:
            tools_arg = types.Tool(function_declarations=gemini_decls)

        contents: list[types.Content] = []
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
        trace.turns.append(TurnRecord(role="user", content=user_message))

        # Match the cross-backend output cap and leave temperature unset by default.
        config_kwargs: dict[str, Any] = {
            "system_instruction": self.system_prompt if self.system_prompt else None,
            "tools": [tools_arg] if tools_arg else None,
            "max_output_tokens": 8000,
        }
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        config = types.GenerateContentConfig(**config_kwargs)

        max_retries = 30
        base_delay = 2.0

        for _ in range(self.max_turns):
            response = None
            for attempt in range(max_retries):
                try:
                    t1 = time.time()
                    response = client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                    latency = time.time() - t1
                    break
                except Exception as exc:
                    err_str = str(exc)
                    retryable = any(k in err_str for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"))
                    if retryable and attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), 60)
                        print(f"    Gemini retry {attempt + 1}/{max_retries} after {delay:.0f}s: {err_str[:120]}")
                        time.sleep(delay)
                    else:
                        trace.error = f"Gemini API error after {attempt + 1} attempts: {exc}\n{traceback.format_exc()}"
                        trace.stop_reason = "api_error"
                        break

            if response is None:
                if trace.stop_reason is None:
                    trace.stop_reason = "api_error"
                break

            if not response.candidates:
                trace.error = "Gemini returned no candidates"
                trace.stop_reason = "api_error"
                break

            candidate = response.candidates[0]
            if candidate.content is None or candidate.content.parts is None:
                reason = getattr(candidate, "finish_reason", None)
                reason_name = getattr(reason, "name", str(reason)) if reason else "unknown"
                if reason_name == "MALFORMED_FUNCTION_CALL":
                    error_msg = (
                        "Your previous function call was malformed and could not be parsed. "
                        "Please try again with a correctly formatted function call."
                    )
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=error_msg)]))
                    turn = TurnRecord(role="assistant", content=f"[malformed function call]", latency_s=latency)
                    trace.turns.append(turn)
                    continue
                trace.error = f"Gemini returned empty content (finish_reason={reason})"
                trace.stop_reason = "api_error"
                break

            in_tok, out_tok = _gemini_token_counts(getattr(response, "usage_metadata", None))

            parts = candidate.content.parts

            fn_calls = [p for p in parts if p.function_call]
            text_parts = [p for p in parts if p.text]

            if fn_calls:
                text_content = " ".join(p.text for p in text_parts) if text_parts else None
                turn = TurnRecord(
                    role="assistant", content=text_content, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )

                contents.append(candidate.content)

                fn_response_parts: list[types.Part] = []
                for fc_part in fn_calls:
                    fc = fc_part.function_call
                    sanitized_fname = fc.name
                    fname = name_map.get(sanitized_fname, sanitized_fname)
                    args = dict(fc.args) if fc.args else {}

                    record = ToolCallRecord(tool_name=fname, arguments=args)
                    try:
                        result = self.tool_executor(fname, args)
                        record.result = result
                    except Exception as exc:
                        record.error = f"{type(exc).__name__}: {exc}"
                        result = {"error": record.error}

                    turn.tool_calls.append(record)

                    if not isinstance(result, dict):
                        result = {"result": result}
                    fn_response_parts.append(
                        types.Part.from_function_response(name=sanitized_fname, response=result)
                    )

                contents.append(types.Content(role="user", parts=fn_response_parts))
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok

                if self._over_token_budget(trace):
                    trace.stop_reason = "token_budget"
                    break
            else:
                full_text = " ".join(p.text for p in text_parts) if text_parts else ""
                turn = TurnRecord(
                    role="assistant", content=full_text, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok
                contents.append(candidate.content)

                if pending_user_messages:
                    trace.turn_answers.append(full_text)
                    next_msg = pending_user_messages.pop(0)
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=next_msg)]))
                    trace.turns.append(TurnRecord(role="user", content=next_msg))
                    continue

                trace.final_answer = full_text
                trace.stop_reason = "completed"
                break

            if candidate.finish_reason and candidate.finish_reason.name == "STOP" and not fn_calls:
                break

        if trace.stop_reason is None:
            trace.stop_reason = "max_turns"
        trace.total_time_s = time.time() - t0
        return trace

    # -- Claude backend (standard tool calling) ------------------------------

    def _run_claude(self, user_message: str, task_id: str,
                     extra_user_messages: list[str] | None = None) -> Trace:
        import anthropic

        client = anthropic.Anthropic()
        trace = Trace(task_id=task_id, model=self.model, system_prompt=self.system_prompt)
        t0 = time.time()

        pending_user_messages = list(extra_user_messages) if extra_user_messages else []

        claude_tools, name_map = _tool_defs_to_claude(self.tool_defs)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]
        trace.turns.append(TurnRecord(role="user", content=user_message))

        max_retries = 10
        base_delay = 2.0

        for _ in range(self.max_turns):
            response = None
            for attempt in range(max_retries):
                try:
                    t1 = time.time()
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "max_tokens": 8000,
                        "messages": messages,
                    }
                    if self.temperature is not None:
                        kwargs["temperature"] = self.temperature
                    if self.system_prompt:
                        kwargs["system"] = self.system_prompt
                    if claude_tools:
                        kwargs["tools"] = claude_tools
                    response = client.messages.create(**kwargs)
                    latency = time.time() - t1
                    break
                except Exception as exc:
                    err_str = str(exc)
                    retryable = any(k in err_str for k in
                                    ("529", "529", "overloaded", "rate_limit", "timeout", "503"))
                    if retryable and attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), 120)
                        print(f"    Claude retry {attempt + 1}/{max_retries} after {delay:.0f}s: {err_str[:120]}")
                        time.sleep(delay)
                    else:
                        trace.error = f"Claude API error after {attempt + 1} attempts: {exc}"
                        trace.stop_reason = "api_error"
                        break

            if response is None:
                if trace.stop_reason is None:
                    trace.stop_reason = "api_error"
                break

            in_tok = response.usage.input_tokens if response.usage else 0
            out_tok = response.usage.output_tokens if response.usage else 0

            text_parts = []
            tool_use_blocks = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            text_content = "\n".join(text_parts) if text_parts else None

            if tool_use_blocks:
                turn = TurnRecord(
                    role="assistant", content=text_content, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )

                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict[str, Any]] = []
                for tb in tool_use_blocks:
                    sanitized_name = tb.name
                    original_name = name_map.get(sanitized_name, sanitized_name)
                    args = dict(tb.input) if tb.input else {}

                    record = ToolCallRecord(tool_name=original_name, arguments=args)
                    try:
                        result = self.tool_executor(original_name, args)
                        record.result = result
                    except Exception as exc:
                        record.error = f"{type(exc).__name__}: {exc}"
                        result = {"error": record.error}

                    turn.tool_calls.append(record)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": result_str,
                    })

                messages.append({"role": "user", "content": tool_results})
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok

                if self._over_token_budget(trace):
                    trace.stop_reason = "token_budget"
                    break

            else:
                turn = TurnRecord(
                    role="assistant", content=text_content, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok

                if pending_user_messages:
                    trace.turn_answers.append(text_content)
                    messages.append({"role": "assistant", "content": response.content})
                    next_msg = pending_user_messages.pop(0)
                    messages.append({"role": "user", "content": next_msg})
                    trace.turns.append(TurnRecord(role="user", content=next_msg))
                    continue

                trace.final_answer = text_content
                trace.stop_reason = "completed"
                break

            if response.stop_reason == "end_turn":
                break

        if trace.stop_reason is None:
            trace.stop_reason = "max_turns"
        trace.total_time_s = time.time() - t0
        return trace

    # -- Claude PTC backend (programmatic tool calling) ----------------------

    # -- OpenAI native code interpreter --------------------------------------

    def _run_openai_native(self, user_message: str, task_id: str,
                            extra_user_messages: list[str] | None = None) -> Trace:
        """Interleave hosted code_interpreter execution with model-level API function calls."""
        import openai
        client = self._get_openai_client("openai")
        trace = Trace(task_id=task_id, model=self.model, system_prompt=self.system_prompt)
        t0 = time.time()

        pending_user_messages = list(extra_user_messages) if extra_user_messages else []

        own_tools = [t for t in self.tool_defs if t.name not in ("ptc_execute", "ptc_cell")]
        chat_tools, name_map = _tool_defs_to_openai(own_tools)
        # Responses API uses a flat function-tool shape.
        fn_tools = [
            {
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {}),
            }
            for t in chat_tools
        ]
        tools = [{"type": "code_interpreter", "container": {"type": "auto"}}] + fn_tools

        trace.turns.append(TurnRecord(role="user", content=user_message))
        input_items: Any = [{"role": "user", "content": user_message}]
        prev_response_id: str | None = None
        max_retries = 8

        for _ in range(self.max_turns):
            if self._over_token_budget(trace):
                trace.stop_reason = "token_budget"
                break
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": input_items,
                "tools": tools,
                "instructions": self.system_prompt or None,
                "store": True,
                "max_output_tokens": 8000,
            }
            if prev_response_id:
                kwargs["previous_response_id"] = prev_response_id
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            response = None
            for attempt in range(max_retries):
                try:
                    t1 = time.time()
                    response = client.responses.create(**kwargs)
                    latency = time.time() - t1
                    break
                except Exception as exc:
                    err = str(exc)
                    retryable = any(k in err for k in (
                        "429", "500", "502", "503", "rate", "overloaded", "timeout",
                    ))
                    if retryable and attempt < max_retries - 1:
                        time.sleep(min(2.0 * (2 ** attempt), 60))
                    else:
                        trace.error = f"OpenAI Responses API error: {exc}\n{traceback.format_exc()}"
                        trace.stop_reason = "api_error"
                        break
            if response is None:
                break

            prev_response_id = response.id
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            trace.total_input_tokens += in_tok
            trace.total_output_tokens += out_tok

            turn = TurnRecord(role="assistant", latency_s=latency,
                              input_tokens=in_tok, output_tokens=out_tok)
            fn_outputs: list[dict] = []
            text_chunks: list[str] = []

            for item in (response.output or []):
                itype = getattr(item, "type", None)
                if itype == "function_call":
                    sanitized = item.name
                    fname = name_map.get(sanitized, sanitized)
                    try:
                        args = json.loads(item.arguments or "{}")
                    except Exception:
                        args = {"_raw": item.arguments}
                    record = ToolCallRecord(tool_name=fname, arguments=args)
                    try:
                        result = self.tool_executor(fname, args)
                        record.result = result
                    except Exception as exc:
                        record.error = f"{type(exc).__name__}: {exc}"
                        result = {"error": record.error}
                    turn.tool_calls.append(record)
                    fn_outputs.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result, default=str),
                    })
                elif itype == "code_interpreter_call":
                    # Hosted execution needs tracing but no local dispatch.
                    outputs = getattr(item, "outputs", None) or []
                    rendered = []
                    for o in outputs:
                        rendered.append(getattr(o, "logs", None) or str(o))
                    turn.tool_calls.append(ToolCallRecord(
                        tool_name="code_interpreter",
                        arguments={"code": getattr(item, "code", "") or ""},
                        result={"outputs": rendered[:10]},
                    ))
                elif itype == "message":
                    for c in (getattr(item, "content", None) or []):
                        if getattr(c, "type", "") == "output_text":
                            text_chunks.append(c.text)

            turn.content = "\n".join(text_chunks) if text_chunks else None
            trace.turns.append(turn)

            if fn_outputs:
                input_items = fn_outputs
                continue
            if pending_user_messages:
                trace.turn_answers.append(turn.content)
                next_msg = pending_user_messages.pop(0)
                trace.turns.append(TurnRecord(role="user", content=next_msg))
                input_items = [{"role": "user", "content": next_msg}]
                continue
            trace.final_answer = turn.content
            if trace.turn_answers:
                trace.turn_answers.append(turn.content)
            trace.stop_reason = "completed"
            break
        else:
            trace.stop_reason = "max_turns"

        if trace.stop_reason is None and trace.error is None:
            trace.stop_reason = trace.stop_reason or "max_turns"
        trace.total_time_s = time.time() - t0
        return trace

    # -- Gemini native code execution ----------------------------------------

    def _run_gemini_native(self, user_message: str, task_id: str,
                            extra_user_messages: list[str] | None = None) -> Trace:
        """Interleave hosted code_execution with model-level API function calls."""
        from google import genai
        from google.genai import types

        client = genai.Client()
        trace = Trace(task_id=task_id, model=self.model, system_prompt=self.system_prompt)
        t0 = time.time()

        pending_user_messages = list(extra_user_messages) if extra_user_messages else []

        own_tools = [t for t in self.tool_defs if t.name not in ("ptc_execute", "ptc_cell")]
        gemini_decls, name_map = _tool_defs_to_gemini(own_tools) if own_tools else ([], {})
        tools_list: list[types.Tool] = [types.Tool(code_execution=types.ToolCodeExecution())]
        if gemini_decls:
            tools_list.append(types.Tool(function_declarations=gemini_decls))

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        ]
        trace.turns.append(TurnRecord(role="user", content=user_message))

        config_kwargs: dict[str, Any] = {
            "system_instruction": self.system_prompt if self.system_prompt else None,
            "tools": tools_list,
            "max_output_tokens": 8000,
        }
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        config = types.GenerateContentConfig(**config_kwargs)

        max_retries = 30
        base_delay = 2.0

        for _ in range(self.max_turns):
            if self._over_token_budget(trace):
                trace.stop_reason = "token_budget"
                break
            response = None
            for attempt in range(max_retries):
                try:
                    t1 = time.time()
                    response = client.models.generate_content(
                        model=self.model, contents=contents, config=config,
                    )
                    latency = time.time() - t1
                    break
                except Exception as exc:
                    err_str = str(exc)
                    retryable = any(k in err_str for k in (
                        "429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded",
                    ))
                    if retryable and attempt < max_retries - 1:
                        time.sleep(min(base_delay * (2 ** attempt), 60))
                    else:
                        trace.error = f"Gemini API error after {attempt + 1} attempts: {exc}\n{traceback.format_exc()}"
                        trace.stop_reason = "api_error"
                        break
            if response is None:
                if trace.stop_reason is None:
                    trace.stop_reason = "api_error"
                break
            if not response.candidates:
                trace.error = "Gemini returned no candidates"
                trace.stop_reason = "api_error"
                break
            candidate = response.candidates[0]
            if candidate.content is None or candidate.content.parts is None:
                trace.error = "Gemini returned empty content"
                trace.stop_reason = "api_error"
                break

            in_tok, out_tok = _gemini_token_counts(getattr(response, "usage_metadata", None))
            trace.total_input_tokens += in_tok
            trace.total_output_tokens += out_tok

            parts = candidate.content.parts
            fn_calls = [p for p in parts if getattr(p, "function_call", None)]
            text_parts = [p for p in parts if getattr(p, "text", None)]

            turn = TurnRecord(
                role="assistant",
                content=" ".join(p.text for p in text_parts) if text_parts else None,
                latency_s=latency, input_tokens=in_tok, output_tokens=out_tok,
            )
            for p in parts:
                ec = getattr(p, "executable_code", None)
                if ec is not None:
                    turn.tool_calls.append(ToolCallRecord(
                        tool_name="code_execution",
                        arguments={"code": getattr(ec, "code", "") or ""},
                    ))
                cer = getattr(p, "code_execution_result", None)
                if cer is not None and turn.tool_calls:
                    last = turn.tool_calls[-1]
                    if last.tool_name == "code_execution" and last.result is None:
                        last.result = {
                            "outcome": str(getattr(cer, "outcome", "")),
                            "output": getattr(cer, "output", "") or "",
                        }

            if fn_calls:
                contents.append(candidate.content)
                fn_response_parts: list[types.Part] = []
                for fc_part in fn_calls:
                    fc = fc_part.function_call
                    fname = name_map.get(fc.name, fc.name)
                    args = dict(fc.args) if fc.args else {}
                    record = ToolCallRecord(tool_name=fname, arguments=args)
                    try:
                        result = self.tool_executor(fname, args)
                        record.result = result
                    except Exception as exc:
                        record.error = f"{type(exc).__name__}: {exc}"
                        result = {"error": record.error}
                    turn.tool_calls.append(record)
                    if not isinstance(result, dict):
                        result = {"result": result}
                    fn_response_parts.append(types.Part.from_function_response(
                        name=fc.name, response=result,
                    ))
                trace.turns.append(turn)
                contents.append(types.Content(role="user", parts=fn_response_parts))
                continue

            trace.turns.append(turn)
            contents.append(candidate.content)
            if pending_user_messages:
                trace.turn_answers.append(turn.content)
                next_msg = pending_user_messages.pop(0)
                trace.turns.append(TurnRecord(role="user", content=next_msg))
                contents.append(types.Content(
                    role="user", parts=[types.Part.from_text(text=next_msg)]))
                continue
            trace.final_answer = turn.content
            if trace.turn_answers:
                trace.turn_answers.append(turn.content)
            trace.stop_reason = "completed"
            break
        else:
            trace.stop_reason = "max_turns"

        trace.total_time_s = time.time() - t0
        return trace

    def _run_claude_ptc(self, user_message: str, task_id: str,
                         extra_user_messages: list[str] | None = None) -> Trace:
        """Service developer tool calls from Claude's native code_execution sandbox until it resumes."""
        import anthropic

        client = anthropic.Anthropic()
        trace = Trace(task_id=task_id, model=self.model, system_prompt=self.system_prompt)
        t0 = time.time()

        pending_user_messages = list(extra_user_messages) if extra_user_messages else []

        ptc_tools, name_map = _tool_defs_to_claude_ptc(self.tool_defs)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]
        trace.turns.append(TurnRecord(role="user", content=user_message))

        max_retries = 10
        base_delay = 2.0
        container_id: str | None = None

        for _ in range(self.max_turns):
            response = None
            for attempt in range(max_retries):
                try:
                    t1 = time.time()
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "max_tokens": 8000,
                        "messages": messages,
                    }
                    if self.temperature is not None:
                        kwargs["temperature"] = self.temperature
                    if self.system_prompt:
                        kwargs["system"] = self.system_prompt
                    if ptc_tools:
                        kwargs["tools"] = ptc_tools
                    if container_id:
                        kwargs["container"] = container_id
                    response = client.messages.create(**kwargs)
                    latency = time.time() - t1
                    if hasattr(response, "container") and response.container:
                        container_id = response.container.id
                    break
                except Exception as exc:
                    err_str = str(exc)
                    retryable = any(k in err_str for k in
                                    ("529", "overloaded", "rate_limit", "timeout", "503"))
                    if retryable and attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), 120)
                        print(f"    Claude PTC retry {attempt + 1}/{max_retries} after {delay:.0f}s: {err_str[:120]}")
                        time.sleep(delay)
                    else:
                        trace.error = f"Claude PTC API error after {attempt + 1} attempts: {exc}"
                        trace.stop_reason = "api_error"
                        break

            if response is None:
                if trace.stop_reason is None:
                    trace.stop_reason = "api_error"
                break

            in_tok = response.usage.input_tokens if response.usage else 0
            out_tok = response.usage.output_tokens if response.usage else 0

            text_parts = []
            tool_use_blocks = []       # tool calls we need to execute
            server_tool_blocks = []    # server-side code execution (for tracing)
            code_exec_results = []     # code_execution_result blocks (for tracing)

            for block in response.content:
                if hasattr(block, "text") and block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)
                elif block.type == "server_tool_use":
                    server_tool_blocks.append(block)
                elif block.type == "code_execution_result":
                    code_exec_results.append(block)

            text_content = "\n".join(text_parts) if text_parts else None

            turn_content_parts = []
            if text_content:
                turn_content_parts.append(text_content)
            for stb in server_tool_blocks:
                code = ""
                if hasattr(stb, "input") and isinstance(stb.input, dict):
                    code = stb.input.get("code", "")
                if code:
                    turn_content_parts.append(f"[PTC Code]\n{code}")
            for cer in code_exec_results:
                if hasattr(cer, "content"):
                    out_text = ""
                    if isinstance(cer.content, list):
                        for part in cer.content:
                            if hasattr(part, "text"):
                                out_text += part.text
                    elif isinstance(cer.content, str):
                        out_text = cer.content
                    if out_text:
                        turn_content_parts.append(f"[PTC Output]\n{out_text}")

            turn_text = "\n".join(turn_content_parts) if turn_content_parts else None

            if tool_use_blocks:
                # Service sandbox tool calls inside one outer turn so they do not consume max_turns.
                turn = TurnRecord(
                    role="assistant", content=turn_text, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )

                max_tool_roundtrips = 200  # safety cap for inner loop
                code_exec_done = False
                for _rt in range(max_tool_roundtrips):
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results: list[dict[str, Any]] = []
                    for tb in tool_use_blocks:
                        sanitized_name = tb.name
                        original_name = name_map.get(sanitized_name, sanitized_name)
                        args = dict(tb.input) if tb.input else {}

                        record = ToolCallRecord(tool_name=original_name, arguments=args)
                        try:
                            result = self.tool_executor(original_name, args)
                            record.result = result
                        except Exception as exc:
                            record.error = f"{type(exc).__name__}: {exc}"
                            result = {"error": record.error}

                        turn.tool_calls.append(record)
                        result_str = json.dumps(result) if not isinstance(result, str) else result
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "content": result_str,
                        })

                    messages.append({"role": "user", "content": tool_results})

                    response = None
                    for attempt in range(max_retries):
                        try:
                            t1 = time.time()
                            kwargs2: dict[str, Any] = {
                                "model": self.model,
                                "max_tokens": 8000,
                                "messages": messages,
                            }
                            if self.temperature is not None:
                                kwargs2["temperature"] = self.temperature
                            if self.system_prompt:
                                kwargs2["system"] = self.system_prompt
                            if ptc_tools:
                                kwargs2["tools"] = ptc_tools
                            if container_id:
                                kwargs2["container"] = container_id
                            response = client.messages.create(**kwargs2)
                            latency2 = time.time() - t1
                            if hasattr(response, "container") and response.container:
                                container_id = response.container.id
                            break
                        except Exception as exc:
                            err_str = str(exc)
                            retryable = any(k in err_str for k in
                                            ("529", "overloaded", "rate_limit", "timeout", "503"))
                            if retryable and attempt < max_retries - 1:
                                delay = min(base_delay * (2 ** attempt), 120)
                                print(f"    Claude PTC retry {attempt + 1}/{max_retries} after {delay:.0f}s: {err_str[:120]}")
                                time.sleep(delay)
                            else:
                                trace.error = f"Claude PTC API error after {attempt + 1} attempts: {exc}"
                                trace.stop_reason = "api_error"
                                break

                    if response is None:
                        if trace.stop_reason is None:
                            trace.stop_reason = "api_error"
                        code_exec_done = True
                        break

                    rt_in = response.usage.input_tokens if response.usage else 0
                    rt_out = response.usage.output_tokens if response.usage else 0
                    in_tok += rt_in
                    out_tok += rt_out
                    turn.latency_s = (turn.latency_s or 0) + latency2

                    tool_use_blocks = []
                    for block in response.content:
                        if hasattr(block, "text") and block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_use_blocks.append(block)
                        elif block.type == "server_tool_use":
                            server_tool_blocks.append(block)
                            code = ""
                            if hasattr(block, "input") and isinstance(block.input, dict):
                                code = block.input.get("code", "")
                            if code:
                                turn_content_parts.append(f"[PTC Code]\n{code}")
                        elif block.type == "code_execution_result":
                            if hasattr(block, "content"):
                                out_text = ""
                                if isinstance(block.content, list):
                                    for part in block.content:
                                        if hasattr(part, "text"):
                                            out_text += part.text
                                elif isinstance(block.content, str):
                                    out_text = block.content
                                if out_text:
                                    turn_content_parts.append(f"[PTC Output]\n{out_text}")

                    if not tool_use_blocks:
                        code_exec_done = True
                        turn.content = "\n".join(turn_content_parts) if turn_content_parts else turn.content
                        text_content = "\n".join(text_parts) if text_parts else None

                        if response.stop_reason == "end_turn":
                            turn.input_tokens = in_tok
                            turn.output_tokens = out_tok
                            trace.turns.append(turn)
                            answer_text = text_content or turn.content
                            trace.total_input_tokens += in_tok
                            trace.total_output_tokens += out_tok

                            if pending_user_messages:
                                trace.turn_answers.append(answer_text)
                                messages.append({"role": "assistant", "content": response.content})
                                next_msg = pending_user_messages.pop(0)
                                messages.append({"role": "user", "content": next_msg})
                                trace.turns.append(TurnRecord(role="user", content=next_msg))
                                # Keep this True: the turn is already recorded; False would double-count it below.
                            else:
                                trace.final_answer = answer_text
                                trace.stop_reason = "completed"
                        else:
                            turn.input_tokens = in_tok
                            turn.output_tokens = out_tok
                            trace.turns.append(turn)
                            trace.total_input_tokens += in_tok
                            trace.total_output_tokens += out_tok
                            messages.append({"role": "assistant", "content": response.content})
                        break


                if code_exec_done and trace.stop_reason == "completed":
                    break
                if trace.error:
                    break
                if not code_exec_done:
                    turn.input_tokens = in_tok
                    turn.output_tokens = out_tok
                    trace.turns.append(turn)
                    trace.total_input_tokens += in_tok
                    trace.total_output_tokens += out_tok

                if self._over_token_budget(trace):
                    trace.stop_reason = "token_budget"
                    break

            elif response.stop_reason == "end_turn":
                turn = TurnRecord(
                    role="assistant", content=turn_text, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok

                if pending_user_messages:
                    trace.turn_answers.append(turn_text)
                    messages.append({"role": "assistant", "content": response.content})
                    next_msg = pending_user_messages.pop(0)
                    messages.append({"role": "user", "content": next_msg})
                    trace.turns.append(TurnRecord(role="user", content=next_msg))
                    continue

                trace.final_answer = turn_text
                trace.stop_reason = "completed"
                break

            else:
                # Server-side execution can continue without external tool calls.
                turn = TurnRecord(
                    role="assistant", content=turn_text, latency_s=latency,
                    input_tokens=in_tok, output_tokens=out_tok,
                )
                trace.turns.append(turn)
                trace.total_input_tokens += in_tok
                trace.total_output_tokens += out_tok
                trace.stop_reason = "completed"
                messages.append({"role": "assistant", "content": response.content})
                break

        if trace.stop_reason is None:
            trace.stop_reason = "max_turns"
        trace.total_time_s = time.time() - t0
        return trace
