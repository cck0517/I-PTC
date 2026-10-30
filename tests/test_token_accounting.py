"""Token and call accounting tests. No network access or API keys needed.

Run from the repository root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from agent_eval import agent as agent_mod
from agent_eval.agent import Agent, ToolDef, _tool_defs_to_claude_ptc
from agent_eval.trajectory_tokens import _tok, compute_trajectory_tokens

_TOOL = ToolDef(
    name="yelp_api.search_businesses",
    description="Search businesses.",
    parameters={"type": "object", "properties": {}},
)


def _response(blocks, stop_reason, input_tokens, output_tokens):
    return NS(
        content=blocks,
        stop_reason=stop_reason,
        usage=NS(input_tokens=input_tokens, output_tokens=output_tokens),
        container=None,
    )


class ClaudeNativeMultiTurnTest(unittest.TestCase):
    """A native code-execution turn that ends while a follow-up user message is
    waiting used to be recorded twice: its tokens were added again and an extra
    assistant turn was appended."""

    def _run(self, responses):
        fake_client = NS(messages=NS(create=mock.Mock(side_effect=responses)))
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = lambda *a, **k: fake_client
        agent = Agent(
            backend="claude_ptc", model="test-model", tool_defs=[_TOOL],
            tool_executor=lambda name, args: {"ok": True},
        )
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            trace = agent.run("first question", "t", extra_user_messages=["follow-up question"])
        return trace, fake_client.messages.create.call_count

    def test_follow_up_after_code_execution_is_counted_once(self):
        _, name_map = _tool_defs_to_claude_ptc([_TOOL])
        tool_name = next(iter(name_map))
        responses = [
            # Turn 1: code calls an API tool, then finishes.
            _response([NS(type="tool_use", name=tool_name, id="tu_1", input={})], "tool_use", 20, 5),
            _response([NS(type="text", text="answer 1")], "end_turn", 10, 5),
            # Turn 2 (after the follow-up question): plain answer.
            _response([NS(type="text", text="answer 2")], "end_turn", 30, 5),
        ]
        trace, calls = self._run(responses)

        self.assertEqual(calls, 3)
        self.assertEqual(trace.stop_reason, "completed")
        self.assertEqual(trace.total_input_tokens, 60)
        self.assertEqual(trace.total_output_tokens, 15)
        assistant_turns = [t for t in trace.turns if t.role == "assistant"]
        self.assertEqual(len(assistant_turns), 2)
        self.assertEqual(sum(t.input_tokens for t in assistant_turns), 60)
        self.assertEqual(trace.turn_answers, ["answer 1"])
        self.assertEqual(trace.final_answer, "answer 2")

    def test_single_turn_is_unchanged(self):
        _, name_map = _tool_defs_to_claude_ptc([_TOOL])
        tool_name = next(iter(name_map))
        agent = Agent(backend="claude_ptc", model="test-model", tool_defs=[_TOOL],
                      tool_executor=lambda name, args: {"ok": True})
        responses = [
            _response([NS(type="tool_use", name=tool_name, id="tu_1", input={})], "tool_use", 20, 5),
            _response([NS(type="text", text="answer")], "end_turn", 10, 5),
        ]
        fake_client = NS(messages=NS(create=mock.Mock(side_effect=responses)))
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = lambda *a, **k: fake_client
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            trace = agent.run("question", "t")
        self.assertEqual((trace.total_input_tokens, trace.total_output_tokens), (30, 10))
        self.assertEqual(len([t for t in trace.turns if t.role == "assistant"]), 1)
        self.assertEqual(trace.final_answer, "answer")

    def test_round_trip_cap_still_records_the_turn_once(self):
        _, name_map = _tool_defs_to_claude_ptc([_TOOL])
        tool_name = next(iter(name_map))
        tool_call = lambda i: _response(  # noqa: E731
            [NS(type="tool_use", name=tool_name, id=f"tu_{i}", input={})], "tool_use", 1, 1)
        # One outer call plus 200 inner round trips, all still asking for tools,
        # then a final plain answer on the next outer iteration.
        responses = [tool_call(i) for i in range(201)] + [_response([NS(type="text", text="done")], "end_turn", 1, 1)]
        agent = Agent(backend="claude_ptc", model="test-model", tool_defs=[_TOOL],
                      tool_executor=lambda name, args: {"ok": True}, max_turns=2)
        fake_client = NS(messages=NS(create=mock.Mock(side_effect=responses)))
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = lambda *a, **k: fake_client
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            trace = agent.run("question", "t")
        self.assertEqual(trace.total_input_tokens, 202)
        self.assertEqual(len([t for t in trace.turns if t.role == "assistant"]), 2)


class GeminiTokenCountTest(unittest.TestCase):
    """Gemini reports thinking tokens separately (thoughts_token_count); they are
    billed as output and must be counted."""

    def test_thinking_tokens_count_as_output(self):
        usage = NS(prompt_token_count=100, candidates_token_count=5, thoughts_token_count=1000)
        self.assertEqual(agent_mod._gemini_token_counts(usage), (100, 1005))

    def test_missing_fields_count_as_zero(self):
        self.assertEqual(agent_mod._gemini_token_counts(NS(prompt_token_count=7, candidates_token_count=3)), (7, 3))
        self.assertEqual(agent_mod._gemini_token_counts(NS(prompt_token_count=None, candidates_token_count=None,
                                                 thoughts_token_count=None)), (0, 0))
        self.assertEqual(agent_mod._gemini_token_counts(None), (0, 0))


class TrajectoryUserMessagesTest(unittest.TestCase):
    """Every user message enters the trajectory once, including follow-ups."""

    def test_follow_up_user_messages_are_counted(self):
        first = "Find the best Chinese restaurant within five miles of me."
        follow_up = "Now only consider places that are open after 10 p.m. on weekends."
        trace = {
            "model": "gpt-5-mini",
            "system_prompt": "",
            "run_config": {"backend": "openai", "mode": "tool_only"},
            "turns": [
                {"role": "user", "content": first},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": follow_up},
                {"role": "assistant", "content": "answer 2"},
            ],
        }
        out = compute_trajectory_tokens(trace)
        expected = _tok(first, "gpt-5-mini") + _tok(follow_up, "gpt-5-mini")
        self.assertEqual(out["tj_user_question_tokens"], expected)

    def test_single_turn_unchanged(self):
        q = "Find the best Chinese restaurant within five miles of me."
        trace = {"model": "gpt-5-mini", "system_prompt": "", "run_config": {},
                 "turns": [{"role": "user", "content": q}, {"role": "assistant", "content": "a"}]}
        self.assertEqual(compute_trajectory_tokens(trace)["tj_user_question_tokens"], _tok(q, "gpt-5-mini"))


if __name__ == "__main__":
    unittest.main()
