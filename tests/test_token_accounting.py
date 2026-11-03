"""Offline regression tests for token and call accounting."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from iptc import agent as agent_mod
from iptc.agent import Agent, ToolDef, _tool_defs_to_claude_ptc
from ptc_bench.trajectory_tokens import compute_trajectory_tokens

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
    def _run(self, responses, *, extra_user_messages=None, max_turns=30):
        fake_client = NS(messages=NS(create=mock.Mock(side_effect=responses)))
        fake_anthropic = NS(Anthropic=lambda **kwargs: fake_client)
        agent = Agent(
            backend="claude_ptc", model="test-model", tool_defs=[_TOOL],
            tool_executor=lambda name, args: {"ok": True}, max_turns=max_turns,
        )
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            trace = agent.run("question", "t", extra_user_messages=extra_user_messages)
        return trace, fake_client.messages.create.call_count

    def test_code_execution_tokens_are_counted_once(self):
        _, name_map = _tool_defs_to_claude_ptc([_TOOL])
        tool_name = next(iter(name_map))
        for follow_up in (False, True):
            with self.subTest(follow_up=follow_up):
                responses = [
                    _response([NS(type="tool_use", name=tool_name, id="tu_1", input={})], "tool_use", 20, 5),
                    _response([NS(type="text", text="answer 1")], "end_turn", 10, 5),
                ]
                if follow_up:
                    responses.append(_response([NS(type="text", text="answer 2")], "end_turn", 30, 5))
                trace, calls = self._run(
                    responses, extra_user_messages=["follow-up question"] if follow_up else None,
                )

                self.assertEqual(calls, len(responses))
                self.assertEqual(trace.stop_reason, "completed")
                self.assertEqual(
                    (trace.total_input_tokens, trace.total_output_tokens),
                    (60, 15) if follow_up else (30, 10),
                )
                assistant_turns = [t for t in trace.turns if t.role == "assistant"]
                self.assertEqual(
                    [(t.input_tokens, t.output_tokens) for t in assistant_turns],
                    [(30, 10), (30, 5)] if follow_up else [(30, 10)],
                )
                self.assertEqual(trace.turn_answers, ["answer 1"] if follow_up else [])
                self.assertEqual(trace.final_answer, "answer 2" if follow_up else "answer 1")

    def test_round_trip_cap_still_records_the_turn_once(self):
        _, name_map = _tool_defs_to_claude_ptc([_TOOL])
        tool_name = next(iter(name_map))
        # One outer call plus 200 inner round trips reaches the cap.
        responses = [
            _response([NS(type="tool_use", name=tool_name, id=f"tu_{i}", input={})], "tool_use", 1, 1)
            for i in range(201)
        ] + [_response([NS(type="text", text="done")], "end_turn", 1, 1)]
        trace, calls = self._run(responses, max_turns=2)
        self.assertEqual(calls, 202)
        self.assertEqual((trace.total_input_tokens, trace.total_output_tokens), (202, 202))
        self.assertEqual(len([t for t in trace.turns if t.role == "assistant"]), 2)
        self.assertEqual(trace.final_answer, "done")


class GeminiTokenCountTest(unittest.TestCase):
    def test_thinking_and_missing_token_counts(self):
        cases = [
            (NS(prompt_token_count=100, candidates_token_count=5, thoughts_token_count=1000), (100, 1005)),
            (NS(prompt_token_count=7, candidates_token_count=3), (7, 3)),
            (NS(prompt_token_count=None, candidates_token_count=None, thoughts_token_count=None), (0, 0)),
            (None, (0, 0)),
        ]
        for usage, expected in cases:
            with self.subTest(usage=usage):
                self.assertEqual(agent_mod._gemini_token_counts(usage), expected)


class TrajectoryUserMessagesTest(unittest.TestCase):
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
        with mock.patch("ptc_bench.trajectory_tokens._tok", side_effect=lambda text, model: len(text.split())):
            out = compute_trajectory_tokens(trace)
        self.assertEqual(out["tj_user_question_tokens"], 22)


if __name__ == "__main__":
    unittest.main()
