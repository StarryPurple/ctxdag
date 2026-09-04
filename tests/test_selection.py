"""Tree-structured tests for initial selection and require recovery."""

import sys
import unittest

from contextdag import (
    KeywordSelector,
    RecentSelector,
    SelectionCandidate,
    Session,
)

sys.path.insert(0, "bench")

from eval_recovery import aggregate, evaluate_record, tool_instruction  # noqa: E402
from recovery import run_message_recovery, run_recovery, run_tool_recovery  # noqa: E402


def candidate(node_id, summary, tokens, position):
    return SelectionCandidate(node_id, summary, tokens, position)


class SelectionLeafTest(unittest.TestCase):
    def test_keyword_selector_prefers_matching_summary_within_budget(self):
        catalog = [
            candidate("a" * 16, "shipping address changes", 5, 0),
            candidate("b" * 16, "refund a delivered order", 5, 1),
        ]

        selected = KeywordSelector().select("How do I get a refund?", catalog, 5)

        self.assertEqual(selected, ["b" * 16])

    def test_selector_never_exceeds_budget(self):
        catalog = [
            candidate("a" * 16, "refund one", 6, 0),
            candidate("b" * 16, "refund two", 5, 1),
        ]

        selected = KeywordSelector().select("refund", catalog, 10)

        self.assertEqual(len(selected), 1)

    def test_fulltext_index_can_differ_from_display_summary(self):
        catalog = [
            SelectionCandidate("a" * 16, "general policy", 5, 0, "refund rules"),
            SelectionCandidate("b" * 16, "refund overview", 5, 1, "shipping times"),
        ]

        selected = KeywordSelector().select("refund", catalog, 5)

        self.assertEqual(selected, ["a" * 16])

    def test_recent_selector_returns_registration_order(self):
        catalog = [
            candidate("a" * 16, "old", 4, 0),
            candidate("b" * 16, "middle", 4, 1),
            candidate("c" * 16, "new", 4, 2),
        ]

        selected = RecentSelector().select("", catalog, 8)

        self.assertEqual(selected, ["b" * 16, "c" * 16])


class RecoveryFeatureTest(unittest.TestCase):
    def test_strict_tool_policy_requires_explicit_evidence(self):
        session = Session()
        visible = session.register("visible evidence")
        candidate = session.register("candidate evidence")
        session.expand(refs=[visible.id], candidates=[candidate.id])

        instruction = tool_instruction(session, "strict")

        self.assertIn("must make one control tool call", instruction)
        self.assertIn("explicitly contain every fact", instruction)

    def test_tool_return_answer_is_scored_as_the_initial_output(self):
        session = Session()
        visible = session.register("VISIBLE_CONTENT")
        session.expand(refs=[visible.id])

        def model(messages, tools, tool_choice=None):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-answer",
                    "type": "function",
                    "function": {
                        "name": "return_answer",
                        "arguments": '{"text": "approved"}',
                    },
                }],
            }

        result = run_tool_recovery(session, model, list, "\nanswer", 2)

        self.assertEqual(result.initial_output, "approved")
        self.assertEqual(result.final_output, "approved")
        self.assertEqual(result.require_rounds, 0)

    def test_invalid_tool_call_keeps_parse_reason(self):
        session = Session()
        visible = session.register("VISIBLE_CONTENT")
        session.expand(refs=[visible.id])

        def model(messages, tools, tool_choice=None):
            return {"role": "assistant", "content": None, "tool_calls": [{}, {}]}

        result = run_tool_recovery(session, model, list, "\nanswer", 2)

        self.assertEqual(result.failure_type, "invalid_tool_call")
        self.assertEqual(
            result.calls[0]["control_error"],
            "exactly one control tool call is allowed",
        )

    def test_tool_recovery_preserves_messages_and_returns_only_delta(self):
        session = Session()
        visible = session.register("UNIQUE_VISIBLE_CONTENT")
        missing = session.register("UNIQUE_MISSING_CONTENT")
        session.expand(refs=[visible.id], candidates=[missing.id])

        class Model:
            def __init__(self):
                self.calls = []

            def __call__(self, messages, tools, tool_choice=None):
                if tool_choice != "required":
                    raise AssertionError("tool routing decision must be required")
                self.calls.append(
                    ([dict(message) for message in messages], list(tools))
                )
                if len(self.calls) == 1:
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "require_context",
                                "arguments": (
                                    '{"node_ids": ["' + missing.id + '"]}'
                                ),
                            },
                        }],
                    }
                return {"role": "assistant", "content": "approved"}

        model = Model()
        result = run_tool_recovery(session, model, list, "\nanswer", 1)

        self.assertEqual(result.final_output, "approved")
        self.assertEqual(result.required_node_ids, (missing.id,))
        self.assertEqual(model.calls[1][0][0], model.calls[0][0][0])
        self.assertEqual(model.calls[1][1], model.calls[0][1])
        self.assertEqual(
            [message["role"] for message in model.calls[1][0]],
            ["user", "assistant", "tool"],
        )
        tool_content = model.calls[1][0][-1]["content"]
        self.assertIn("UNIQUE_MISSING_CONTENT", tool_content)
        self.assertNotIn("UNIQUE_VISIBLE_CONTENT", tool_content)
        self.assertIn("final retrieval round", tool_content)

    def test_missing_node_is_loaded_and_accounted_once(self):
        session = Session()
        visible = session.register("visible")
        missing = session.register("refund policy")
        session.expand(refs=[visible.id], candidates=[missing.id])
        outputs = iter([f"<require={missing.id}>", "approved"])
        catalogs = []

        def suffix():
            catalogs.append(session.current_context.catalog)
            return "\nanswer or require"

        result = run_recovery(
            session,
            lambda prompt: next(outputs),
            list,
            suffix,
            max_require_rounds=2,
        )

        self.assertEqual(result.final_output, "approved")
        self.assertEqual(result.required_node_ids, (missing.id,))
        self.assertEqual(result.require_rounds, 1)
        self.assertGreater(result.recovery_prompt_tokens, 0)
        self.assertEqual(
            result.total_prompt_tokens,
            result.initial_prompt_tokens + result.recovery_prompt_tokens,
        )
        self.assertEqual(session.page_faults, 1)
        self.assertEqual(catalogs, [(missing.id,), ()])

    def test_multiple_nodes_load_in_one_recovery_round(self):
        session = Session()
        visible = session.register("visible")
        first = session.register("first evidence")
        second = session.register("second evidence")
        session.expand(refs=[visible.id], candidates=[first.id, second.id])
        outputs = iter([
            f"<require={first.id}><require={second.id}>",
            "complete",
        ])

        result = run_recovery(
            session, lambda prompt: next(outputs), list, "", 2
        )

        self.assertEqual(result.require_rounds, 1)
        self.assertEqual(set(result.required_node_ids), {first.id, second.id})
        self.assertGreater(result.incremental_node_tokens, 0)
        self.assertEqual(session.page_faults, 2)

    def test_message_recovery_preserves_prefix_and_sends_only_delta(self):
        session = Session()
        visible = session.register("UNIQUE_VISIBLE_CONTENT")
        missing = session.register("UNIQUE_MISSING_CONTENT")
        session.expand(refs=[visible.id], candidates=[missing.id])

        class Model:
            def __init__(self):
                self.calls = []

            def __call__(self, messages):
                self.calls.append([dict(message) for message in messages])
                if len(self.calls) == 1:
                    return f"<require={missing.id}>"
                return "approved"

        model = Model()
        result = run_message_recovery(
            session, model, list, "\nanswer", "\ncontinue", 2
        )

        self.assertEqual(result.final_output, "approved")
        self.assertEqual(result.require_rounds, 1)
        self.assertEqual(model.calls[1][0], model.calls[0][0])
        self.assertEqual(
            [message["role"] for message in model.calls[1]],
            ["user", "assistant", "user"],
        )
        delta_message = model.calls[1][-1]["content"]
        self.assertIn("UNIQUE_MISSING_CONTENT", delta_message)
        self.assertNotIn("UNIQUE_VISIBLE_CONTENT", delta_message)
        self.assertEqual(result.calls[1]["message_count"], 3)
        self.assertEqual(session.page_faults, 1)

    def test_repeated_require_stops_without_second_page_fault(self):
        session = Session()
        visible = session.register("visible")
        missing = session.register("refund policy")
        session.expand(refs=[visible.id], candidates=[missing.id])
        outputs = iter(
            [f"<require={missing.id}>", f"<require={missing.id}>"]
        )

        result = run_recovery(
            session,
            lambda prompt: next(outputs),
            list,
            "",
            max_require_rounds=2,
        )

        self.assertEqual(result.failure_type, "repeated_require")
        self.assertEqual(session.page_faults, 1)
        self.assertEqual(session.expands, 2)

    def test_out_of_catalog_require_is_rejected(self):
        session = Session()
        visible = session.register("visible")
        hidden = session.register("hidden")
        session.expand(refs=[visible.id], candidates=())

        result = run_recovery(
            session,
            lambda prompt: f"<require={hidden.id}>",
            list,
            "",
            max_require_rounds=2,
        )

        self.assertEqual(result.failure_type, "rejected_require")
        self.assertEqual(session.requires_rejected, 1)


class SelectionRecoveryScenarioTest(unittest.TestCase):
    def test_evaluator_accepts_native_tool_transport(self):
        record = {
            "_id": "tool-eval",
            "dataset": "triviaqa_e",
            "context": "The answer is yes.",
            "input": "What is the answer?",
            "answers": ["yes"],
        }

        def model(messages, tools):
            self.assertEqual(tools, [])
            return {"role": "assistant", "content": "yes"}

        row = evaluate_record(
            "full",
            record,
            model,
            list,
            lambda value: "".join(value),
            500,
            0.5,
            1,
            transport="tools",
        )

        self.assertEqual(row["transport"], "tools")
        self.assertEqual(row["final_score"], 1.0)
        self.assertEqual(row["calls"][0]["message_count"], 1)

    def test_initial_miss_recovers_through_visible_catalog(self):
        session = Session()
        policy = session.register("Refunds are allowed within thirty days.")
        recent = session.register("Shipping takes five days.")
        for node, summary in [
            (policy, "refund policy"),
            (recent, "shipping estimate"),
        ]:
            session.summaries.set(node.id, summary, source="service")
        catalog = [
            candidate(policy.id, "refund policy", 10, 0),
            candidate(recent.id, "shipping estimate", 10, 1),
        ]
        selected = RecentSelector().select("Can I get a refund?", catalog, 10)
        question = session.register("Can I get a refund?", refs=selected)
        session.expand(
            refs=[question.id],
            candidates=[item.node_id for item in catalog if item.node_id not in selected],
        )

        def model(prompt):
            if "Refunds are allowed" not in prompt:
                return f"<require={policy.id}>"
            return "yes"

        result = run_recovery(session, model, list, "", max_require_rounds=2)

        self.assertEqual(result.final_output, "yes")
        self.assertEqual(result.require_rounds, 1)
        self.assertIn(policy.id, session.current_node_ids)


    def test_call_usage_is_preserved_and_aggregated(self):
        class Model:
            last_usage = {}

            def __call__(self, prompt):
                self.last_usage = {
                    "prompt_tokens": 10,
                    "cached_tokens": 4,
                    "completion_tokens": 6,
                }
                return "yes"

        record = {
            "_id": "usage",
            "dataset": "triviaqa_e",
            "context": "The answer is yes.",
            "input": "What is the answer?",
            "answers": ["yes"],
        }
        row = evaluate_record(
            "full", record, Model(), list, lambda value: "".join(value),
            500, 0.5, 2,
        )

        totals = aggregate([row])
        self.assertEqual(totals["api_prompt_tokens"], 10)
        self.assertEqual(totals["cached_tokens"], 4)
        self.assertEqual(totals["completion_tokens"], 6)
        self.assertGreater(totals["total_latency_seconds"], 0)

    def test_benchmark_observation_separates_initial_and_recovery_costs(self):
        record = {
            "_id": "sample",
            "dataset": "triviaqa_e",
            "context": "Refund answer: yes.\n\nShipping answer: five days.",
            "input": "Is the refund answer yes?",
            "answers": ["yes"],
        }

        row = evaluate_record(
            "keyword-recovery",
            record,
            lambda prompt: "yes",
            list,
            lambda chars: "".join(chars),
            500,
            0.5,
            2,
        )

        self.assertEqual(row["initial_score"], 1.0)
        self.assertEqual(row["final_score"], 1.0)
        self.assertEqual(row["recovery_prompt_tokens"], 0)
        self.assertEqual(row["total_prompt_tokens"], row["initial_prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
