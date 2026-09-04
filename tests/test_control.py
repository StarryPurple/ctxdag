"""Tree-structured tests for control actions and tool transport."""

import json
import unittest

from contextdag import (
    ControlError,
    DependencyError,
    RequireContext,
    ReturnAnswer,
    Session,
    ToolRequest,
    ToolTransport,
)


def tool_message(node_ids, call_id="call-1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "require_context",
                    "arguments": json.dumps({"node_ids": node_ids}),
                },
            }
        ],
    }


class ControlLeafTest(unittest.TestCase):
    def test_require_context_rejects_invalid_or_duplicate_ids(self):
        with self.assertRaises(ControlError):
            RequireContext(())
        with self.assertRaises(ControlError):
            RequireContext(("not-an-id",))
        with self.assertRaises(ControlError):
            RequireContext(("a" * 16, "a" * 16))

    def test_tool_call_requires_transport_id_and_string_node_ids(self):
        no_id = tool_message(["a" * 16])
        del no_id["tool_calls"][0]["id"]
        with self.assertRaises(ControlError):
            ToolTransport.parse(no_id)
        with self.assertRaises(ControlError):
            ToolTransport.parse(tool_message([123]))

    def test_tool_schema_can_be_grounded_to_current_catalog(self):
        tools = ToolTransport.tools(("a" * 16, "b" * 16))
        items = tools[0]["function"]["parameters"]["properties"]["node_ids"][
            "items"
        ]
        self.assertEqual(items["enum"], ["a" * 16, "b" * 16])
        self.assertEqual(
            ToolTransport.tools(())[0]["function"]["name"],
            "return_answer",
        )

    def test_return_answer_tool_is_a_semantic_answer(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "answer-1",
                "type": "function",
                "function": {
                    "name": "return_answer",
                    "arguments": '{"text":"done"}',
                },
            }],
        }
        self.assertEqual(ToolTransport.parse(message), ReturnAnswer("done"))

    def test_plain_assistant_message_is_an_answer(self):
        action = ToolTransport.parse({"role": "assistant", "content": "done"})
        self.assertEqual(action, ReturnAnswer("done"))

    def test_tool_schema_requires_a_unique_nonempty_id_array(self):
        parameters = ToolTransport.tools()[0]["function"]["parameters"]
        node_ids = parameters["properties"]["node_ids"]
        self.assertEqual(parameters["required"], ["node_ids"])
        self.assertEqual(node_ids["minItems"], 1)
        self.assertNotIn("uniqueItems", node_ids)
        with self.assertRaises(ControlError):
            RequireContext(("a" * 16, "a" * 16))


class ToolTransportFeatureTest(unittest.TestCase):
    def test_valid_tool_call_round_trips_to_tool_result(self):
        request = ToolTransport.parse(tool_message(["a" * 16, "b" * 16]))

        self.assertEqual(
            request,
            ToolRequest("call-1", RequireContext(("a" * 16, "b" * 16))),
        )
        result = ToolTransport.result(request.call_id, "rendered nodes")
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call-1")
        self.assertEqual(result["content"], "rendered nodes")

    def test_malformed_or_unknown_tool_call_is_rejected(self):
        malformed = tool_message(["a" * 16])
        malformed["tool_calls"][0]["function"]["arguments"] = "{"
        with self.assertRaises(ControlError):
            ToolTransport.parse(malformed)

        unknown = tool_message(["a" * 16])
        unknown["tool_calls"][0]["function"]["name"] = "delete_context"
        with self.assertRaises(ControlError):
            ToolTransport.parse(unknown)


class ControlSessionScenarioTest(unittest.TestCase):
    def test_tool_request_loads_multiple_nodes_in_one_expand(self):
        session = Session()
        visible = session.register("visible")
        first = session.register("first")
        second = session.register("second")
        session.expand(refs=[visible.id], candidates=[first.id, second.id])
        request = ToolTransport.parse(tool_message([first.id, second.id]))

        context = session.require_many(request.action.node_ids)

        self.assertIn(first.id, context.order)
        self.assertIn(second.id, context.order)
        self.assertEqual(session.page_faults, 2)
        self.assertEqual(session.expands, 2)

    def test_multi_request_is_atomic_when_any_node_is_unauthorized(self):
        session = Session()
        visible = session.register("visible")
        authorized = session.register("authorized")
        hidden = session.register("hidden")
        original = session.expand(refs=[visible.id], candidates=[authorized.id])

        with self.assertRaises(DependencyError):
            session.require_many([authorized.id, hidden.id])

        self.assertIs(session.current_context, original)
        self.assertNotIn(authorized.id, session.current_node_ids)
        self.assertEqual(session.page_faults, 0)
        self.assertEqual(session.expands, 1)


if __name__ == "__main__":
    unittest.main()
