"""Tests for the conversational agent wrapper."""

import unittest

from contextdag import ContextAgent, Session


class ContextAgentTest(unittest.TestCase):
    def test_two_turn_chain_builds_dag(self):
        agent = ContextAgent(model=lambda prompt: "ok")
        agent.say("你好")
        agent.say("继续")
        nodes = agent.session.registry.nodes()
        self.assertEqual(len(nodes), 4)  # user1, agent1, user2, agent2
        user1, agent1, user2, agent2 = nodes
        self.assertEqual(user1.refs, ())
        self.assertEqual(agent1.refs, (user1.id,))
        self.assertEqual(user2.refs, (agent1.id,))
        self.assertEqual(agent2.refs, (user2.id,))

    def test_catalog_ids_visible_in_context(self):
        session = Session()
        policy = session.register("policy text", meta={"summary": "政策"})
        agent = ContextAgent(model=lambda prompt: "ok", session=session, catalog_ids=[policy.id])
        agent.say("hi")
        self.assertIn("<目录 可申请范围>", agent.last_context.text)
        self.assertIn(policy.id, agent.last_context.catalog)

    def test_default_catalog_from_session_recent(self):
        session = Session(catalog_size=2)
        user1 = session.register("hello")
        policy = session.register("policy", meta={"summary": "政策"})
        agent = ContextAgent(model=lambda prompt: "ok", session=session)
        agent.say("hi")
        # recent(2) = [policy, user-message]; the user message is already
        # expanded, so only the policy remains as a require-able candidate.
        self.assertEqual(agent.last_context.catalog, (policy.id,))
        self.assertNotIn(user1.id, agent.last_context.catalog)

    def test_require_in_model_output_counts_faults(self):
        session = Session()
        policy = session.register("policy", meta={"summary": "政策"})

        def model(prompt):
            return f"需要政策。<require={policy.id}> 继续。"

        agent = ContextAgent(model=model, session=session, catalog_ids=[policy.id])
        agent.say("hi")
        self.assertEqual(session.page_faults, 1)
        self.assertEqual(len(session.registry.nodes()), 4)  # policy, user, seg1, seg2

    def test_instruction_is_appended_to_prompt(self):
        prompts: list[str] = []

        def model(prompt):
            prompts.append(prompt)
            return "ok"

        agent = ContextAgent(model=model, instruction="请使用标签。")
        agent.say("hi")
        self.assertTrue(prompts[0].endswith("\n\n请使用标签。"))


if __name__ == "__main__":
    unittest.main()
