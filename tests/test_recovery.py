"""
Offline tests for core/recovery.py (the policy on its own).

Run from the project root:

    python -m unittest tests.test_recovery -v
"""

import unittest

from core.planner import Planner
from core.recovery import RecoveryAction, RecoveryPolicy
from core.registry import AgentRegistry, AgentSpec


class Dummy:

    def __init__(self, **params):
        self.params = params


def spec(name, requires=(), produces=("x",), **kwargs):

    return AgentSpec(
        name=name,
        description="d",
        requires=requires,
        produces=produces,
        factory=Dummy,
        **kwargs
    )


def node_for(registry, goal="x", **plan_kwargs):

    return Planner(registry).plan(goal, **plan_kwargs).nodes[-1]


FAILED = {"status": "failed", "error": "boom"}
INVALID = {"status": "invalid", "error": "too few"}


class LadderTests(unittest.TestCase):

    def setUp(self):

        self.registry = AgentRegistry()
        self.registry.register(spec(
            "a",
            escalations=({}, {"wide": True})
        ))

        self.policy = RecoveryPolicy(self.registry)
        self.node = node_for(self.registry)

    def test_walks_the_ladder_in_order(self):

        first = self.policy.decide(self.node, INVALID, [], 3)

        self.assertEqual(first.kind, "retry")
        self.assertEqual(first.params, {})

        second = self.policy.decide(self.node, INVALID, [first], 2)

        self.assertEqual(second.kind, "retry")
        self.assertEqual(second.params, {"wide": True})

    def test_gives_up_when_the_ladder_is_used_and_no_alternative(self):

        taken = [
            RecoveryAction("retry", "r"),
            RecoveryAction("retry", "r", params={"wide": True})
        ]

        action = self.policy.decide(self.node, INVALID, taken, 1)

        self.assertEqual(action.kind, "give_up")

    def test_no_escalations_means_no_retries(self):

        registry = AgentRegistry()
        registry.register(spec("a"))

        action = RecoveryPolicy(registry).decide(
            node_for(registry), FAILED, [], 3
        )

        self.assertEqual(action.kind, "give_up")

    def test_run_budget_stops_recovery(self):

        action = self.policy.decide(self.node, FAILED, [], 0)

        self.assertEqual(action.kind, "give_up")
        self.assertIn("budget", action.reason)

    def test_node_limit_stops_recovery(self):

        policy = RecoveryPolicy(
            self.registry,
            max_actions_per_node=1
        )

        action = policy.decide(
            self.node,
            FAILED,
            [RecoveryAction("retry", "r")],
            5
        )

        self.assertEqual(action.kind, "give_up")
        self.assertIn("limit", action.reason)


class SwapTests(unittest.TestCase):

    def registry(self, alt_requires=()):

        registry = AgentRegistry()
        registry.register(spec("main"))
        registry.register(spec("alt", requires=alt_requires))

        return registry

    def test_swaps_to_an_alternative_when_the_ladder_is_empty(self):

        registry = self.registry()

        node = node_for(registry)

        action = RecoveryPolicy(registry).decide(
            node, FAILED, [], 3
        )

        self.assertEqual(action.kind, "swap_agent")
        self.assertEqual(action.agent, "alt")
        self.assertEqual(action.from_agent, "main")

    def test_does_not_swap_to_an_agent_needing_other_inputs(self):

        registry = self.registry(alt_requires=("other",))

        registry.register(spec("maker", produces=("other",)))

        node = node_for(registry)

        action = RecoveryPolicy(registry).decide(
            node, FAILED, [], 3
        )

        self.assertEqual(action.kind, "give_up")

    def test_never_swaps_back_to_an_agent_already_tried(self):

        registry = self.registry()

        policy = RecoveryPolicy(registry)

        node = node_for(registry)

        swap = policy.decide(node, FAILED, [], 3)

        # After the swap the node runs "alt", whose alternative is
        # "main", which was already tried.
        swapped = type(node)(
            id=node.id,
            agent="alt",
            params={},
            requires=(),
            produces=("x",),
            depends_on=(),
            alternatives=("main",)
        )

        action = policy.decide(swapped, FAILED, [swap], 2)

        self.assertEqual(action.kind, "give_up")


class ActionTests(unittest.TestCase):

    def test_to_dict(self):

        import json

        action = RecoveryAction(
            "swap_agent", "why", agent="b", from_agent="a"
        )

        json.dumps(action.to_dict())

        self.assertEqual(action.to_dict()["agent"], "b")


if __name__ == "__main__":
    unittest.main()
