"""
Offline tests for core/planner.py.

Run from the project root:

    python -m unittest tests.test_planner -v
"""

import os
import unittest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.default_registry import build_default_registry
from core.planner import Plan, PlanError, Planner
from core.registry import AgentRegistry, AgentSpec


class Dummy:

    def __init__(self, **params):
        self.params = params


def make_registry(*specs):

    registry = AgentRegistry()

    for name, requires, produces in specs:
        registry.register(AgentSpec(
            name=name,
            description="d",
            requires=requires,
            produces=produces,
            factory=Dummy,
            default_params={"p": 1}
        ))

    return registry


class DefaultPipelinePlanTests(unittest.TestCase):

    def setUp(self):

        self.planner = Planner(build_default_registry())

    def test_goal_scores_gives_the_four_agents_in_order(self):

        plan = self.planner.plan("scores")

        self.assertEqual(
            [node.id for node in plan.nodes],
            [
                "icp_agent",
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )

        scoring = plan.get("scoring_agent")

        # Scoring needs the ICP and the enriched leads.
        self.assertEqual(
            scoring.depends_on,
            ("icp_agent", "enrichment_agent")
        )

    def test_smaller_goals_need_fewer_agents(self):

        self.assertEqual(
            [n.id for n in self.planner.plan("icp").nodes],
            ["icp_agent"]
        )
        self.assertEqual(
            [n.id for n in self.planner.plan("leads").nodes],
            ["icp_agent", "lead_discovery_agent"]
        )

    def test_available_artifacts_are_not_recomputed(self):

        plan = self.planner.plan("scores", available={"icp"})

        self.assertEqual(
            [node.id for node in plan.nodes],
            [
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )

        # No dependency on a node that is not in the plan.
        self.assertEqual(
            plan.get("lead_discovery_agent").depends_on,
            ()
        )

    def test_goal_already_available_gives_an_empty_plan(self):

        plan = self.planner.plan("scores", available={"scores"})

        self.assertEqual(plan.nodes, [])

    def test_param_overrides_merge_over_defaults(self):

        plan = self.planner.plan(
            "leads",
            params={"lead_discovery_agent": {"max_industries": 3}}
        )

        params = plan.get("lead_discovery_agent").params

        self.assertEqual(params["max_industries"], 3)
        self.assertEqual(params["results_per_query"], 3)

    def test_the_chain_has_one_node_per_layer(self):

        layers = self.planner.plan("scores").layers()

        self.assertEqual(
            [[n.id for n in layer] for layer in layers],
            [
                ["icp_agent"],
                ["lead_discovery_agent"],
                ["enrichment_agent"],
                ["scoring_agent"]
            ]
        )


class GraphShapeTests(unittest.TestCase):

    def diamond(self):

        return Planner(make_registry(
            ("a", (), ("x",)),
            ("b", ("x",), ("y",)),
            ("c", ("x",), ("z",)),
            ("d", ("y", "z"), ("goal",))
        ))

    def test_independent_branches_share_a_layer(self):

        layers = self.diamond().plan("goal").layers()

        self.assertEqual(
            [sorted(n.id for n in layer) for layer in layers],
            [["a"], ["b", "c"], ["d"]]
        )

    def test_shared_dependency_appears_once(self):

        plan = self.diamond().plan("goal")

        self.assertEqual(
            sorted(n.id for n in plan.nodes),
            ["a", "b", "c", "d"]
        )

    def test_ready_follows_dependencies(self):

        plan = self.diamond().plan("goal")

        self.assertEqual([n.id for n in plan.ready(set())], ["a"])
        self.assertEqual(
            sorted(n.id for n in plan.ready({"a"})),
            ["b", "c"]
        )
        self.assertEqual(
            [n.id for n in plan.ready({"a", "b"})],
            ["c"]
        )
        self.assertEqual(
            [n.id for n in plan.ready({"a", "b", "c"})],
            ["d"]
        )
        self.assertEqual(
            plan.ready({"a", "b", "c", "d"}),
            []
        )

    def test_agent_producing_two_artifacts_is_one_node(self):

        planner = Planner(make_registry(
            ("a", (), ("x", "y")),
            ("b", ("x", "y"), ("goal",))
        ))

        plan = planner.plan("goal")

        self.assertEqual([n.id for n in plan.nodes], ["a", "b"])
        self.assertEqual(plan.get("b").depends_on, ("a",))


class AlternativesAndErrorsTests(unittest.TestCase):

    def alternatives(self):

        return Planner(make_registry(
            ("fast", (), ("leads",)),
            ("slow", (), ("leads",)),
            ("use", ("leads",), ("goal",))
        ))

    def test_first_producer_is_default_and_others_are_listed(self):

        node = self.alternatives().plan("goal").get("fast")

        self.assertEqual(node.alternatives, ("slow",))

    def test_prefer_selects_a_specific_producer(self):

        plan = self.alternatives().plan(
            "goal",
            prefer={"leads": "slow"}
        )

        self.assertEqual(
            [n.id for n in plan.nodes],
            ["slow", "use"]
        )
        self.assertEqual(plan.get("slow").alternatives, ("fast",))

    def test_prefer_with_wrong_agent_is_an_error(self):

        with self.assertRaises(PlanError):
            self.alternatives().plan(
                "goal",
                prefer={"leads": "use"}
            )

    def test_unreachable_goal_is_an_error(self):

        planner = Planner(make_registry(
            ("a", ("ghost",), ("goal",))
        ))

        with self.assertRaises(PlanError) as context:
            planner.plan("goal")

        self.assertIn("ghost", str(context.exception))

    def test_unknown_goal_is_an_error(self):

        with self.assertRaises(PlanError):
            Planner(build_default_registry()).plan("nonsense")

    def test_circular_dependency_is_an_error(self):

        planner = Planner(make_registry(
            ("a", ("y",), ("x",)),
            ("b", ("x",), ("y",))
        ))

        with self.assertRaises(PlanError) as context:
            planner.plan("x")

        self.assertIn("Circular", str(context.exception))


class PlanOutputTests(unittest.TestCase):

    def test_to_dict_is_json_friendly(self):

        import json

        plan = Planner(build_default_registry()).plan("scores")

        data = plan.to_dict()

        json.dumps(data)

        self.assertEqual(data["goal"], "scores")
        self.assertEqual(len(data["nodes"]), 4)

    def test_plan_object_defaults(self):

        self.assertEqual(Plan(goal="g").nodes, [])


if __name__ == "__main__":
    unittest.main()
