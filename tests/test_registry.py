"""
Offline tests for core/registry.py and core/default_registry.py.

Run from the project root:

    python -m unittest tests.test_registry -v
"""

import os
import unittest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.default_registry import build_default_registry
from core.registry import AgentRegistry, AgentSpec


class Dummy:

    def __init__(self, **params):
        self.params = params


def spec(name="a", requires=(), produces=("x",), **kwargs):

    kwargs.setdefault("factory", Dummy)

    return AgentSpec(
        name=name,
        description="d",
        requires=requires,
        produces=produces,
        **kwargs
    )


class RegistryTests(unittest.TestCase):

    def test_register_and_get(self):

        registry = AgentRegistry()
        registry.register(spec("a"))

        self.assertEqual(registry.get("a").name, "a")
        self.assertEqual(registry.names(), ["a"])

    def test_duplicate_name_rejected(self):

        registry = AgentRegistry()
        registry.register(spec("a"))

        with self.assertRaises(ValueError):
            registry.register(spec("a"))

    def test_unknown_agent_raises(self):

        with self.assertRaises(KeyError):
            AgentRegistry().get("nope")

    def test_invalid_specs_rejected(self):

        registry = AgentRegistry()

        with self.assertRaises(ValueError):
            registry.register(spec("a", produces=()))

        with self.assertRaises(ValueError):
            registry.register(spec("a", factory=None))

        with self.assertRaises(ValueError):
            registry.register(
                spec("a", requires=("x",), produces=("x",))
            )

        with self.assertRaises(ValueError):
            registry.register(spec(""))

    def test_producers_of_returns_every_alternative(self):

        registry = AgentRegistry()
        registry.register(spec("a", produces=("leads",)))
        registry.register(spec("b", produces=("leads",)))
        registry.register(spec("c", produces=("scores",)))

        self.assertEqual(
            [s.name for s in registry.producers_of("leads")],
            ["a", "b"]
        )
        self.assertEqual(registry.producers_of("nothing"), [])

    def test_build_merges_params_over_defaults(self):

        registry = AgentRegistry()
        registry.register(spec(
            "a",
            default_params={"n": 1, "m": 2}
        ))

        agent = registry.build("a", n=5)

        self.assertEqual(agent.params, {"n": 5, "m": 2})
        self.assertEqual(registry.build("a").params, {"n": 1, "m": 2})

    def test_check_flags_unproducible_requirements(self):

        registry = AgentRegistry()
        registry.register(spec("a", requires=("ghost",)))

        problems = registry.check()

        self.assertEqual(len(problems), 1)
        self.assertIn("ghost", problems[0])

    def test_external_input_is_allowed(self):

        registry = AgentRegistry()
        registry.register(
            spec("a", requires=("startup_description",))
        )

        self.assertEqual(registry.check(), [])

    def test_a_new_kind_of_agent_needs_no_core_changes(self):
        """The 'universal' claim: an unrelated agent just registers."""

        registry = build_default_registry()

        registry.register(spec(
            "outreach_agent",
            requires=("scores", "icp"),
            produces=("outreach_drafts",)
        ))

        self.assertEqual(registry.check(), [])
        self.assertEqual(
            [s.name for s in registry.producers_of("outreach_drafts")],
            ["outreach_agent"]
        )


class DefaultRegistryTests(unittest.TestCase):

    def test_contains_the_four_agents_and_is_consistent(self):

        registry = build_default_registry()

        self.assertEqual(
            registry.names(),
            [
                "icp_agent",
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )
        self.assertEqual(registry.check(), [])

    def test_specs_match_the_real_agent_names(self):

        registry = build_default_registry()

        for name in registry.names():
            self.assertEqual(registry.build(name).name, name)

    def test_discovery_params_reach_the_agent(self):

        agent = build_default_registry().build(
            "lead_discovery_agent",
            max_industries=3,
            results_per_query=5
        )

        self.assertEqual(agent.max_industries, 3)
        self.assertEqual(agent.results_per_query, 5)

    def test_producers(self):

        registry = build_default_registry()

        self.assertEqual(
            [s.name for s in registry.producers_of("scores")],
            ["scoring_agent"]
        )


if __name__ == "__main__":
    unittest.main()
