"""
Offline tests for the named-artifact interface on AgentState.

Run from the project root:

    python -m unittest tests.test_state_artifacts -v
"""

import unittest

from core.state import AgentState


class ArtifactTests(unittest.TestCase):

    def test_known_names_use_the_existing_attributes(self):

        state = AgentState("a startup")

        self.assertEqual(
            state.get_artifact("startup_description"),
            "a startup"
        )

        state.set_artifact("leads", ["x"])
        self.assertEqual(state.leads, ["x"])

        state.scores = {"A": {"score": 5}}
        self.assertEqual(
            state.get_artifact("scores"),
            {"A": {"score": 5}}
        )

        state.set_artifact("icp", "profile")
        self.assertEqual(state.icp, "profile")

    def test_new_names_go_to_the_artifacts_dict(self):

        state = AgentState("a startup")

        self.assertIsNone(state.get_artifact("outreach_drafts"))

        state.set_artifact("outreach_drafts", ["hello"])

        self.assertEqual(
            state.artifacts,
            {"outreach_drafts": ["hello"]}
        )
        self.assertEqual(
            state.get_artifact("outreach_drafts"),
            ["hello"]
        )

    def test_has_artifact_treats_empty_as_missing(self):

        state = AgentState("a startup")

        # Fresh state: nothing produced yet.
        for name in ("icp", "leads", "enriched_leads", "scores"):
            self.assertFalse(state.has_artifact(name), name)

        self.assertTrue(state.has_artifact("startup_description"))

        state.leads = []
        self.assertFalse(state.has_artifact("leads"))

        state.leads = ["x"]
        self.assertTrue(state.has_artifact("leads"))

        state.set_artifact("custom", {})
        self.assertFalse(state.has_artifact("custom"))

        state.set_artifact("custom", 0)
        self.assertTrue(state.has_artifact("custom"))

        state.icp = object()
        self.assertTrue(state.has_artifact("icp"))

    def test_states_do_not_share_artifacts(self):

        first = AgentState("a")
        second = AgentState("b")

        first.set_artifact("custom", [1])

        self.assertIsNone(second.get_artifact("custom"))


if __name__ == "__main__":
    unittest.main()
