"""
Recovery policy: what to do when a node fails or its output is rejected.

The policy is a pure function of what happened, so it needs no API
calls and is easy to test. For a bad node it chooses ONE action:

  retry       run the same agent again, optionally with changed
              params. The agent's `escalations` (see AgentSpec) is a
              ladder of param changes tried in order. An empty change
              `{}` means "same settings again", which helps when the
              result varies from run to run.
  swap_agent  replace the agent with another that produces the same
              artifacts and needs no inputs the node does not already
              have.
  give_up     stop trying; the node is marked failed and everything
              that depends on it is skipped.

Every recovery costs real LLM and search calls, so two limits keep it
bounded: a total budget per run and a cap on actions per node.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecoveryAction:

    kind: str
    reason: str

    # retry: param changes merged over the node's current params.
    params: dict = field(default_factory=dict)

    # swap_agent: the agent swapped in, and the one it replaced.
    agent: str | None = None
    from_agent: str | None = None

    def to_dict(self) -> dict:

        return {
            "kind": self.kind,
            "reason": self.reason,
            "params": dict(self.params),
            "agent": self.agent,
            "from_agent": self.from_agent
        }


class RecoveryPolicy:

    def __init__(
        self,
        registry,
        max_recoveries_per_run: int = 3,
        max_actions_per_node: int = 3
    ):

        self.registry = registry
        self.max_recoveries_per_run = max_recoveries_per_run
        self.max_actions_per_node = max_actions_per_node

    def decide(
        self,
        node,
        step: dict,
        prior: list,
        budget_left: int
    ) -> RecoveryAction:
        """
        node:        the PlanNode that just failed or was rejected.
        step:        its step record (status, error, validation).
        prior:       RecoveryActions already taken for this node.
        budget_left: recoveries still allowed in this run.
        """

        what = f"{node.agent} {step['status']}: {step['error']}"

        if budget_left <= 0:
            return RecoveryAction(
                "give_up",
                f"{what}. Run recovery budget used up."
            )

        if len(prior) >= self.max_actions_per_node:
            return RecoveryAction(
                "give_up",
                f"{what}. Node recovery limit reached."
            )

        # Actions since the last swap belong to the current agent.
        since_swap = []

        for action in prior:

            if action.kind == "swap_agent":
                since_swap = []
            else:
                since_swap.append(action)

        ladder = self.registry.get(node.agent).escalations

        if len(since_swap) < len(ladder):

            change = dict(ladder[len(since_swap)])

            return RecoveryAction(
                "retry",
                f"{what}. Retrying with {change or 'same settings'}.",
                params=change
            )

        candidate = self._swap_candidate(node, prior)

        if candidate is not None:

            return RecoveryAction(
                "swap_agent",
                f"{what}. Swapping to {candidate}.",
                agent=candidate,
                from_agent=node.agent
            )

        return RecoveryAction(
            "give_up",
            f"{what}. No retries or alternatives left."
        )

    def _swap_candidate(self, node, prior) -> str | None:

        tried = {node.agent}

        for action in prior:

            if action.kind == "swap_agent":
                tried.add(action.agent)
                tried.add(action.from_agent)

        for name in node.alternatives:

            if name in tried:
                continue

            spec = self.registry.get(name)

            # The swapped-in agent must work with the inputs the
            # node already has, because the graph does not change.
            if set(spec.requires) <= set(node.requires):
                return name

        return None
