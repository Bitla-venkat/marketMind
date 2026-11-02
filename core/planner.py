"""
Deterministic planner: turns a goal into a task DAG.

Given a goal artifact (for example "scores") and a registry, it works
backwards: find an agent that produces the goal, then agents that
produce what that agent requires, and so on until everything needed is
either already available or an external input. No LLM is involved, so
this is a clean reference to compare any LLM-based planner against.

A Plan has one node per agent. Each node lists the nodes it depends on,
so the executor can run every node whose dependencies are finished,
in parallel where the graph allows it.
"""

from dataclasses import dataclass, field

from core.registry import AgentRegistry, EXTERNAL_ARTIFACTS


class PlanError(Exception):
    """The goal cannot be reached with the registered agents."""


@dataclass(frozen=True)
class PlanNode:

    # Equal to the agent name; a plan holds each agent at most once.
    id: str
    agent: str
    params: dict
    requires: tuple
    produces: tuple

    # Ids of the nodes that must finish first.
    depends_on: tuple

    # Other agents that could produce the same artifacts. A
    # replanner can swap one in if this agent fails.
    alternatives: tuple = ()


@dataclass
class Plan:

    goal: str

    # Topological order: every node appears after its dependencies.
    nodes: list = field(default_factory=list)

    def get(self, node_id: str) -> PlanNode:

        for node in self.nodes:
            if node.id == node_id:
                return node

        raise KeyError(f"No node '{node_id}' in the plan.")

    def ready(self, done: set) -> list:
        """Nodes not yet done whose dependencies are all done."""

        return [
            node
            for node in self.nodes
            if node.id not in done
            and all(dep in done for dep in node.depends_on)
        ]

    def layers(self) -> list:
        """
        Nodes grouped into waves. Nodes in one wave do not depend on
        each other, so they may run in parallel.
        """

        depth = {}

        for node in self.nodes:
            depth[node.id] = 1 + max(
                (depth[dep] for dep in node.depends_on),
                default=-1
            )

        waves = {}

        for node in self.nodes:
            waves.setdefault(depth[node.id], []).append(node)

        return [waves[level] for level in sorted(waves)]

    def to_dict(self) -> dict:
        """Plain dict for logging in run records."""

        return {
            "goal": self.goal,
            "nodes": [
                {
                    "id": node.id,
                    "params": node.params,
                    "depends_on": list(node.depends_on),
                    "alternatives": list(node.alternatives)
                }
                for node in self.nodes
            ]
        }


class Planner:

    def __init__(self, registry: AgentRegistry):

        self.registry = registry

    def plan(
        self,
        goal: str,
        available=None,
        prefer: dict | None = None,
        params: dict | None = None
    ) -> Plan:
        """
        Build a plan that produces `goal`.

        available:
            Artifacts that already exist, so nothing needs to run for
            them. Defaults to the external inputs only.

        prefer:
            {artifact: agent_name} to choose between several agents
            that produce the same artifact. Otherwise the first
            registered producer is used.

        params:
            {agent_name: {param: value}} overriding the spec's
            default params for this plan.
        """

        have = set(EXTERNAL_ARTIFACTS)
        have.update(available or ())

        prefer = prefer or {}
        params = params or {}

        plan = Plan(goal=goal)

        # agent name -> node, for agents already placed in the plan.
        placed = {}

        # Artifacts currently being resolved, to detect cycles.
        in_progress = []

        def resolve(artifact: str):
            """Make sure `artifact` will exist; return its node id."""

            if artifact in have:
                return None

            producers = self.registry.producers_of(artifact)

            if not producers:
                raise PlanError(
                    f"Nothing can produce '{artifact}'."
                )

            if artifact in prefer:

                chosen = [
                    spec
                    for spec in producers
                    if spec.name == prefer[artifact]
                ]

                if not chosen:
                    raise PlanError(
                        f"'{prefer[artifact]}' cannot produce "
                        f"'{artifact}'."
                    )

                spec = chosen[0]

            else:
                spec = producers[0]

            if spec.name in placed:
                return spec.name

            if artifact in in_progress:
                cycle = " -> ".join(in_progress + [artifact])
                raise PlanError(f"Circular dependency: {cycle}.")

            in_progress.append(artifact)

            depends_on = []

            for needed in spec.requires:

                dependency = resolve(needed)

                if dependency and dependency not in depends_on:
                    depends_on.append(dependency)

            in_progress.pop()

            node = PlanNode(
                id=spec.name,
                agent=spec.name,
                params={
                    **spec.default_params,
                    **params.get(spec.name, {})
                },
                requires=tuple(spec.requires),
                produces=tuple(spec.produces),
                depends_on=tuple(depends_on),
                alternatives=tuple(
                    other.name
                    for other in producers
                    if other.name != spec.name
                )
            )

            placed[spec.name] = node

            plan.nodes.append(node)

            return spec.name

        resolve(goal)

        return plan
