"""
Agent Registry: what each agent needs, what it produces, and how to
build it.

The orchestrator never looks inside an agent. It reads only these
specs. An agent is usable by the orchestrator if it has

  * an `async run(task, state)` method (see agents/base.py),
  * an AgentSpec declaring the artifacts it reads (`requires`) and
    writes (`produces`), and optionally a validator for its output,
  * cost reporting through state.llm_calls and, if it searches,
    `search_tool.search_count` (as the existing agents do).

An "artifact" is a named piece of data in the shared state, such as
"icp" or "leads". `startup_description` is the one external input.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


# Artifacts supplied by the caller rather than produced by an agent.
EXTERNAL_ARTIFACTS = frozenset({"startup_description"})


@dataclass(frozen=True)
class AgentSpec:
    """Description of one agent, as the orchestrator sees it."""

    name: str
    description: str

    # Artifact names this agent reads / writes.
    requires: tuple = ()
    produces: tuple = ()

    # Builds the agent: factory(**params) -> agent instance.
    factory: Callable[..., Any] = None

    # Tunable settings the replanner may change (for example
    # {"max_industries": 1}). Passed to the factory on build.
    default_params: dict = field(default_factory=dict)

    # Rough relative cost, only a prior for the optimizer. Real
    # costs are always measured, never taken from here.
    cost_hint: dict = field(default_factory=dict)

    # Optional check on this agent's output: validator(state) ->
    # ValidationResult (see core/validation.py). The executor runs it
    # after the agent finishes. None means "accept whatever it did".
    validator: Callable[..., Any] | None = None

    # Recovery ladder: param changes to try, in order, when this
    # agent fails or its output is rejected. {} means "same
    # settings again". Empty means no retries (see core/recovery.py).
    escalations: tuple = ()


class AgentRegistry:

    def __init__(self):

        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:

        if not spec.name:
            raise ValueError("An agent spec needs a name.")

        if spec.name in self._specs:
            raise ValueError(
                f"Agent '{spec.name}' is already registered."
            )

        if not spec.produces:
            raise ValueError(
                f"Agent '{spec.name}' must produce at least one "
                "artifact."
            )

        if spec.factory is None:
            raise ValueError(
                f"Agent '{spec.name}' needs a factory."
            )

        overlap = set(spec.requires) & set(spec.produces)

        if overlap:
            raise ValueError(
                f"Agent '{spec.name}' both requires and produces: "
                f"{sorted(overlap)}."
            )

        self._specs[spec.name] = spec

    def get(self, name: str) -> AgentSpec:

        if name not in self._specs:
            raise KeyError(f"Unknown agent '{name}'.")

        return self._specs[name]

    def names(self) -> list[str]:

        return list(self._specs)

    def specs(self) -> list[AgentSpec]:

        return list(self._specs.values())

    def producers_of(self, artifact: str) -> list[AgentSpec]:
        """Every agent that can produce this artifact."""

        return [
            spec
            for spec in self._specs.values()
            if artifact in spec.produces
        ]

    def build(self, name: str, **params) -> Any:
        """
        Create an agent instance. `params` override the spec's
        default_params.
        """

        spec = self.get(name)

        return spec.factory(**{**spec.default_params, **params})

    def check(self) -> list[str]:
        """
        Problems that would stop a plan from ever being built.
        An empty list means the registry is consistent.
        """

        problems = []

        producible = set(EXTERNAL_ARTIFACTS)

        for spec in self._specs.values():
            producible.update(spec.produces)

        for spec in self._specs.values():
            for artifact in spec.requires:

                if artifact not in producible:
                    problems.append(
                        f"'{spec.name}' requires '{artifact}', "
                        "which nothing produces."
                    )

        return problems
