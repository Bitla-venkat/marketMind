"""
The dynamic pipeline: plan, execute, validate, recover.

Same interface as FixedPipeline (`run(startup_description, label)`
returning `(state, record)`), so the two can be run on the same inputs
and compared directly.

Flow of one run:

  1. Planner works backwards from the goal artifact ("scores") through
     the agent registry and builds a task graph.
  2. Executor runs every node whose dependencies are done, in parallel
     where the graph allows it.
  3. After each node its validator checks the output.
  4. If a node fails or its output is rejected, the recovery policy
     picks one action: retry (maybe with wider settings), swap in an
     alternative agent, or give up.
  5. The run record lists every attempt, its cost, and every recovery.

Variants, for ablation studies:

  dynamic              validation + recovery (the full system)
  dynamic-no-recovery  validation only: bad output stops the run
  dynamic-plain        neither: the executor alone
"""

from core.default_registry import build_default_registry
from core.executor import Executor
from core.planner import Planner
from core.recovery import RecoveryPolicy


DYNAMIC_VARIANTS = {
    "dynamic": {"validate": True, "recovery": True},
    "dynamic-no-recovery": {"validate": True, "recovery": False},
    "dynamic-plain": {"validate": False, "recovery": False}
}


class DynamicPipeline:

    def __init__(
        self,
        name: str = "dynamic",
        goal: str = "scores",
        min_leads: int = 3,
        validate: bool = True,
        recovery: bool = True,
        max_recoveries_per_run: int = 3,
        max_concurrency: int = 4
    ):

        self.name = name
        self.goal = goal

        self.registry = build_default_registry(min_leads=min_leads)

        self.plan = Planner(self.registry).plan(goal)

        policy = (
            RecoveryPolicy(
                self.registry,
                max_recoveries_per_run=max_recoveries_per_run
            )
            if recovery
            else None
        )

        self.executor = Executor(
            self.registry,
            max_concurrency=max_concurrency,
            name=name,
            validate=validate,
            recovery=policy
        )

    async def run(
        self,
        startup_description: str,
        label: str | None = None
    ):
        """Plan was built once; every run executes it afresh."""

        return await self.executor.run(
            self.plan,
            startup_description,
            label=label
        )


def build_dynamic_pipeline(
    variant: str = "dynamic",
    **kwargs
) -> DynamicPipeline:
    """
    variant: one of DYNAMIC_VARIANTS. Extra keyword arguments
    (min_leads, max_recoveries_per_run, ...) go to DynamicPipeline.
    """

    if variant not in DYNAMIC_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from: {', '.join(DYNAMIC_VARIANTS)}."
        )

    return DynamicPipeline(
        name=variant,
        **{**DYNAMIC_VARIANTS[variant], **kwargs}
    )
