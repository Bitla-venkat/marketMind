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
     alternative agent, or give up. A rejected result that is still
     above the validator's minimum is kept as "degraded" and the run
     carries on with it.
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
        min_leads_floor: int = 1,
        validate: bool = True,
        recovery: bool = True,
        max_recoveries_per_run: int = 3,
        max_concurrency: int = 4
    ):

        self.name = name
        self.goal = goal

        self.registry = build_default_registry(
            min_leads=min_leads,
            min_leads_floor=min_leads_floor
        )

        self._planner = Planner(self.registry)

        # The plan used when no ICP is supplied.
        self.plan = self._planner.plan(goal)

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
        label: str | None = None,
        icp=None
    ):
        """
        Execute the plan afresh.

        icp: a ready-made ICP. The planner then leaves the ICP agent
        out of the plan, so every pipeline can start from the same
        ICP.
        """

        if icp is None:

            return await self.executor.run(
                self.plan,
                startup_description,
                label=label
            )

        plan = self._planner.plan(self.goal, available={"icp"})

        return await self.executor.run(
            plan,
            startup_description,
            label=label,
            initial={"icp": icp}
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
