"""
Output validators.

A validator is a plain function `validator(state) -> ValidationResult`
that inspects the shared state after an agent has run and says whether
the output is good enough to build on. It must not change the state
and must not call any API, so it costs nothing and is easy to test.

Validators are attached to agents through AgentSpec.validator. The
executor runs them; the replanner (a later step) reacts to failures.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationResult:

    passed: bool
    reason: str = ""

    # Numbers behind the verdict, kept in the run record.
    metrics: dict = field(default_factory=dict)

    # Below the TARGET but still above the MINIMUM needed to carry
    # on. When a failed result is acceptable and recovery cannot
    # improve it, the executor continues with it ("degraded") instead
    # of failing the run. None means: acceptable only if passed.
    acceptable: bool | None = None

    @property
    def can_continue(self) -> bool:

        if self.acceptable is None:
            return self.passed

        return self.passed or self.acceptable

    def to_dict(self) -> dict:

        return {
            "passed": self.passed,
            "acceptable": self.can_continue,
            "reason": self.reason,
            "metrics": dict(self.metrics)
        }


def _pass(**metrics) -> ValidationResult:

    return ValidationResult(True, "", metrics)


def _fail(
    reason: str,
    acceptable: bool | None = None,
    **metrics
) -> ValidationResult:

    return ValidationResult(False, reason, metrics, acceptable)


def validate_icp(state) -> ValidationResult:
    """An ICP exists and names at least one target industry."""

    icp = state.icp

    if icp is None:
        return _fail("No ICP was produced.")

    industries = len(icp.target_industries or [])

    if industries == 0:
        return _fail(
            "The ICP has no target industries.",
            industries=0
        )

    return _pass(industries=industries)


def make_leads_validator(min_leads: int = 3, floor: int = 1):
    """
    Discovery found at least `min_leads` leads (the target).

    Fewer than the target triggers recovery. If recovery cannot reach
    the target, the run continues as long as at least `floor` leads
    were found; below the floor it fails.
    """

    if floor > min_leads:
        raise ValueError("floor cannot be above the target.")

    def validate_leads(state) -> ValidationResult:

        found = len(state.leads)

        if found < min_leads:
            return _fail(
                f"Found {found} lead(s); need at least {min_leads}.",
                acceptable=found >= floor,
                leads=found,
                min_leads=min_leads,
                floor=floor
            )

        return _pass(leads=found, min_leads=min_leads, floor=floor)

    return validate_leads


def make_enrichment_validator(min_ok: int = 1):
    """At least `min_ok` leads were enriched successfully."""

    def validate_enrichment(state) -> ValidationResult:

        total = len(state.enriched_leads)

        ok = sum(
            1
            for enriched in state.enriched_leads
            if enriched.status == "ok"
        )

        if ok < min_ok:
            return _fail(
                f"{ok} of {total} lead(s) enriched OK; "
                f"need at least {min_ok}.",
                enriched_ok=ok,
                enriched_total=total,
                min_ok=min_ok
            )

        return _pass(
            enriched_ok=ok,
            enriched_total=total,
            min_ok=min_ok
        )

    return validate_enrichment


def validate_scores(state) -> ValidationResult:
    """At least one lead scored, every score within 0 to 100."""

    scored = len(state.scores)

    if scored == 0:
        return _fail("No leads were scored.", scored=0)

    for name, entry in state.scores.items():

        value = entry.get("score")

        if not isinstance(value, (int, float)) or not (
            0 <= value <= 100
        ):
            return _fail(
                f"Score for '{name}' is not within 0 to 100.",
                scored=scored
            )

    return _pass(scored=scored)