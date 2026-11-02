"""
Run the fixed sequential baseline pipelines and record what they cost.

For each example startup and each variant this runs

    ICP -> Lead discovery -> Enrichment -> Scoring

with no validation or recovery (see pipelines/fixed_pipeline.py), then
prints a table and a per-pipeline summary and writes every run record to
a NEW timestamped JSON file in experiments/.

Variants:
  narrow : discovery searches 1 industry, 3 results per query
  wide   : discovery searches 3 industries, 5 results per query

Uses REAL Gemini and Tavily calls. A single run makes roughly 7 to 12
Gemini calls (ICP, discovery, one per lead for enrichment, scoring) and
4 to 15 Tavily searches, so start small.

Run from the project root:

    python -m experiments.run_baseline --n 2
    python -m experiments.run_baseline --n 2 --variants narrow,dynamic --repeat 3
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from experiments.probe_discovery import STARTUPS
from pipelines.dynamic_pipeline import (
    DYNAMIC_VARIANTS,
    build_dynamic_pipeline
)
from pipelines.fixed_pipeline import VARIANTS, build_fixed_pipeline

# Every name --variants accepts. Fixed ones first.
ALL_VARIANTS = [*VARIANTS, *DYNAMIC_VARIANTS]

# Variants run when --variants is not given.
DEFAULT_VARIANTS = ["narrow", "wide", "dynamic"]

# A "good" run for the stricter success rate: it also scored at
# least this many leads.
MIN_GOOD_LEADS = 3


def build_pipeline(variant: str):
    """The pipeline object for a variant name."""

    if variant in VARIANTS:
        return build_fixed_pipeline(variant)

    return build_dynamic_pipeline(variant)


EXPERIMENTS_DIR = Path(__file__).parent


def _mean(values: list) -> float | None:

    return round(sum(values) / len(values), 2) if values else None


def _sum_models(runs: list[dict]) -> dict:
    """Total calls served by each model across the runs."""

    totals = {}

    for run in runs:
        for model, count in run.get("models_used", {}).items():
            totals[model] = totals.get(model, 0) + count

    return totals


def summarise_runs(records: list[dict]) -> dict:
    """Per-pipeline averages and failure breakdown."""

    summary = {}

    for name in sorted({r["pipeline"] for r in records}):

        runs = [r for r in records if r["pipeline"] == name]

        successes = [r for r in runs if r["success"]]

        failed_steps = {}

        for run in runs:

            if run["failed_step"]:
                failed_steps[run["failed_step"]] = (
                    failed_steps.get(run["failed_step"], 0) + 1
                )

        # Runs that raised nothing but scored nothing.
        empty_runs = sum(
            1
            for run in runs
            if run["failed_step"] is None
            and run["leads_scored"] == 0
        )

        good = [
            r for r in successes
            if r["leads_scored"] >= MIN_GOOD_LEADS
        ]

        summary[name] = {
            "runs": len(runs),
            "success_rate": round(len(successes) / len(runs), 3),
            f"success_rate_min{MIN_GOOD_LEADS}_leads": round(
                len(good) / len(runs), 3
            ),
            "mean_recoveries": _mean(
                [len(r.get("recoveries", [])) for r in runs]
            ),
            "models_used": _sum_models(runs),
            "failed_steps": failed_steps,
            "finished_with_no_scored_leads": empty_runs,
            "mean_latency_s": _mean([r["latency_s"] for r in runs]),
            "mean_llm_calls": _mean([r["llm_calls"] for r in runs]),
            "mean_llm_failed_attempts": _mean(
                [r["llm_failed_attempts"] for r in runs]
            ),
            "mean_input_tokens": _mean(
                [r["input_tokens"] for r in runs]
            ),
            "mean_output_tokens": _mean(
                [r["output_tokens"] for r in runs]
            ),
            "mean_search_calls": _mean(
                [r["search_calls"] for r in runs]
            ),
            "mean_leads_scored": _mean(
                [r["leads_scored"] for r in runs]
            )
        }

    return summary


async def main(n: int, variants: list[str], repeat: int):

    load_dotenv()

    records = []

    for startup in STARTUPS[:n]:

        for variant in variants:

            for run_number in range(1, repeat + 1):

                print(
                    f"\n######## {startup['id']} / fixed-{variant} "
                    f"/ run {run_number}/{repeat} ########"
                )

                pipeline = build_pipeline(variant)

                _, record = await pipeline.run(
                    startup["description"],
                    label=startup["id"]
                )

                record["repeat"] = run_number

                records.append(record)

    summary = summarise_runs(records)

    print("\n\n==================== RESULTS ====================\n")

    print(
        f"{'startup':<19}{'pipeline':<22}{'run':>4}  {'ok':<6}"
        f"{'failed step':<22}{'sec':>7}{'llm':>5}{'search':>7}"
        f"{'rec':>4}{'found':>6}{'enr':>5}{'scored':>7}"
    )

    for record in records:

        print(
            f"{record['label']:<19}{record['pipeline']:<22}"
            f"{record['repeat']:>4}  "
            f"{str(record['success']):<6}"
            f"{str(record['failed_step'] or '-'):<22}"
            f"{record['latency_s']:>7.1f}"
            f"{record['llm_calls']:>5}"
            f"{record['search_calls']:>7}"
            f"{len(record.get('recoveries', [])):>4}"
            f"{record['leads_found']:>6}"
            f"{record['leads_enriched_ok']:>5}"
            f"{record['leads_scored']:>7}"
        )

    print("\nSummary:")

    print(json.dumps(summary, indent=2))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    out_path = EXPERIMENTS_DIR / f"baseline_run_{stamp}.json"

    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "variants": variants,
                "repeat": repeat,
                "records": records,
                "summary": summary
            },
            indent=2
        )
    )

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of example startups to run (max 5)."
    )

    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "Comma-separated, from: "
            + ", ".join(ALL_VARIANTS)
            + f" (default {','.join(DEFAULT_VARIANTS)})."
        )
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Runs per startup and variant."
    )

    args = parser.parse_args()

    chosen = [
        item.strip()
        for item in args.variants.split(",")
        if item.strip()
    ]

    unknown = [item for item in chosen if item not in ALL_VARIANTS]

    if unknown:
        parser.error(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(ALL_VARIANTS)}."
        )

    if args.repeat < 1:
        parser.error("--repeat must be at least 1.")

    asyncio.run(main(args.n, chosen, args.repeat))
