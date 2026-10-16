"""
Probe how reliably lead discovery returns leads.

For each example startup this runs the ICP agent once, then runs lead
discovery under one or more settings, optionally several times each
against the SAME ICP (so repeats show run-to-run randomness):

  baseline : 1 industry, 3 Tavily results per query  (current behaviour)
  wide     : 3 industries, 5 Tavily results per query

It prints a table and a summary, and writes the full rows plus the
ICPs used to a NEW timestamped file in experiments/, so earlier
results are never overwritten.

Uses REAL Gemini and Tavily calls. Per startup, per repeat:
  baseline = 1 Gemini call + 3 Tavily searches
  wide     = 1 Gemini call + 9 Tavily searches
plus 1 ICP Gemini call per startup.

Run from the project root. Examples:

    python -m experiments.probe_discovery --n 5
    python -m experiments.probe_discovery --only gst_billing \\
        --configs wide --repeat 5
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agents.icp_agent import ICPAgent
from agents.lead_discovery_agent import LeadDiscoveryAgent
from core.state import AgentState


# Example inputs written for this probe. Replace or extend them
# with startups relevant to your evaluation.
STARTUPS = [
    {
        "id": "marketmind",
        "description": (
            "MarketMind is an AI marketing platform for Indian "
            "small and medium-sized businesses. It helps them "
            "generate personalized marketing campaigns, identify "
            "potential customers, and automate parts of their "
            "marketing workflow. The target market is India."
        )
    },
    {
        "id": "gst_billing",
        "description": (
            "A cloud billing and GST-compliance tool for small "
            "retailers and traders in India. It creates invoices, "
            "files GST returns and tracks payments."
        )
    },
    {
        "id": "ecom_logistics",
        "description": (
            "A shipping aggregator for small online sellers in "
            "India. It compares courier rates, automates label "
            "printing and tracks deliveries and returns."
        )
    },
    {
        "id": "factory_hr",
        "description": (
            "An attendance and payroll software for small "
            "manufacturing units in India, with shift management "
            "and statutory compliance (PF, ESI)."
        )
    },
    {
        "id": "whatsapp_commerce",
        "description": (
            "A WhatsApp-based ordering and customer-support "
            "chatbot for local Indian businesses such as "
            "restaurants, salons and clinics."
        )
    }
]

CONFIGS = {
    "baseline": {
        "max_industries": 1,
        "results_per_query": 3
    },
    "wide": {
        "max_industries": 3,
        "results_per_query": 5
    }
}

EXPERIMENTS_DIR = Path(__file__).parent


def is_error_outcome(outcome: str) -> bool:
    """True for infrastructure failures, not discovery quality."""

    return outcome.endswith("_error") or outcome == "icp_failed"


def summarise(rows: list[dict]) -> dict:
    """
    Zero-lead rate and mean leads per configuration, plus a
    per-startup breakdown.

    Infrastructure failures (Gemini/Tavily errors) are counted
    separately so they are not confused with a discovery run that
    genuinely found nothing.
    """

    summary = {}

    for config in CONFIGS:

        config_rows = [
            row for row in rows
            if row["config"] == config
        ]

        if not config_rows:
            continue

        zero = [
            row for row in config_rows
            if row["leads"] == 0
        ]

        errors = [
            row for row in config_rows
            if is_error_outcome(row["outcome"])
        ]

        answered = [
            row for row in config_rows
            if not is_error_outcome(row["outcome"])
        ]

        zero_answered = [
            row for row in answered
            if row["leads"] == 0
        ]

        outcomes = {}

        for row in config_rows:
            outcomes[row["outcome"]] = (
                outcomes.get(row["outcome"], 0) + 1
            )

        summary[config] = {
            "runs": len(config_rows),
            "zero_lead_runs": len(zero),
            "zero_lead_rate": round(
                len(zero) / len(config_rows), 3
            ),
            "error_runs": len(errors),
            "zero_lead_rate_excluding_errors": (
                round(len(zero_answered) / len(answered), 3)
                if answered else None
            ),
            "mean_leads": round(
                sum(row["leads"] for row in config_rows)
                / len(config_rows),
                2
            ),
            "outcomes": outcomes
        }

    by_startup = {}

    for row in rows:

        if "startup" not in row:
            continue

        key = f"{row['startup']}/{row['config']}"

        cell = by_startup.setdefault(
            key,
            {"runs": 0, "zero_lead_runs": 0}
        )

        cell["runs"] += 1

        if row["leads"] == 0:
            cell["zero_lead_runs"] += 1

    if by_startup:
        summary["by_startup"] = by_startup

    return summary


async def run_discovery(
    startup: dict,
    icp,
    config_name: str,
    config: dict,
    repeat: int = 1
) -> dict:
    """Run discovery once and return one result row."""

    state = AgentState(startup["description"])
    state.icp = icp

    row = {
        "startup": startup["id"],
        "config": config_name,
        "repeat": repeat,
        "outcome": None,
        "leads": 0,
        "leads_grounded": 0,
        "search_results_unique": 0,
        "llm_latency_s": None,
        "input_tokens": None,
        "output_tokens": None,
        "models": [],
        "failures": [],
        "industries": [],
        "result_titles": None,
        "error": None
    }

    try:

        agent = LeadDiscoveryAgent(**config)

        leads = await agent.run(
            task={"type": "discover_leads"},
            state=state
        )

        row["leads"] = len(leads)

    except Exception as e:

        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    diagnostics = state.diagnostics.get(
        LeadDiscoveryAgent.name,
        {}
    )

    row["outcome"] = (
        diagnostics.get("outcome") or "error"
    )

    row["leads_grounded"] = diagnostics.get(
        "leads_grounded", 0
    )

    row["search_results_unique"] = diagnostics.get(
        "search_results_unique", 0
    )

    row["industries"] = diagnostics.get("industries", [])

    # Keep the titles the model was shown only for runs that
    # produced nothing, so they can be inspected.
    if row["leads"] == 0:
        row["result_titles"] = diagnostics.get("result_titles")

    calls = state.llm_calls

    if calls:

        row["llm_latency_s"] = round(
            sum(call["latency_s"] for call in calls), 2
        )

        row["input_tokens"] = sum(
            call["input_tokens"] or 0 for call in calls
        )

        row["output_tokens"] = sum(
            call["output_tokens"] or 0 for call in calls
        )

        row["models"] = [call["model"] for call in calls]

    # Failed Gemini attempts: from successful calls (retried
    # then recovered) and from a call that failed outright.
    row["failures"] = [
        failure
        for call in calls
        for failure in call["failures"]
    ] + diagnostics.get("llm_failures", [])

    return row


def parse_csv(value: str | None) -> list[str]:

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


async def main(
    n: int,
    repeat: int,
    only: list[str],
    configs: list[str]
):

    load_dotenv()

    if only:
        startups = [s for s in STARTUPS if s["id"] in only]
    else:
        startups = STARTUPS[:n]

    if not startups:
        print(
            "No matching startups. Available ids: "
            + ", ".join(s["id"] for s in STARTUPS)
        )
        return

    selected_configs = {
        name: CONFIGS[name]
        for name in (configs or CONFIGS)
    }

    rows = []
    icps = {}

    for startup in startups:

        print(f"\n######## {startup['id']} ########")

        icp_state = AgentState(startup["description"])

        try:

            icp = await ICPAgent().run(
                task={"type": "define_icp"},
                state=icp_state
            )

        except Exception as e:

            print(f"ICP failed: {type(e).__name__}: {e}")

            for config_name in selected_configs:
                rows.append({
                    "startup": startup["id"],
                    "config": config_name,
                    "repeat": 1,
                    "outcome": "icp_failed",
                    "leads": 0,
                    "leads_grounded": 0,
                    "search_results_unique": 0,
                    "llm_latency_s": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "models": [],
                    "failures": [],
                    "industries": [],
                    "result_titles": None,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"
                })

            continue

        icps[startup["id"]] = icp.model_dump()

        for config_name, config in selected_configs.items():

            for run_number in range(1, repeat + 1):

                print(
                    f"\n--- {startup['id']} / {config_name} "
                    f"/ run {run_number}/{repeat} ---"
                )

                rows.append(
                    await run_discovery(
                        startup,
                        icp,
                        config_name,
                        config,
                        run_number
                    )
                )

    summary = summarise(rows)

    print("\n\n==================== RESULTS ====================\n")

    print(
        f"{'startup':<20}{'config':<10}{'run':>4}  "
        f"{'outcome':<20}{'results':>8}{'leads':>7}{'grounded':>10}"
    )

    for row in rows:

        print(
            f"{row['startup']:<20}{row['config']:<10}"
            f"{row['repeat']:>4}  "
            f"{row['outcome']:<20}"
            f"{row['search_results_unique']:>8}"
            f"{row['leads']:>7}"
            f"{row['leads_grounded']:>10}"
        )

    print("\nSummary:")

    print(json.dumps(summary, indent=2))

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    out_path = (
        EXPERIMENTS_DIR / f"discovery_probe_{stamp}.json"
    )

    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "configs": selected_configs,
                "repeat": repeat,
                "icps": icps,
                "rows": rows,
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
        default=3,
        help="Number of example startups to run (max 5)."
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Runs per startup/config on the same ICP."
    )

    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated startup ids (overrides --n)."
    )

    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated: baseline, wide (default both)."
    )

    args = parser.parse_args()

    chosen_configs = parse_csv(args.configs)

    unknown = [c for c in chosen_configs if c not in CONFIGS]

    if unknown:
        parser.error(
            f"Unknown config(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(CONFIGS)}."
        )

    if args.repeat < 1:
        parser.error("--repeat must be at least 1.")

    asyncio.run(
        main(
            args.n,
            args.repeat,
            parse_csv(args.only),
            chosen_configs
        )
    )