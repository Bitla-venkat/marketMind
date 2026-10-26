"""
Probe how well the enrichment agent works.

For each example startup this runs ICP -> lead discovery (default
settings) -> enrichment, then reports:

  - enrichment status counts (ok, no_search_results, llm_error, ...)
  - how often each field was filled, among successful enrichments
  - how many evidence URLs were actually in the search results shown
    to the model ("grounded")
  - Gemini calls, tokens and latency spent on enrichment

Full rows and every enriched lead are written to a NEW timestamped JSON
file in experiments/ so you can spot-check facts by hand. Grounding only
shows that a cited URL was supplied, not that the claim is true: check a
sample of leads against their evidence links yourself.

Uses REAL Gemini and Tavily calls. Per startup: 1 ICP call, 1 discovery
call, 3 discovery searches, then 1 search and 1 Gemini call per lead.

Run from the project root:

    python -m experiments.probe_enrichment --n 2
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agents.enrichment_agent import EnrichmentAgent
from agents.icp_agent import ICPAgent
from agents.lead_discovery_agent import LeadDiscoveryAgent
from core.state import AgentState
from experiments.probe_discovery import STARTUPS


FIELDS = [
    "website",
    "linkedin",
    "company_size",
    "location",
    "industry",
    "founders",
    "buyer_roles",
    "products_services",
    "buying_signals",
    "evidence"
]

EXPERIMENTS_DIR = Path(__file__).parent


def fill_rates(enriched: list[dict]) -> dict:
    """
    Fraction of successful enrichments in which each field is
    filled (non-null, non-empty). Failed enrichments are excluded.
    """

    ok = [
        item for item in enriched
        if item["status"] == "ok" and item["details"]
    ]

    if not ok:
        return {}

    return {
        field: round(
            sum(1 for item in ok if item["details"].get(field))
            / len(ok),
            3
        )
        for field in FIELDS
    }


def summarise(rows: list[dict]) -> dict:
    """Overall status counts, fill rates and grounding."""

    status_counts = {}

    all_enriched = []

    evidence_total = 0
    evidence_grounded = 0

    for row in rows:

        for status, count in row["status_counts"].items():
            status_counts[status] = (
                status_counts.get(status, 0) + count
            )

        all_enriched.extend(row["enriched"])

        evidence_total += row["evidence_total"]
        evidence_grounded += row["evidence_grounded"]

    leads = sum(status_counts.values())

    return {
        "startups": len(rows),
        "startups_without_leads": sum(
            1 for row in rows if row["leads"] == 0
        ),
        "leads_enriched": leads,
        "status_counts": status_counts,
        "success_rate": (
            round(status_counts.get("ok", 0) / leads, 3)
            if leads else None
        ),
        "field_fill_rates": fill_rates(all_enriched),
        "evidence_total": evidence_total,
        "evidence_grounded": evidence_grounded,
        "evidence_grounded_rate": (
            round(evidence_grounded / evidence_total, 3)
            if evidence_total else None
        )
    }


async def run_startup(startup: dict) -> dict:

    state = AgentState(startup["description"])

    row = {
        "startup": startup["id"],
        "leads": 0,
        "status_counts": {},
        "evidence_total": 0,
        "evidence_grounded": 0,
        "enrichment_llm_calls": 0,
        "input_tokens": None,
        "output_tokens": None,
        "llm_latency_s": None,
        "enriched": [],
        "error": None
    }

    try:

        await ICPAgent().run(
            task={"type": "define_icp"},
            state=state
        )

        leads = await LeadDiscoveryAgent().run(
            task={"type": "discover_leads"},
            state=state
        )

        row["leads"] = len(leads)

        if leads:

            enriched = await EnrichmentAgent().run(
                task=None,
                state=state
            )

            row["enriched"] = [
                item.model_dump() for item in enriched
            ]

    except Exception as e:

        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    diagnostics = state.diagnostics.get(
        EnrichmentAgent.name,
        {}
    )

    row["status_counts"] = diagnostics.get("status_counts", {})
    row["evidence_total"] = diagnostics.get("evidence_total", 0)
    row["evidence_grounded"] = diagnostics.get(
        "evidence_grounded", 0
    )

    calls = [
        call for call in state.llm_calls
        if call["agent"] == EnrichmentAgent.name
    ]

    row["enrichment_llm_calls"] = len(calls)

    if calls:

        row["input_tokens"] = sum(
            call["input_tokens"] or 0 for call in calls
        )

        row["output_tokens"] = sum(
            call["output_tokens"] or 0 for call in calls
        )

        row["llm_latency_s"] = round(
            sum(call["latency_s"] for call in calls), 2
        )

    return row


async def main(n: int):

    load_dotenv()

    rows = []

    for startup in STARTUPS[:n]:

        print(f"\n######## {startup['id']} ########")

        rows.append(await run_startup(startup))

    summary = summarise(rows)

    print("\n\n==================== RESULTS ====================\n")

    print(
        f"{'startup':<20}{'leads':>6}  {'statuses':<40}"
        f"{'evidence':>9}{'grounded':>9}"
    )

    for row in rows:

        print(
            f"{row['startup']:<20}{row['leads']:>6}  "
            f"{json.dumps(row['status_counts']):<40}"
            f"{row['evidence_total']:>9}"
            f"{row['evidence_grounded']:>9}"
        )

        if row["error"]:
            print(f"    ERROR: {row['error']}")

    print("\nSummary:")

    print(json.dumps(summary, indent=2))

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    out_path = (
        EXPERIMENTS_DIR / f"enrichment_probe_{stamp}.json"
    )

    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
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
        default=2,
        help="Number of example startups to run (max 5)."
    )

    args = parser.parse_args()

    asyncio.run(main(args.n))