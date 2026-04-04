from __future__ import annotations

import argparse
import sys
from datetime import datetime

from f1_planner.main import run_crew_with_trip_inputs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="f1-plan",
        description="Run the F1 trip planner crew with trip inputs from the command line.",
    )
    p.add_argument(
        "--source-city",
        default="Hyderabad",
        help="Departure city (default: %(default)s)",
    )
    p.add_argument(
        "--destination-city",
        default="Singapore",
        help="Grand Prix host city (default: %(default)s)",
    )
    p.add_argument(
        "--grand-prix",
        default="2026 Singapore Grand Prix",
        metavar="NAME",
        help="Race name label for agents (default: %(default)s)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=6,
        help="Trip length in calendar days (arrival through departure; default: %(default)s)",
    )
    p.add_argument(
        "--amount",
        default="200000",
        help="Total budget as digits only, no decimals (default: %(default)s)",
    )
    p.add_argument(
        "--currency",
        default="INR",
        metavar="CODE",
        help="ISO 4217 currency code (default: %(default)s)",
    )
    p.add_argument(
        "--year",
        default=None,
        metavar="YYYY",
        help="Season year for agents (default: current calendar year)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    year = args.year if args.year is not None else str(datetime.now().year)
    raw_inputs = {
        "source_city": args.source_city,
        "destination_city": args.destination_city,
        "grand_prix": args.grand_prix,
        "days": args.days,
        "amount": args.amount,
        "currency_code": args.currency,
        "current_year": year,
    }
    run_crew_with_trip_inputs(raw_inputs)


if __name__ == "__main__":
    main(sys.argv[1:])
