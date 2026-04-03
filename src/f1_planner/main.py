#!/usr/bin/env python
import sys
import time
import warnings
from datetime import datetime

from pydantic import ValidationError

from f1_planner.crew import F1Planner
from f1_planner.logging_config import setup_logging
from f1_planner.schemas import TripInput, post_process_outputs, validate_outputs

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """Run the crew with input validation, structured logging, and output checks."""
    logger, run_id = setup_logging()
    logger.info("Starting F1 Planner run %s", run_id)

    raw_inputs = {
        'source_city': 'Hyderabad',
        'destination_city': 'Singapore',
        'grand_prix': '2026 Singapore Grand Prix',
        'days': 6,
        'amount': '200000',
        'currency_code': 'INR',
        'current_year': str(datetime.now().year),
    }

    try:
        validated = TripInput(**raw_inputs)
    except ValidationError as e:
        logger.error("Input validation failed:\n%s", e)
        raise SystemExit(1)

    inputs = validated.model_dump()
    inputs["nights"] = validated.nights
    logger.info("Validated inputs: %s", inputs)

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            result = F1Planner().crew().kickoff(inputs=inputs)
            elapsed = time.perf_counter() - start
            logger.info("Crew finished in %.1f s (attempt %d/%d)", elapsed, attempt, max_attempts)
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("Crew failed after %.1f s: %s", elapsed, e)
            raise

        logger.info("Post-processing outputs...")
        post_process_outputs()

        logger.info("Running output validation...")
        warnings = validate_outputs()
        failed = {f: w for f, w in warnings.items() if w}

        if not failed:
            logger.info("All outputs passed validation")
            break

        if attempt < max_attempts:
            logger.warning(
                "Attempt %d: %d file(s) failed validation, retrying...",
                attempt, len(failed),
            )
        else:
            logger.warning(
                "Attempt %d: %d file(s) still failing after %d attempts: %s",
                attempt, len(failed), max_attempts, list(failed.keys()),
            )

    print("\n\n=== FINAL DECISION ===\n\n")
    print(result.raw)
    logger.info("Run %s complete", run_id)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        F1Planner().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        F1Planner().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        F1Planner().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = F1Planner().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
