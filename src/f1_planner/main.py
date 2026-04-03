#!/usr/bin/env python
import os
import sys
import time
import warnings
from datetime import datetime

from pydantic import ValidationError

from f1_planner.crew import F1Planner
from f1_planner.logging_config import setup_logging
from f1_planner.schemas import TripInput, post_process_outputs, validate_outputs
from f1_planner.usage_metrics import (
    UsageSnapshot,
    estimate_openai_cost_usd,
    extract_usage,
    format_usage_summary,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """Run the crew."""
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

    pricing_model = os.environ.get("F1_PLANNER_PRICING_MODEL", "gpt-4o")
    cumulative_usage = UsageSnapshot(0, 0, 0, 0, 0, False)
    last_usage = cumulative_usage
    last_attempt = 1

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

        last_usage = extract_usage(result)
        last_attempt = attempt
        cumulative_usage = cumulative_usage.add(last_usage)
        est = estimate_openai_cost_usd(last_usage, model=pricing_model)
        logger.info(
            "LLM usage attempt %d/%d: prompt=%d completion=%d total=%d requests=%d est_openai_usd=%s",
            attempt,
            max_attempts,
            last_usage.prompt_tokens,
            last_usage.completion_tokens,
            last_usage.total_tokens,
            last_usage.successful_requests,
            f"{est:.4f}" if est is not None else "n/a",
        )
        if not last_usage.reported:
            logger.info(
                "LLM usage not reported for attempt %d (provider may omit token counts).",
                attempt,
            )

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

    cumulative_cost = estimate_openai_cost_usd(cumulative_usage, model=pricing_model)
    print(
        format_usage_summary(
            last_usage,
            last_attempt,
            max_attempts,
            cumulative_usage if last_attempt > 1 else None,
            cumulative_cost if last_attempt > 1 else None,
            pricing_model,
        )
    )
    est_final = estimate_openai_cost_usd(last_usage, model=pricing_model)
    if last_attempt > 1:
        logger.info(
            "Run total LLM usage (all attempts): total_tokens=%d est_openai_usd=%s",
            cumulative_usage.total_tokens,
            f"{cumulative_cost:.4f}" if cumulative_cost is not None else "n/a",
        )
    elif est_final is not None:
        logger.info(
            "Run total LLM usage: total_tokens=%d est_openai_usd=%.4f",
            last_usage.total_tokens,
            est_final,
        )
    elif not last_usage.reported:
        logger.info("Run total LLM usage: not reported by provider")

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
