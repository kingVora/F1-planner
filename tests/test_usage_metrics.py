"""Unit tests for usage extraction and cost estimation (no API keys)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from f1_planner.usage_metrics import (
    UsageSnapshot,
    estimate_openai_cost_usd,
    extract_usage,
)


class TestExtractUsage(unittest.TestCase):
    def test_from_crewai_like_token_usage(self):
        token_usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "cached_prompt_tokens": 0,
                "successful_requests": 3,
            }
        )
        result = SimpleNamespace(token_usage=token_usage, tasks_output=[])
        snap = extract_usage(result)
        self.assertTrue(snap.reported)
        self.assertEqual(snap.prompt_tokens, 1000)
        self.assertEqual(snap.completion_tokens, 200)
        self.assertEqual(snap.total_tokens, 1200)
        self.assertEqual(snap.successful_requests, 3)

    def test_infers_total_when_missing(self):
        token_usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
                "successful_requests": 1,
            }
        )
        result = SimpleNamespace(token_usage=token_usage, tasks_output=[])
        snap = extract_usage(result)
        self.assertEqual(snap.total_tokens, 600)

    def test_not_reported_when_empty(self):
        token_usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
                "successful_requests": 0,
            }
        )
        result = SimpleNamespace(token_usage=token_usage, tasks_output=[])
        snap = extract_usage(result)
        self.assertFalse(snap.reported)

    def test_missing_token_usage(self):
        result = SimpleNamespace(token_usage=None, tasks_output=[])
        snap = extract_usage(result)
        self.assertFalse(snap.reported)

    def test_usage_snapshot_add(self):
        a = UsageSnapshot(10, 20, 30, 5, 1, True)
        b = UsageSnapshot(1, 2, 3, 0, 1, True)
        c = a.add(b)
        self.assertEqual(c.prompt_tokens, 11)
        self.assertEqual(c.completion_tokens, 22)
        self.assertEqual(c.total_tokens, 33)
        self.assertEqual(c.cached_prompt_tokens, 5)
        self.assertTrue(c.reported)


class TestEstimateCost(unittest.TestCase):
    def test_estimate_gpt4o(self):
        usage = UsageSnapshot(1_000_000, 1_000_000, 2_000_000, 0, 5, True)
        with patch.dict(os.environ, {}, clear=True):
            cost = estimate_openai_cost_usd(usage, model="gpt-4o")
        self.assertIsNotNone(cost)
        # Default rates: 1M in @ $2.50/M + 1M out @ $10/M
        self.assertAlmostEqual(cost, 2.50 + 10.00, places=4)

    def test_total_only_uses_max_rate_not_input_only(self):
        """When only total_tokens is set, cost uses max(in, out) per 1M — not input-only."""
        usage = UsageSnapshot(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=1_000_000,
            cached_prompt_tokens=0,
            successful_requests=1,
            reported=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            cost = estimate_openai_cost_usd(usage, model="gpt-4o")
        # Default out_rate 10 > in_rate 2.5 → 1M @ $10/M, not $2.5/M
        self.assertAlmostEqual(cost, 10.00, places=4)

    def test_none_when_not_reported(self):
        usage = UsageSnapshot(0, 0, 0, 0, 0, False)
        self.assertIsNone(estimate_openai_cost_usd(usage, model="gpt-4o"))

    def test_env_override_pricing(self):
        usage = UsageSnapshot(1_000_000, 0, 1_000_000, 0, 1, True)
        env = {
            "OPENAI_GPT4O_INPUT_PER_1M_USD": "1.0",
            "OPENAI_GPT4O_OUTPUT_PER_1M_USD": "5.0",
        }
        with patch.dict(os.environ, env, clear=True):
            cost = estimate_openai_cost_usd(usage, model="gpt-4o")
        self.assertAlmostEqual(cost, 1.0, places=4)

    def test_cached_tokens_not_added_to_prompt_total(self):
        """Cached prompt tokens are a subset of prompt_tokens; must not double-count."""
        usage = UsageSnapshot(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            total_tokens=1_000_000,
            cached_prompt_tokens=400_000,
            successful_requests=1,
            reported=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            cost = estimate_openai_cost_usd(usage, model="gpt-4o")
        # 600k uncached @ $2.5/M + 400k cached @ $1.25/M (50% of input); not 1.4M @ input
        self.assertAlmostEqual(cost, 0.6 * 2.50 + 0.4 * 1.25, places=4)

    def test_cached_rate_split_when_env_set(self):
        usage = UsageSnapshot(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            total_tokens=1_000_000,
            cached_prompt_tokens=500_000,
            successful_requests=1,
            reported=True,
        )
        env = {
            "OPENAI_GPT4O_INPUT_PER_1M_USD": "2.0",
            "OPENAI_GPT4O_CACHED_INPUT_PER_1M_USD": "0.5",
        }
        with patch.dict(os.environ, env, clear=True):
            cost = estimate_openai_cost_usd(usage, model="gpt-4o")
        # 500k @ 2 + 500k @ 0.5 = 1.0 + 0.25
        self.assertAlmostEqual(cost, 1.25, places=4)


if __name__ == "__main__":
    unittest.main()
