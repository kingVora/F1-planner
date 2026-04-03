"""Extract LLM token usage from CrewAI results and estimate OpenAI USD cost.

Costs are estimates only; refresh defaults from https://openai.com/pricing when
rates change. SerpApi / Serper usage is billed separately and not included here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Defaults for gpt-4o (USD per 1M tokens). Override via env for 12-factor config.
# Last reviewed: align with OpenAI published API pricing.
_DEFAULT_GPT4O_INPUT_PER_1M = 2.50
_DEFAULT_GPT4O_OUTPUT_PER_1M = 10.00

# OpenAI Prompt Caching: GPT-4o lists cached input at half the uncached input rate
# ($1.25/M vs $2.50/M). Same 50% pattern for GPT-4o mini in their table.
# https://openai.com/index/api-prompt-caching/
_CACHED_INPUT_FRACTION = 0.5


@dataclass(frozen=True)
class UsageSnapshot:
    """Normalized token counts from a single crew kickoff."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int
    successful_requests: int
    reported: bool

    def add(self, other: UsageSnapshot) -> UsageSnapshot:
        return UsageSnapshot(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            successful_requests=self.successful_requests + other.successful_requests,
            reported=self.reported or other.reported,
        )


def extract_usage(result: Any) -> UsageSnapshot:
    """Read token usage from a CrewAI ``CrewOutput`` (or compatible object).

    If ``token_usage`` is missing or all counts are zero, returns a snapshot with
    ``reported=False`` (some providers never populate metrics).
    """
    token_usage = getattr(result, "token_usage", None)
    if token_usage is None:
        return UsageSnapshot(0, 0, 0, 0, 0, False)

    if hasattr(token_usage, "model_dump"):
        data = token_usage.model_dump()
    elif isinstance(token_usage, dict):
        data = token_usage
    else:
        data = {
            "prompt_tokens": int(getattr(token_usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(token_usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(token_usage, "total_tokens", 0) or 0),
            "cached_prompt_tokens": int(
                getattr(token_usage, "cached_prompt_tokens", 0) or 0
            ),
            "successful_requests": int(
                getattr(token_usage, "successful_requests", 0) or 0
            ),
        }

    prompt = int(data.get("prompt_tokens", 0) or 0)
    completion = int(data.get("completion_tokens", 0) or 0)
    total = int(data.get("total_tokens", 0) or 0)
    cached = int(data.get("cached_prompt_tokens", 0) or 0)
    requests = int(data.get("successful_requests", 0) or 0)

    # CrewAI may only fill total; infer missing pieces for cost estimate when possible.
    if total == 0 and (prompt > 0 or completion > 0):
        total = prompt + completion
    reported = total > 0 or prompt > 0 or completion > 0 or requests > 0
    return UsageSnapshot(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_prompt_tokens=cached,
        successful_requests=requests,
        reported=reported,
    )


def _pricing_for_model(model: str) -> tuple[float, float]:
    """Return (input_per_1m_usd, output_per_1m_usd)."""
    m = (model or "gpt-4o").lower().replace("openai/", "")
    if m == "gpt-4o":
        inp = float(
            os.environ.get(
                "OPENAI_GPT4O_INPUT_PER_1M_USD", str(_DEFAULT_GPT4O_INPUT_PER_1M)
            )
        )
        out = float(
            os.environ.get(
                "OPENAI_GPT4O_OUTPUT_PER_1M_USD", str(_DEFAULT_GPT4O_OUTPUT_PER_1M)
            )
        )
        return inp, out
    # Unknown model: use gpt-4o defaults but caller can set env vars for primary model
    return _pricing_for_model("gpt-4o")


def estimate_openai_cost_usd(
    usage: UsageSnapshot,
    model: str = "gpt-4o",
) -> float | None:
    """Estimated spend in USD from token counts. Returns None if usage not reported."""
    if not usage.reported:
        return None
    inp_rate, out_rate = _pricing_for_model(model)

    # Some runtimes only populate total_tokens. True cost is between (total * input_rate)
    # and (total * output_rate) per million; using input-only undersells when tokens are
    # mostly completions. Use max(input, output) as a conservative single-number bound.
    if usage.prompt_tokens == 0 and usage.completion_tokens == 0 and usage.total_tokens > 0:
        max_rate = max(inp_rate, out_rate)
        logger.warning(
            "OpenAI cost estimate: total_tokens=%d but prompt_tokens and completion_tokens "
            "are both zero — using max(input, output) $/1M (conservative bound, not exact).",
            usage.total_tokens,
        )
        return (usage.total_tokens / 1_000_000.0) * max_rate

    # OpenAI-style usage: prompt_tokens is the total size of the prompt. Cached prompt tokens
    # are a subset — never add to prompt_tokens. When cached > 0, split billing: uncached at
    # inp_rate, cached at OPENAI_GPT4O_CACHED_INPUT_PER_1M_USD or default inp_rate * 0.5
    # (OpenAI’s published GPT-4o prompt-caching discount).
    cached = min(usage.cached_prompt_tokens, usage.prompt_tokens)
    non_cached = max(0, usage.prompt_tokens - cached)
    cached_inp_env = os.environ.get("OPENAI_GPT4O_CACHED_INPUT_PER_1M_USD")
    if cached > 0:
        if cached_inp_env:
            cached_inp_rate = float(cached_inp_env)
        else:
            cached_inp_rate = inp_rate * _CACHED_INPUT_FRACTION
        billable_prompt_cost = (non_cached / 1_000_000.0) * inp_rate + (
            cached / 1_000_000.0
        ) * cached_inp_rate
    else:
        billable_prompt_cost = (usage.prompt_tokens / 1_000_000.0) * inp_rate
    return billable_prompt_cost + (usage.completion_tokens / 1_000_000.0) * out_rate


def format_usage_summary(
    usage: UsageSnapshot,
    attempt: int,
    max_attempts: int,
    cumulative: UsageSnapshot | None,
    cumulative_cost_usd: float | None,
    model: str,
) -> str:
    """Human-readable block for stdout."""
    lines = [
        "",
        "=== LLM usage (this run) ===",
        f"Attempt: {attempt}/{max_attempts}",
        f"Model (for pricing): {model}",
    ]
    if usage.reported:
        lines.extend(
            [
                f"Prompt tokens:     {usage.prompt_tokens:,}",
                f"Completion tokens: {usage.completion_tokens:,}",
                f"Total tokens:      {usage.total_tokens:,}",
            ]
        )
        if usage.cached_prompt_tokens:
            lines.append(f"Cached prompt tok: {usage.cached_prompt_tokens:,}")
        if usage.successful_requests:
            lines.append(f"LLM requests:      {usage.successful_requests:,}")
        est = estimate_openai_cost_usd(usage, model=model)
        if est is not None:
            lines.append(f"Est. OpenAI cost:  ${est:.4f} USD (approx.)")
    else:
        lines.append("Token usage not reported by the runtime (often provider-specific).")

    lines.append("Note: SerpApi / Serper charges are not included.")
    if cumulative is not None:
        lines.append("")
        lines.append("--- Cumulative (all attempts) ---")
        lines.append(f"Total tokens: {cumulative.total_tokens:,}")
        if cumulative_cost_usd is not None:
            lines.append(f"Est. OpenAI cost: ${cumulative_cost_usd:.4f} USD (approx.)")
    lines.append("")
    return "\n".join(lines)
