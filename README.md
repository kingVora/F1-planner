# F1 Trip Planner

A multi-agent AI system built with [CrewAI](https://crewai.com) that plans a complete, budget-aware trip to any Formula 1 Grand Prix — flights, accommodation, race tickets, city itinerary, and a final executive briefing — all from a single set of inputs.

## What it does

Given a source city, destination Grand Prix, trip duration, and budget, the crew runs six specialist agents in sequence and produces six output files:

| Agent | Output file | What it produces |
|---|---|---|
| Travel Logistics | `flight_research.md` | Live flight options with prices and a Google Flights booking link |
| Travel Logistics | `accommodation.md` | Three hotel tiers (Trackside / Hub / Value Play) with nightly rates and booking links |
| F1 Experience Strategist | `race_ticket.md` | Grandstand recommendations with live ticket prices, circuit map insight, and budget assessment |
| Budget Planner | `budget_plan.md` | Accommodation tier selection, daily allowance, budget classification (CRITICAL / BUDGET / COMFORTABLE / LUXURY), and spending allocation |
| Local Guide | `local_guide.md` | Day-by-day city itinerary for all trip days, with costs targeting the daily allowance |
| Master Planner | `master_planner.md` | Full executive briefing — day-by-day table, total trip cost, risks, and booking priorities |

## Agents

- **Travel Logistics Agent** — Researches race weekend dates, calculates arrival/departure dates, searches live flights (via SerpApi Google Flights), and searches three hotel tiers (via SerpApi Google Hotels). Split into two tasks to keep each tool-call count low and prevent iteration exhaustion.
- **F1 Experience Strategist** — Searches official ticket categories and prices, fetches live exchange rates, recommends the best-value grandstand, and performs per-ticket budget assessment.
- **Budget Planner Agent** — Selects the hotel tier using a top-down trial (Trackside → Hub → Value Play) to keep fixed costs within 70% of the total budget, calculates the daily allowance, and classifies the trip (CRITICAL / BUDGET / COMFORTABLE / LUXURY).
- **Local Guide** — Plans a full day-by-day city itinerary that targets the daily allowance, distinguishing F1 session days from free days.
- **Master Planner** — Synthesizes all reports into a single executive briefing. Uses only data from upstream agents — never fabricates activities or costs.

All agents run on `openai/gpt-4o`. Earlier iterations tested `gpt-4o-mini` on tool-heavy agents for cost savings, but it produced summary-only outputs too frequently (empty accommodation reports, missing ticket data). The reliability cost exceeded the token savings. All tasks run sequentially. Each downstream task receives its upstream tasks as context.

## Tools

| Tool | Source | Purpose |
|---|---|---|
| `GoogleFlightsTool` | SerpApi Google Flights | Live flight prices, schedules, and booking URL |
| `GoogleHotelsPriceTool` | SerpApi Google Hotels | Live nightly rates, total costs, and booking links per tier |
| `CurrencyExchangeTool` | open.er-api.com (no key required) | Live exchange rates for ticket price conversion |
| `SerperDevTool` | Serper.dev | General web search for race dates, session times, attractions |

## Requirements

- Python 3.10–3.13
- [uv](https://docs.astral.sh/uv/) (dependency manager)
- OpenAI API key
- SerpApi API key
- Serper API key

## Setup

**1. Install uv**

```bash
pip install uv
```

**2. Clone and install dependencies**

```bash
git clone <repo-url> && cd f1-planner
uv sync
```

**3. Create a `.env` file** (a template is provided):

```bash
cp .env.example .env
# then fill in your keys
```

```
OPENAI_API_KEY=sk-...
SERPAPI_API_KEY=...
SERPER_API_KEY=...
```

## Configuration

**CLI (recommended)** — pass trip fields as flags; values are validated with the same Pydantic `TripInput` model as the default run:

```bash
uv run f1-plan --help
uv run f1-plan \
  --source-city Hyderabad \
  --destination-city Singapore \
  --grand-prix "2026 Singapore Grand Prix" \
  --days 6 \
  --amount 200000 \
  --currency INR \
  --year 2026
```

Omitted flags use the same defaults as the example above (`--year` defaults to the current calendar year).

**Default run** — `crewai run` / `uv run f1_planner` uses `default_inputs()` in `src/f1_planner/main.py`. Change that dict if you want a different fixed default without using the CLI.

`days` is the total number of calendar days including arrival and departure. Hotel nights are computed automatically (`nights = days - 1`). For example, `days=6` means a 6-day / 5-night trip. The `amount` and `currency_code` are passed through all agents — the budget planner, ticket strategist, and local guide all work in the same currency end-to-end.

## Running

```bash
crewai run          # or: uv run crewai run  — default trip from main.py
uv run f1-plan      # same pipeline, trip from CLI flags (see above)
```

Outputs are written to the `output/` directory as Markdown files. The final `master_planner.md` is the single document to share or act on. Per-run logs are written to `logs/`.

## Project structure

```
f1_planner/
├── src/f1_planner/
│   ├── config/
│   │   ├── agents.yaml        # Agent roles, goals, backstories, LLM
│   │   └── tasks.yaml         # Task descriptions, expected outputs, context chains
│   ├── tools/
│   │   └── tools.py           # GoogleFlightsTool, GoogleHotelsPriceTool, CurrencyExchangeTool
│   ├── cache.py               # Disk cache for SerpApi responses (diskcache, 6 h TTL)
│   ├── crew.py                # Crew assembly — agents, tasks, max_iter settings
│   ├── logging_config.py      # Per-run structured logging with correlation IDs
│   ├── main.py                # Default run + shared crew pipeline
│   ├── cli.py                 # f1-plan — trip inputs from command-line flags
│   ├── schemas.py             # Pydantic TripInput model + post-run output validator
│   └── usage_metrics.py       # Token usage extraction + estimated OpenAI USD cost
├── .env.example               # Template for required API keys
└── pyproject.toml
```

## Reliability features

- **Retry with exponential backoff** — All third-party API calls (SerpApi, exchange-rate API) are wrapped with `tenacity` retry (3 attempts, exponential wait). A single network glitch no longer crashes the entire pipeline.
- **Disk cache** — SerpApi responses are cached to `.cache/serpapi/` with a 6-hour TTL, keyed on search parameters. Repeated dev runs reuse cached data, reducing SerpApi spend by ~90% during iteration.
- **Input validation** — A Pydantic `TripInput` model validates all inputs (currency format, budget range, date format) before any API call is made. Invalid inputs fail fast with a clear error.
- **Output post-processing** — Before validation, `post_process_outputs()` strips leaked CrewAI reasoning prefixes (`Thought:`, `Action:`) and expands single-line markdown tables back into readable multi-line format.
- **Output validation** — After the crew finishes, `validate_outputs()` reads each markdown file and checks for required content signals (prices, tier labels, cost estimates). Missing or suspect outputs are logged as warnings to surface hallucinated or truncated agent output.
- **Auto-retry** — If output validation detects failures, the crew is automatically re-run (max 1 retry). Cached tool results mean the retry only costs LLM tokens, not additional SerpApi calls.
- **Structured logging** — Every run gets a unique 8-character ID. All log events (cache hits, validation results, timing) are tagged with this ID and written to both console and `logs/run_<id>.log`.

### LLM usage and estimated cost

After each crew `kickoff()`, the run logs **prompt / completion / total tokens** from CrewAI’s `token_usage` and prints a short summary before the final decision. An **approximate OpenAI USD** line is computed from default **gpt-4o** per-1M rates (see [OpenAI pricing](https://openai.com/pricing)); refresh the defaults in `usage_metrics.py` when prices change. Override rates without code changes using `OPENAI_GPT4O_INPUT_PER_1M_USD` and `OPENAI_GPT4O_OUTPUT_PER_1M_USD`. When `cached_prompt_tokens` is non-zero, estimates split **non-cached** vs **cached** prompt tokens (cached is a subset of `prompt_tokens`, never added on top). If `OPENAI_GPT4O_CACHED_INPUT_PER_1M_USD` is unset, cached tokens use **50% of your configured input $/1M**, matching OpenAI’s published [GPT-4o prompt caching](https://openai.com/index/api-prompt-caching/) ratio ($1.25 vs $2.50 per 1M). Override with the env var for exact list pricing. Set `F1_PLANNER_PRICING_MODEL` if you standardize on another model name for pricing only.

This estimate covers **OpenAI chat completions only**. **SerpApi, Serper, and other tools are billed separately** and are not included. Some providers may omit token counts; the summary will say so instead of guessing.

## Key design decisions

- **Task splitting**: The original `travel_logistics_task` was split into `flight_research_task` and `accommodation_research_task` because a single task with 7+ tool calls frequently exhausted the agent's iteration budget. Smaller tasks (3–4 tool calls each) complete reliably.
- **Anti-batching instructions**: The local guide agent previously tried to batch multiple search queries into a single action, which CrewAI rejects. Explicit "call tools ONE AT A TIME" instructions prevent this.
- **Top-down budget trial**: The budget planner tries Trackside → Hub → Value Play in order and accepts the first tier where fixed costs stay within 70% of the total budget. This prevents negative daily allowances.
- **Live currency conversion**: The `CurrencyExchangeTool` uses `open.er-api.com` (no API key required) to get live exchange rates. The F1 Experience Strategist is explicitly instructed to use this tool by name rather than searching or guessing.
- **Master planner gating**: The master planner checks whether the local guide output contains "Estimated Day Cost" lines before synthesising the itinerary. If the local guide failed, it states so explicitly rather than fabricating activities.


## Disclaimer
This project is designed to provide a planning baseline for an F1 trip, not final travel advice. Prices, schedules, availability, and regulations can change quickly, so always verify all details with official booking providers and event organizers before making payments or travel decisions.