"""Data contracts at system boundaries: input validation and output verification."""

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation — runs BEFORE any API budget is spent
# ---------------------------------------------------------------------------

class TripInput(BaseModel):
    source_city: str = Field(min_length=1)
    destination_city: str = Field(min_length=1)
    grand_prix: str = Field(min_length=1)
    days: int = Field(ge=2, le=21)
    amount: str
    currency_code: str
    current_year: str

    @property
    def nights(self) -> int:
        """Hotel nights = calendar days minus 1 (arrive Day 1, depart Day N)."""
        return self.days - 1

    @field_validator("currency_code")
    @classmethod
    def must_be_iso4217(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Z]{3}", v.upper()):
            raise ValueError(
                f"currency_code must be a 3-letter ISO 4217 code, got '{v}'"
            )
        return v.upper()

    @field_validator("amount")
    @classmethod
    def must_be_positive_numeric(cls, v: str) -> str:
        if not v.isdigit() or int(v) <= 0:
            raise ValueError(f"amount must be a positive integer string, got '{v}'")
        return v

    @field_validator("current_year")
    @classmethod
    def must_be_four_digit_year(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}", v):
            raise ValueError(f"current_year must be a 4-digit year, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Output validation — runs AFTER the crew finishes, reads markdown files
# ---------------------------------------------------------------------------

_OUTPUT_CHECKS: dict[str, list[tuple[str, str]]] = {
    "flight_research.md": [
        (r"\d[\d,]*", "must contain at least one numeric price"),
        (r"(Flight|Option|→)", "must contain flight routing data"),
    ],
    "accommodation.md": [
        (r"(?i)trackside|hub|value play", "must mention at least one hotel tier"),
        (r"\d[\d,]*", "must contain at least one numeric price"),
    ],
    "race_ticket.md": [
        (r"(?i)grandstand|walkabout|zone", "must mention a grandstand or zone"),
        (r"\d[\d,]*", "must contain at least one numeric price"),
    ],
    "budget_plan.md": [
        (
            r"(?i)(CRITICAL|BUDGET|COMFORTABLE|LUXURY)",
            "must contain a budget classification label",
        ),
        (r"(?i)daily allowance", "must contain a daily allowance figure"),
    ],
    "local_guide.md": [
        (r"(?i)estimated day cost", "must contain per-day cost estimates"),
    ],
    "master_planner.md": [
        (r"(?i)grand total|total trip cost", "must contain a total trip cost"),
    ],
}


_AGENT_REASONING_RE = re.compile(
    r"^(Thought:\s.*?\n|Action:\s.*?\n|Action Input:\s.*?\n|Observation:\s.*?\n)+",
    re.MULTILINE,
)

_MARKDOWN_START_RE = re.compile(r"^(#{1,6}\s|---|\*\*|\|)", re.MULTILINE)

_FLAT_TABLE_RE = re.compile(
    r"\*{0,2}\|[^\n]{80,}\|\*{0,2}"
)


def _strip_preamble(content: str) -> str:
    """Remove conversational filler before the first markdown element.

    Agents sometimes prepend lines like "I now have all the necessary
    information..." before the actual report.  This finds the first real
    markdown anchor (heading, bold text, HR, or table row) and drops
    everything before it.
    """
    m = _MARKDOWN_START_RE.search(content)
    if m and m.start() > 0:
        return content[m.start():]
    return content


def post_process_outputs(output_dir: str = "output") -> int:
    """Clean up common agent output artefacts in-place.

    1. Strip leaked CrewAI reasoning prefixes (Thought:/Action:/Observation:).
    2. Strip conversational preamble before the first markdown element.
    3. Expand single-line markdown tables into properly formatted multi-line tables.

    Returns the number of files modified.
    """
    base = Path(output_dir)
    modified = 0

    for md_file in base.glob("*.md"):
        original = md_file.read_text(encoding="utf-8")
        content = original

        content = _AGENT_REASONING_RE.sub("", content).lstrip("\n")

        content = _strip_preamble(content)

        if _FLAT_TABLE_RE.search(content):
            content = _expand_flat_tables(content)

        if content != original:
            md_file.write_text(content, encoding="utf-8")
            logger.info("Post-processed: %s", md_file.name)
            modified += 1

    logger.info("Post-processing complete: %d file(s) modified", modified)
    return modified


def _expand_flat_tables(text: str) -> str:
    """Turn single-line pipe-delimited tables back into multi-line markdown."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip().strip("*")
        if stripped.startswith("|") and stripped.count("|") > 6:
            cells = [c.strip() for c in stripped.split("|")]
            rows: list[list[str]] = []
            current_row: list[str] = []
            for cell in cells:
                if cell == "":
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                else:
                    current_row.append(cell)
            if current_row:
                rows.append(current_row)

            if len(rows) >= 2:
                ncols = len(rows[0])
                out.append("| " + " | ".join(rows[0]) + " |")
                out.append("| " + " | ".join(["---"] * ncols) + " |")
                for row in rows[1:]:
                    if all(c.strip() == "---" for c in row):
                        continue
                    padded = row + [""] * (ncols - len(row))
                    out.append("| " + " | ".join(padded[:ncols]) + " |")
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def validate_outputs(output_dir: str = "output") -> dict[str, list[str]]:
    """Read each expected output file and check for required content signals.

    Returns a dict of {filename: [warning_messages]}.  An empty list means
    the file passed all checks.  A missing file produces a single warning.
    """
    results: dict[str, list[str]] = {}
    base = Path(output_dir)

    for filename, checks in _OUTPUT_CHECKS.items():
        warnings: list[str] = []
        filepath = base / filename

        if not filepath.exists():
            warnings.append(f"file not found: {filepath}")
            results[filename] = warnings
            continue

        content = filepath.read_text(encoding="utf-8")

        if len(content.strip()) < 20:
            warnings.append(f"file is nearly empty ({len(content)} chars)")

        for pattern, message in checks:
            if not re.search(pattern, content):
                warnings.append(message)

        results[filename] = warnings

    passed = sum(1 for w in results.values() if not w)
    total = len(results)
    logger.info(
        "Output validation: %d/%d files passed all checks", passed, total
    )
    for fname, warns in results.items():
        if warns:
            for w in warns:
                logger.warning("  %s: %s", fname, w)
        else:
            logger.info("  %s: OK", fname)

    return results
