"""Configuration for sheet processing."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProcessingConfig:
    """Configuration for LLM-powered spreadsheet processing."""

    prompt_template: str
    """Prompt template with {column_name} placeholders for row data."""

    output_columns: list[str] = field(default_factory=lambda: ["ai_result"])
    """Column names for structured output fields."""

    model: str = "claude-sonnet-4-20250514"
    """LLM model to use."""

    provider: str = "anthropic"
    """LLM provider: 'anthropic' or 'openai'."""

    api_key: Optional[str] = None
    """API key. Falls back to ANTHROPIC_API_KEY or OPENAI_API_KEY env vars."""

    max_tokens: int = 1024
    """Max tokens per response."""

    batch_size: int = 10
    """Number of rows to process before saving progress."""

    max_retries: int = 3
    """Max retries per row on API failure."""

    retry_delay: float = 2.0
    """Base delay between retries (exponential backoff)."""

    dry_run: bool = False
    """If True, process only first 3 rows."""

    skip_filled: bool = True
    """Skip rows where output columns already have values."""

    temperature: float = 0.0
    """LLM temperature."""

    system_prompt: Optional[str] = None
    """Optional system prompt prepended to each request."""
