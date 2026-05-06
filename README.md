# sheetai

Process spreadsheet rows through LLM APIs. Feed a CSV or Excel file, get AI-enriched output with new columns.

## Install

```bash
pip install sheetai
```

## Quick start

```bash
# Classify products using Claude
export ANTHROPIC_API_KEY=sk-ant-...
sheetai products.csv -o enriched.csv \
  -p "Classify this product into one of: Electronics, Clothing, Food, Other. Product: {name} - {description}. Reply with JSON: {\"category\": \"...\", \"confidence\": \"high/medium/low\"}" \
  --output-columns category confidence

# Enrich leads
sheetai leads.xlsx -o enriched.csv \
  -p "Write a one-line summary of what {company_name} does based on: {website}" \
  --output-columns summary

# Preview what the prompt looks like before spending API credits
sheetai data.csv -p "Analyze: {text}" --preview

# Dry run (first 3 rows only)
sheetai data.csv -o test.csv -p "{review_text}" --dry-run
```

## Python API

```python
from sheetai import SheetProcessor, ProcessingConfig

config = ProcessingConfig(
    prompt_template="Classify this review as positive/negative/neutral: {review_text}",
    output_columns=["sentiment", "confidence"],
    model="claude-sonnet-4-20250514",
    system_prompt="Reply with JSON only: {\"sentiment\": \"...\", \"confidence\": \"...\"}",
)

processor = SheetProcessor(config)
results = processor.process("reviews.csv", "reviews_enriched.csv")
```

## Features

- **CSV and Excel** input (.csv, .tsv, .xlsx)
- **Claude and OpenAI** providers (`--provider anthropic|openai`)
- **Structured output** — JSON responses auto-parsed into separate columns
- **Resume-safe** — skips rows where output columns already have values
- **Batch saves** — writes progress every N rows (no lost work on interruption)
- **Retry with backoff** — automatic retry on API errors
- **Dry run** — test with first 3 rows before committing API spend
- **Preview mode** — see formatted prompts without calling the API

## Config file

For complex prompts, use a JSON config:

```json
{
  "prompt_template": "You are analyzing customer feedback.\n\nReview: {review_text}\nProduct: {product_name}\n\nClassify sentiment and extract key themes.",
  "output_columns": ["sentiment", "themes", "action_items"],
  "model": "claude-sonnet-4-20250514",
  "system_prompt": "Reply with JSON: {\"sentiment\": \"positive|negative|neutral\", \"themes\": [\"...\"], \"action_items\": [\"...\"]}",
  "temperature": 0.0,
  "max_tokens": 512
}
```

```bash
sheetai reviews.csv -o enriched.csv --config config.json
```

## License

MIT
