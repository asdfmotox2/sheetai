"""CLI entry point for sheetai."""

import argparse
import json
import sys
from pathlib import Path

from .config import ProcessingConfig
from .processor import SheetProcessor


def main():
    parser = argparse.ArgumentParser(
        prog="sheetai",
        description="Process spreadsheet rows through LLM APIs. Feed CSV/Excel, get AI-enriched output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Classify products using Claude
  sheetai products.csv -o enriched.csv \\
    -p "Classify this product into a category: {name} - {description}" \\
    --output-columns category confidence

  # Enrich leads with company research
  sheetai leads.xlsx -o leads_enriched.csv \\
    -p "Research this company and provide a one-line summary: {company_name}" \\
    --output-columns summary

  # Use a config file for complex prompts
  sheetai data.csv -o out.csv --config my_config.json

  # Dry run (process first 3 rows only)
  sheetai data.csv -o preview.csv -p "{text}" --dry-run
""",
    )
    parser.add_argument("input", help="Input CSV or Excel file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: stdout as CSV)")
    parser.add_argument("-p", "--prompt", help="Prompt template with {column_name} placeholders")
    parser.add_argument("--output-columns", nargs="+", default=["ai_result"],
                        help="Output column names (default: ai_result)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LLM model name")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--api-key", help="API key (default: from env var)")
    parser.add_argument("--system-prompt", help="System prompt for the LLM")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--batch-size", type=int, default=10, help="Rows between saves")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per row")
    parser.add_argument("--dry-run", action="store_true", help="Process only first 3 rows")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-process rows even if output columns are filled")
    parser.add_argument("--config", help="JSON config file (overrides CLI args)")
    parser.add_argument("--preview", action="store_true",
                        help="Show first row's formatted prompt and exit")

    args = parser.parse_args()

    # Build config
    if args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        config = ProcessingConfig(**cfg_data)
    else:
        if not args.prompt:
            parser.error("--prompt is required (or use --config)")
        config = ProcessingConfig(
            prompt_template=args.prompt,
            output_columns=args.output_columns,
            model=args.model,
            provider=args.provider,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            dry_run=args.dry_run,
            skip_filled=not args.no_skip,
            system_prompt=args.system_prompt,
        )

    processor = SheetProcessor(config)

    # Preview mode
    if args.preview:
        rows = processor.read_input(args.input)
        if rows:
            prompt = processor.format_prompt(rows[0])
            print(f"Columns: {list(rows[0].keys())}")
            print(f"Row count: {len(rows)}")
            print(f"\n--- Formatted prompt for row 1 ---")
            print(prompt)
        else:
            print("No rows found in input file.")
        return

    # Process
    print(f"Processing {args.input}...", file=sys.stderr)
    results = processor.process(args.input, args.output)

    # If no output file, write to stdout
    if not args.output and results:
        import csv
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
