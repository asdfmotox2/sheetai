"""Core processing engine: reads spreadsheets, calls LLMs, writes results."""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .config import ProcessingConfig


class SheetProcessor:
    """Process spreadsheet rows through an LLM API."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        api_key = self.config.api_key
        if self.config.provider == "anthropic":
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No API key: set ANTHROPIC_API_KEY or pass api_key in config")
            try:
                import anthropic
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        elif self.config.provider == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key: set OPENAI_API_KEY or pass api_key in config")
            try:
                import openai
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def read_input(self, path: str) -> list[dict[str, str]]:
        """Read CSV or Excel file into list of row dicts."""
        p = Path(path)
        if p.suffix in (".xlsx", ".xlsm", ".xls"):
            return self._read_excel(p)
        elif p.suffix in (".csv", ".tsv"):
            return self._read_csv(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)

    def _read_excel(self, path: Path) -> list[dict[str, str]]:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(next(rows_iter))]
        result = []
        for row in rows_iter:
            d = {}
            for h, v in zip(headers, row):
                d[h] = str(v) if v is not None else ""
            result.append(d)
        wb.close()
        return result

    def format_prompt(self, row: dict[str, str]) -> str:
        """Substitute row values into the prompt template."""
        prompt = self.config.prompt_template
        for key, value in row.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def call_llm(self, prompt: str) -> str:
        """Call the configured LLM and return the response text."""
        if self.config.provider == "anthropic":
            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.config.system_prompt:
                kwargs["system"] = self.config.system_prompt
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        elif self.config.provider == "openai":
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        raise ValueError(f"Unknown provider: {self.config.provider}")

    def parse_response(self, text: str) -> dict[str, str]:
        """Parse LLM response into output column values.
        
        Tries JSON first, then falls back to mapping the full text
        to the first output column.
        """
        out_cols = self.config.output_columns
        # Try JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                result = {}
                for col in out_cols:
                    result[col] = str(data.get(col, ""))
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: entire response goes to first output column
        result = {col: "" for col in out_cols}
        result[out_cols[0]] = text.strip()
        return result

    def process(self, input_path: str, output_path: Optional[str] = None) -> list[dict[str, str]]:
        """Process all rows and return enriched data.
        
        Args:
            input_path: Path to CSV/Excel input file.
            output_path: Optional path to write CSV output. If None, returns data only.
            
        Returns:
            List of row dicts with output columns added.
        """
        rows = self.read_input(input_path)
        if not rows:
            return []

        total = len(rows)
        if self.config.dry_run:
            rows = rows[:3]
            total = len(rows)

        out_cols = self.config.output_columns
        results = []
        processed = 0
        skipped = 0

        for i, row in enumerate(rows):
            # Check if already filled
            if self.config.skip_filled and all(row.get(c) for c in out_cols):
                results.append(row)
                skipped += 1
                continue

            prompt = self.format_prompt(row)

            # Retry loop
            last_error = None
            for attempt in range(self.config.max_retries):
                try:
                    response_text = self.call_llm(prompt)
                    parsed = self.parse_response(response_text)
                    row.update(parsed)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)
                        time.sleep(delay)

            if last_error:
                # Mark row with error
                for col in out_cols:
                    row[col] = f"ERROR: {last_error}"

            results.append(row)
            processed += 1

            # Progress
            pct = (i + 1) / total * 100
            print(f"\r  [{i+1}/{total}] {pct:.0f}%", end="", flush=True, file=sys.stderr)

            # Batch save
            if output_path and processed % self.config.batch_size == 0:
                self._write_csv(results, output_path)

        print(file=sys.stderr)  # newline after progress

        if output_path:
            self._write_csv(results, output_path)
            print(f"  Wrote {len(results)} rows to {output_path} ({skipped} skipped, {processed} processed)", file=sys.stderr)

        return results

    def _write_csv(self, rows: list[dict], path: str):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        # Ensure output columns are included
        for col in self.config.output_columns:
            if col not in fieldnames:
                fieldnames.append(col)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
