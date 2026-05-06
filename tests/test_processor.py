"""Tests for sheetai processor."""

import csv
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for testing without install
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sheetai.config import ProcessingConfig
from sheetai.processor import SheetProcessor


@pytest.fixture
def basic_config():
    return ProcessingConfig(
        prompt_template="Classify: {name} - {description}",
        output_columns=["category"],
        api_key="test-key",
    )


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "test.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "description"])
        writer.writeheader()
        writer.writerow({"name": "Widget A", "description": "A small gadget"})
        writer.writerow({"name": "Gizmo B", "description": "A large tool"})
        writer.writerow({"name": "Doohickey", "description": "Unknown purpose"})
    return str(path)


@pytest.fixture
def tsv_file(tmp_path):
    path = tmp_path / "test.tsv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"name": "X", "value": "100"})
    return str(path)


class TestReadInput:
    def test_read_csv(self, basic_config, csv_file):
        proc = SheetProcessor(basic_config)
        rows = proc.read_input(csv_file)
        assert len(rows) == 3
        assert rows[0]["name"] == "Widget A"
        assert rows[1]["description"] == "A large tool"

    def test_read_tsv(self, basic_config, tsv_file):
        proc = SheetProcessor(basic_config)
        rows = proc.read_input(tsv_file)
        assert len(rows) == 1
        assert rows[0]["name"] == "X"
        assert rows[0]["value"] == "100"

    def test_read_xlsx(self, basic_config, tmp_path):
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")
        path = tmp_path / "test.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["name", "description"])
        ws.append(["Widget", "A thing"])
        wb.save(path)
        proc = SheetProcessor(basic_config)
        rows = proc.read_input(str(path))
        assert len(rows) == 1
        assert rows[0]["name"] == "Widget"

    def test_unsupported_format(self, basic_config):
        proc = SheetProcessor(basic_config)
        with pytest.raises(ValueError, match="Unsupported"):
            proc.read_input("data.json")


class TestFormatPrompt:
    def test_basic_substitution(self, basic_config):
        proc = SheetProcessor(basic_config)
        result = proc.format_prompt({"name": "Foo", "description": "Bar"})
        assert result == "Classify: Foo - Bar"

    def test_missing_placeholder(self, basic_config):
        proc = SheetProcessor(basic_config)
        result = proc.format_prompt({"name": "Foo"})
        assert "{description}" in result

    def test_extra_columns_ignored(self, basic_config):
        proc = SheetProcessor(basic_config)
        result = proc.format_prompt({"name": "Foo", "description": "Bar", "extra": "Baz"})
        assert result == "Classify: Foo - Bar"


class TestParseResponse:
    def test_json_response(self, basic_config):
        basic_config.output_columns = ["category", "confidence"]
        proc = SheetProcessor(basic_config)
        result = proc.parse_response('{"category": "Electronics", "confidence": "high"}')
        assert result["category"] == "Electronics"
        assert result["confidence"] == "high"

    def test_plain_text_fallback(self, basic_config):
        proc = SheetProcessor(basic_config)
        result = proc.parse_response("This is electronics")
        assert result["category"] == "This is electronics"

    def test_json_missing_keys(self, basic_config):
        basic_config.output_columns = ["category", "confidence"]
        proc = SheetProcessor(basic_config)
        result = proc.parse_response('{"category": "Food"}')
        assert result["category"] == "Food"
        assert result["confidence"] == ""

    def test_invalid_json_fallback(self, basic_config):
        proc = SheetProcessor(basic_config)
        result = proc.parse_response("{broken json")
        assert result["category"] == "{broken json"


class TestProcess:
    @patch.object(SheetProcessor, "call_llm")
    @patch.object(SheetProcessor, "_build_client")
    def test_basic_process(self, mock_client, mock_llm, basic_config, csv_file, tmp_path):
        mock_llm.return_value = '{"category": "Gadget"}'
        proc = SheetProcessor(basic_config)
        output = str(tmp_path / "out.csv")
        results = proc.process(csv_file, output)
        assert len(results) == 3
        assert all(r["category"] == "Gadget" for r in results)
        assert mock_llm.call_count == 3
        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["category"] == "Gadget"

    @patch.object(SheetProcessor, "call_llm")
    @patch.object(SheetProcessor, "_build_client")
    def test_dry_run(self, mock_client, mock_llm, basic_config, csv_file):
        basic_config.dry_run = True
        mock_llm.return_value = "result"
        proc = SheetProcessor(basic_config)
        results = proc.process(csv_file)
        assert len(results) == 3
        assert mock_llm.call_count == 3

    @patch.object(SheetProcessor, "call_llm")
    @patch.object(SheetProcessor, "_build_client")
    def test_skip_filled(self, mock_client, mock_llm, basic_config, tmp_path):
        path = tmp_path / "prefilled.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "description", "category"])
            writer.writeheader()
            writer.writerow({"name": "A", "description": "a", "category": "DONE"})
            writer.writerow({"name": "B", "description": "b", "category": ""})
        mock_llm.return_value = "new"
        proc = SheetProcessor(basic_config)
        results = proc.process(str(path))
        assert mock_llm.call_count == 1
        assert results[0]["category"] == "DONE"
        assert results[1]["category"] == "new"

    @patch.object(SheetProcessor, "call_llm")
    @patch.object(SheetProcessor, "_build_client")
    def test_retry_on_error(self, mock_client, mock_llm, basic_config, csv_file):
        basic_config.retry_delay = 0.01
        basic_config.max_retries = 3
        basic_config.dry_run = True
        mock_llm.side_effect = [Exception("fail"), Exception("fail"), "success",
                                 "ok", "ok", "ok"]  # extra for remaining rows
        proc = SheetProcessor(basic_config)
        results = proc.process(csv_file)
        assert mock_llm.call_count >= 3

    @patch.object(SheetProcessor, "call_llm")
    @patch.object(SheetProcessor, "_build_client")
    def test_all_retries_fail(self, mock_client, mock_llm, basic_config, tmp_path):
        basic_config.retry_delay = 0.01
        basic_config.max_retries = 2
        path = tmp_path / "one.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "description"])
            writer.writeheader()
            writer.writerow({"name": "X", "description": "Y"})
        mock_llm.side_effect = Exception("permanent fail")
        proc = SheetProcessor(basic_config)
        results = proc.process(str(path))
        assert "ERROR" in results[0]["category"]

    @patch.object(SheetProcessor, "call_llm")
    @patch.object(SheetProcessor, "_build_client")
    def test_empty_input(self, mock_client, mock_llm, basic_config, tmp_path):
        path = tmp_path / "empty.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name"])
            writer.writeheader()
        proc = SheetProcessor(basic_config)
        results = proc.process(str(path))
        assert results == []
        mock_llm.assert_not_called()


class TestConfig:
    def test_defaults(self):
        cfg = ProcessingConfig(prompt_template="test")
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.provider == "anthropic"
        assert cfg.batch_size == 10
        assert cfg.dry_run is False
        assert cfg.skip_filled is True
        assert cfg.temperature == 0.0

    def test_custom_values(self):
        cfg = ProcessingConfig(
            prompt_template="t",
            output_columns=["a", "b"],
            model="gpt-4o",
            provider="openai",
            batch_size=5,
        )
        assert cfg.output_columns == ["a", "b"]
        assert cfg.model == "gpt-4o"
        assert cfg.provider == "openai"
