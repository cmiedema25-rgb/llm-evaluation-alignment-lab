from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_lab.benchmark import load_benchmark_cases, run_benchmark
from llm_eval_lab.cli import main


def test_retained_benchmark_produces_expected_summary() -> None:
    report = run_benchmark(
        "data/evaluation_cases.jsonl",
        "data/sample_preferences.jsonl",
    )

    assert report["passed"] is True
    assert report["schema_version"] == "1.1"
    assert report["summary"] == {
        "evaluation_cases": 14,
        "rubric_preference_wins": 14,
        "rubric_preference_win_rate": 1.0,
        "prompt_checks_passed": 14,
        "prompt_check_pass_rate": 1.0,
        "preference_annotations": 8,
        "exported_preference_records": 7,
        "non_tie_annotations": 7,
        "annotation_quality_passing": 8,
        "reviewer_disagreement_prompts": 0,
    }
    assert report["preference_summary"]["tie_rate"] == 0.125
    assert report["annotation_quality"]["flagged"] == 0


def test_benchmark_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "benchmark",
            "data/evaluation_cases.jsonl",
            "--preferences",
            "data/sample_preferences.jsonl",
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_export_and_quality_cli(tmp_path: Path) -> None:
    out = tmp_path / "prefs.jsonl"
    assert main(["export-preferences", "data/sample_preferences.jsonl", "--output", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 7
    assert "chosen" in json.loads(lines[0])
    assert main(["check-annotations", "data/sample_preferences.jsonl"]) == 0


def test_benchmark_loader_reports_invalid_line(tmp_path: Path) -> None:
    suite = tmp_path / "invalid.jsonl"
    suite.write_text('{"case_id": "missing-fields"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_benchmark_cases(suite)
