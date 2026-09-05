from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import PromptCase
from .preferences import (
    annotation_quality_report,
    disagreement_by_prompt,
    export_preference_records,
    load_jsonl,
    preference_summary,
)
from .prompt_suite import check_prompt_case
from .rubric import evaluate_response


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    preferred_response: str
    rejected_response: str
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    max_words: int | None = None

    def __post_init__(self) -> None:
        required = {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "preferred_response": self.preferred_response,
            "rejected_response": self.rejected_response,
        }
        for name, value in required.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.max_words is not None and self.max_words <= 0:
            raise ValueError("max_words must be positive")


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                cases.append(
                    BenchmarkCase(
                        case_id=payload["case_id"],
                        prompt=payload["prompt"],
                        preferred_response=payload["preferred_response"],
                        rejected_response=payload["rejected_response"],
                        required_terms=tuple(payload.get("required_terms", ())),
                        forbidden_terms=tuple(payload.get("forbidden_terms", ())),
                        max_words=payload.get("max_words"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid benchmark case on line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("benchmark suite must contain at least one case")
    return cases


def run_benchmark(
    cases_path: str | Path,
    preferences_path: str | Path,
) -> dict[str, object]:
    cases = load_benchmark_cases(cases_path)
    results: list[dict[str, object]] = []

    for case in cases:
        preferred = evaluate_response(
            case.prompt,
            case.preferred_response,
            required_terms=case.required_terms,
            max_words=case.max_words,
        )
        rejected = evaluate_response(
            case.prompt,
            case.rejected_response,
            required_terms=case.required_terms,
            max_words=case.max_words,
        )
        prompt_check = check_prompt_case(
            PromptCase(
                case_id=case.case_id,
                prompt=case.prompt,
                required_terms=case.required_terms,
                forbidden_terms=case.forbidden_terms,
                max_words=case.max_words,
            ),
            case.preferred_response,
        )
        results.append(
            {
                "case_id": case.case_id,
                "rubric_preference_correct": preferred.aggregate > rejected.aggregate,
                "preferred_aggregate": preferred.aggregate,
                "rejected_aggregate": rejected.aggregate,
                "aggregate_margin": round(preferred.aggregate - rejected.aggregate, 4),
                "prompt_check_passed": prompt_check.passed,
                "prompt_check_failures": list(prompt_check.failures),
            }
        )

    annotations = load_jsonl(str(preferences_path))
    exported = export_preference_records(annotations)
    exported_with_meta = export_preference_records(annotations, include_metadata=True)
    disagreements = disagreement_by_prompt(annotations)
    quality = annotation_quality_report(annotations)
    non_ties = sum(1 for item in annotations if item.preference != "tie")
    rubric_wins = sum(bool(item["rubric_preference_correct"]) for item in results)
    prompt_passes = sum(bool(item["prompt_check_passed"]) for item in results)
    disagreement_count = sum(disagreements.values())
    case_count = len(results)
    quality_pass = int(quality["flagged"]) == 0
    passed = (
        rubric_wins == case_count
        and prompt_passes == case_count
        and len(exported) == non_ties
        and disagreement_count == 0
        and quality_pass
    )

    return {
        "schema_version": "1.1",
        "passed": passed,
        "summary": {
            "evaluation_cases": case_count,
            "rubric_preference_wins": rubric_wins,
            "rubric_preference_win_rate": round(rubric_wins / case_count, 4),
            "prompt_checks_passed": prompt_passes,
            "prompt_check_pass_rate": round(prompt_passes / case_count, 4),
            "preference_annotations": len(annotations),
            "exported_preference_records": len(exported),
            "non_tie_annotations": non_ties,
            "annotation_quality_passing": int(quality["passing"]),
            "reviewer_disagreement_prompts": disagreement_count,
        },
        "preference_summary": preference_summary(annotations),
        "annotation_quality": {
            "pass_rate": quality["pass_rate"],
            "dimension_frequency": quality["dimension_frequency"],
            "flagged": quality["flagged"],
        },
        "results": results,
        "sample_export_with_metadata": exported_with_meta[:1],
        "limitations": (
            f"This deterministic benchmark uses {case_count} authored regression cases and "
            f"{len(annotations)} sample preference annotations. It validates the evaluation "
            "and labeling pipeline (rubric ranking, prompt checks, export shape, rationale "
            "quality gates), not production model quality, large-scale inter-annotator "
            "agreement, or customer ROI."
        ),
    }


def write_report(report: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
