from __future__ import annotations

from dataclasses import dataclass

from .models import PromptCase


@dataclass(frozen=True)
class PromptCheckResult:
    case_id: str
    passed: bool
    failures: tuple[str, ...]


def check_prompt_case(case: PromptCase, response: str) -> PromptCheckResult:
    lowered = response.lower()
    failures: list[str] = []

    for term in case.required_terms:
        if term.lower() not in lowered:
            failures.append(f"missing required term: {term}")

    for term in case.forbidden_terms:
        if term.lower() in lowered:
            failures.append(f"contains forbidden term: {term}")

    if case.max_words is not None:
        word_count = len(response.split())
        if word_count > case.max_words:
            failures.append(f"word count {word_count} exceeds limit {case.max_words}")

    return PromptCheckResult(
        case_id=case.case_id,
        passed=not failures,
        failures=tuple(failures),
    )
