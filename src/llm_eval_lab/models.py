from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Preference = Literal["A", "B", "tie"]


@dataclass(frozen=True)
class EvaluationResult:
    relevance: float
    instruction_following: float
    safety: float
    clarity: float
    aggregate: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairwiseAnnotation:
    prompt_id: str
    prompt: str
    response_a: str
    response_b: str
    preference: Preference
    rationale: str
    dimensions: tuple[str, ...] = ("helpfulness", "correctness", "clarity", "safety")
    reviewer: str = "reviewer-1"

    def __post_init__(self) -> None:
        if self.preference not in {"A", "B", "tie"}:
            raise ValueError("preference must be 'A', 'B', or 'tie'")
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not self.response_a.strip() or not self.response_b.strip():
            raise ValueError("both candidate responses are required")
        if not self.rationale.strip():
            raise ValueError("annotation rationale is required")


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    prompt: str
    required_terms: tuple[str, ...] = field(default_factory=tuple)
    forbidden_terms: tuple[str, ...] = field(default_factory=tuple)
    max_words: int | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.prompt.strip():
            raise ValueError("case_id and prompt are required")
        if self.max_words is not None and self.max_words <= 0:
            raise ValueError("max_words must be positive")
