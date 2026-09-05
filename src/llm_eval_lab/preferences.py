from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import WRITING_QUALITY_DIMENSIONS, PairwiseAnnotation

_MIN_RATIONALE_CHARS = 40


def preference_summary(annotations: Iterable[PairwiseAnnotation]) -> dict[str, float | int]:
    items = list(annotations)
    counts = Counter(item.preference for item in items)
    total = len(items)
    if total == 0:
        return {"total": 0, "a_win_rate": 0.0, "b_win_rate": 0.0, "tie_rate": 0.0}
    return {
        "total": total,
        "a_win_rate": round(counts["A"] / total, 4),
        "b_win_rate": round(counts["B"] / total, 4),
        "tie_rate": round(counts["tie"] / total, 4),
    }


def disagreement_by_prompt(annotations: Iterable[PairwiseAnnotation]) -> dict[str, bool]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in annotations:
        grouped[item.prompt_id].add(item.preference)
    return {prompt_id: len(labels) > 1 for prompt_id, labels in grouped.items()}


def dimension_frequency(annotations: Iterable[PairwiseAnnotation]) -> dict[str, int]:
    """Count how often each labeling dimension appears across annotations."""
    counts: Counter[str] = Counter()
    for item in annotations:
        counts.update(item.dimensions)
    return dict(sorted(counts.items()))


def rationale_quality_issues(annotation: PairwiseAnnotation) -> tuple[str, ...]:
    """Flag thin rationales that would fail a writing-quality review pass.

    Checks for evidence-based labeling habits: enough detail, explicit comparison,
    and at least one concrete cue from the preferred (or both) responses.
    """
    issues: list[str] = []
    rationale = annotation.rationale.strip()
    lowered = rationale.lower()

    if len(rationale) < _MIN_RATIONALE_CHARS:
        issues.append(f"rationale shorter than {_MIN_RATIONALE_CHARS} characters")

    comparative_markers = (
        "prefer",
        "better",
        "worse",
        "more",
        "less",
        "while",
        "whereas",
        "unlike",
        "compared",
        "both",
        "tie",
        "equivalent",
    )
    if not any(marker in lowered for marker in comparative_markers):
        issues.append("rationale lacks an explicit comparative judgment")

    # Evidence cue: quote marks, or a distinctive fragment from either response.
    has_quote = '"' in rationale or "'" in rationale or "“" in rationale
    a_tokens = {tok.lower() for tok in annotation.response_a.split() if len(tok) > 4}
    b_tokens = {tok.lower() for tok in annotation.response_b.split() if len(tok) > 4}
    rationale_tokens = {tok.lower().strip(".,;:()") for tok in rationale.split()}
    cites_response = bool((a_tokens | b_tokens) & rationale_tokens)
    if not has_quote and not cites_response:
        issues.append("rationale does not cite evidence from either response")

    if annotation.preference == "tie" and "tie" not in lowered and "equivalent" not in lowered:
        issues.append("tie label without explaining why the responses are equivalent")

    return tuple(issues)


def annotation_quality_report(
    annotations: Iterable[PairwiseAnnotation],
) -> dict[str, object]:
    items = list(annotations)
    flagged = [
        {"prompt_id": item.prompt_id, "reviewer": item.reviewer, "issues": list(issues)}
        for item in items
        if (issues := rationale_quality_issues(item))
    ]
    return {
        "total": len(items),
        "passing": len(items) - len(flagged),
        "flagged": len(flagged),
        "pass_rate": round((len(items) - len(flagged)) / len(items), 4) if items else 0.0,
        "known_writing_dimensions": list(WRITING_QUALITY_DIMENSIONS),
        "dimension_frequency": dimension_frequency(items),
        "flagged_annotations": flagged,
    }


def export_preference_records(
    annotations: Iterable[PairwiseAnnotation],
    *,
    include_metadata: bool = False,
) -> list[dict[str, object]]:
    """Convert non-tie pairwise labels into chosen/rejected preference records.

    Default export is the clean DPO-style triple (prompt / chosen / rejected).
    With include_metadata=True, also attaches rationale, dimensions, reviewer,
    and confidence for audit or rater-calibration workflows.
    """
    records: list[dict[str, object]] = []
    for item in annotations:
        if item.preference == "tie":
            continue
        chosen = item.response_a if item.preference == "A" else item.response_b
        rejected = item.response_b if item.preference == "A" else item.response_a
        record: dict[str, object] = {
            "prompt": item.prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_id": item.prompt_id,
        }
        if include_metadata:
            record.update(
                {
                    "rationale": item.rationale,
                    "dimensions": list(item.dimensions),
                    "reviewer": item.reviewer,
                    "confidence": item.confidence,
                    "preference": item.preference,
                }
            )
        records.append(record)
    return records


def load_jsonl(path: str) -> list[PairwiseAnnotation]:
    items: list[PairwiseAnnotation] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                items.append(
                    PairwiseAnnotation(
                        prompt_id=data["prompt_id"],
                        prompt=data["prompt"],
                        response_a=data["response_a"],
                        response_b=data["response_b"],
                        preference=data["preference"],
                        rationale=data["rationale"],
                        dimensions=tuple(
                            data.get(
                                "dimensions",
                                ("helpfulness", "correctness", "clarity", "safety"),
                            )
                        ),
                        reviewer=data.get("reviewer", "reviewer-1"),
                        confidence=float(data.get("confidence", 1.0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid annotation on line {line_number}: {exc}") from exc
    return items
