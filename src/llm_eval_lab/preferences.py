from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import PairwiseAnnotation


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


def export_preference_records(annotations: Iterable[PairwiseAnnotation]) -> list[dict[str, str]]:
    """Convert non-tie pairwise labels into chosen/rejected preference records."""
    records: list[dict[str, str]] = []
    for item in annotations:
        if item.preference == "tie":
            continue
        chosen = item.response_a if item.preference == "A" else item.response_b
        rejected = item.response_b if item.preference == "A" else item.response_a
        records.append(
            {
                "prompt": item.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "prompt_id": item.prompt_id,
            }
        )
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
                        dimensions=tuple(data.get("dimensions", ("helpfulness", "correctness"))),
                        reviewer=data.get("reviewer", "reviewer-1"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid annotation on line {line_number}: {exc}") from exc
    return items
