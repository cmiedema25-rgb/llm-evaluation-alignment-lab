from llm_eval_lab.models import PairwiseAnnotation
from llm_eval_lab.preferences import disagreement_by_prompt, export_preference_records, preference_summary


def sample(preference: str, reviewer: str = "r1") -> PairwiseAnnotation:
    return PairwiseAnnotation(
        prompt_id="p1",
        prompt="Explain testing",
        response_a="A detailed explanation",
        response_b="A short answer",
        preference=preference,
        rationale="Comparative quality judgment",
        reviewer=reviewer,
    )


def test_preference_summary() -> None:
    summary = preference_summary([sample("A"), sample("B"), sample("A")])
    assert summary["total"] == 3
    assert summary["a_win_rate"] == 0.6667


def test_export_uses_chosen_and_rejected() -> None:
    records = export_preference_records([sample("B")])
    assert records[0]["chosen"] == "A short answer"
    assert records[0]["rejected"] == "A detailed explanation"


def test_disagreement_detection() -> None:
    disagreement = disagreement_by_prompt([sample("A", "r1"), sample("B", "r2")])
    assert disagreement["p1"] is True
