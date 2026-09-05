from llm_eval_lab.models import PairwiseAnnotation
from llm_eval_lab.preferences import (
    annotation_quality_report,
    disagreement_by_prompt,
    export_preference_records,
    preference_summary,
    rationale_quality_issues,
)


def sample(preference: str, reviewer: str = "r1", **overrides: object) -> PairwiseAnnotation:
    payload = {
        "prompt_id": "p1",
        "prompt": "Explain testing",
        "response_a": "A detailed explanation covering regressions and repeatability",
        "response_b": "A short answer",
        "preference": preference,
        "rationale": (
            "Prefer the detailed explanation because it is more complete and clearer "
            "while the short answer omits regressions."
        ),
        "reviewer": reviewer,
    }
    payload.update(overrides)
    return PairwiseAnnotation(**payload)  # type: ignore[arg-type]


def test_preference_summary() -> None:
    summary = preference_summary([sample("A"), sample("B"), sample("A")])
    assert summary["total"] == 3
    assert summary["a_win_rate"] == 0.6667


def test_export_uses_chosen_and_rejected() -> None:
    records = export_preference_records([sample("B")])
    assert records[0]["chosen"] == "A short answer"
    assert records[0]["rejected"] == "A detailed explanation covering regressions and repeatability"


def test_export_skips_ties_and_can_include_metadata() -> None:
    records = export_preference_records(
        [
            sample(
                "tie",
                rationale="Mark as tie: both answers are equivalent in clarity and correctness.",
            )
        ],
    )
    assert records == []

    with_meta = export_preference_records([sample("A")], include_metadata=True)
    assert with_meta[0]["rationale"].startswith("Prefer")
    assert "dimensions" in with_meta[0]
    assert with_meta[0]["reviewer"] == "r1"


def test_disagreement_detection() -> None:
    disagreement = disagreement_by_prompt([sample("A", "r1"), sample("B", "r2")])
    assert disagreement["p1"] is True


def test_rationale_quality_flags_thin_notes() -> None:
    thin = sample("A", rationale="A is better.")
    issues = rationale_quality_issues(thin)
    assert issues
    assert any("shorter" in issue for issue in issues)


def test_annotation_quality_report_on_strong_rationale() -> None:
    report = annotation_quality_report([sample("A")])
    assert report["flagged"] == 0
    assert report["pass_rate"] == 1.0
