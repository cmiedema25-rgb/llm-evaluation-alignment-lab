from llm_eval_lab.models import PromptCase
from llm_eval_lab.prompt_suite import check_prompt_case


def test_prompt_case_passes_when_requirements_are_met() -> None:
    case = PromptCase(
        case_id="c1",
        prompt="Explain CI",
        required_terms=("automated", "tests"),
        forbidden_terms=("guaranteed",),
        max_words=20,
    )
    result = check_prompt_case(case, "Automated CI runs tests whenever code changes.")
    assert result.passed is True
    assert result.failures == ()


def test_prompt_case_reports_multiple_failures() -> None:
    case = PromptCase(
        case_id="c2",
        prompt="Explain alignment",
        required_terms=("preference",),
        forbidden_terms=("magic",),
        max_words=3,
    )
    result = check_prompt_case(case, "Magic automatically solves every alignment problem.")
    assert result.passed is False
    assert len(result.failures) == 3
