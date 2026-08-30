from llm_eval_lab.rubric import evaluate_response


def test_relevant_response_scores_higher_than_irrelevant_response() -> None:
    prompt = "Explain why deterministic tests matter for AI applications"
    good = evaluate_response(
        prompt,
        "Deterministic tests for AI applications make regressions visible and repeatable.",
    )
    bad = evaluate_response(prompt, "The weather is warm today.")
    assert good.relevance > bad.relevance
    assert good.aggregate > bad.aggregate


def test_instruction_constraints_affect_score() -> None:
    result = evaluate_response(
        "Summarize evaluation",
        "Evaluation uses a rubric to score a response.",
        required_terms=["rubric"],
        max_words=20,
    )
    assert result.instruction_following == 1.0


def test_empty_response_is_handled() -> None:
    result = evaluate_response("Explain testing", "   ")
    assert result.aggregate == 0.2
    assert "empty response" in result.notes
