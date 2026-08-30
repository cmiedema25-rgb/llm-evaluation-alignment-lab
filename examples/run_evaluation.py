from llm_eval_lab.models import PromptCase
from llm_eval_lab.prompt_suite import check_prompt_case
from llm_eval_lab.rubric import evaluate_response

prompt = "Explain why deterministic prompt tests matter for AI applications."
response = (
    "Deterministic prompt tests make AI application regressions visible by checking "
    "the same behavioral requirements after each prompt change."
)

score = evaluate_response(
    prompt,
    response,
    required_terms=("regressions", "requirements"),
    max_words=35,
)
print("Evaluation:", score)

case = PromptCase(
    case_id="prompt-regression-demo",
    prompt=prompt,
    required_terms=("regressions",),
    forbidden_terms=("guaranteed",),
    max_words=35,
)
print("Prompt check:", check_prompt_case(case, response))
