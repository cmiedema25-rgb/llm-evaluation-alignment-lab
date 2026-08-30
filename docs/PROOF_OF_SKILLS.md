# Proof of Skills

This page maps each portfolio skill to inspectable implementation, tests,
retained evidence, and a command a reviewer can run.

## Natural Language Processing and LLM Evaluation

- `src/llm_eval_lab/rubric.py` implements deterministic lexical relevance,
  instruction-following, safety-pattern, clarity, and weighted aggregate scores.
- `src/llm_eval_lab/benchmark.py` evaluates both responses in every pair and
  records whether the intended response receives the higher aggregate score.
- `data/evaluation_cases.jsonl` contains ten authored pairs spanning evaluation,
  prompt regression, preference data, safety refusal, and LoRA concepts.
- `tests/test_rubric.py` and `tests/test_benchmark.py` verify ordering, scoring,
  retained report output, and invalid-input handling.

Observed result: the intended response ranked higher in 10/10 included pairs.

## Prompt Engineering

- `src/llm_eval_lab/prompt_suite.py` checks required terms, forbidden terms, and
  maximum response length without brittle exact-string comparison.
- Every benchmark case declares explicit behavioral requirements.
- The per-case report retains failures rather than only an overall exit code.

Observed result: 10/10 included preferred responses passed their declared prompt
requirements.

## Data Annotation and Labeling

- `src/llm_eval_lab/models.py` validates pairwise labels, non-empty candidates,
  rationales, evaluation dimensions, and reviewer identifiers.
- `data/sample_preferences.jsonl` contains three inspectable sample annotations.
- `src/llm_eval_lab/preferences.py` calculates label rates, flags conflicting
  labels per prompt, and exports non-tie labels as chosen/rejected records.

Observed result: all 3 annotations exported to preference-training records, and
the small single-reviewer sample contained 0 conflicting prompt labels.

## RLHF and Alignment

The chosen/rejected export is suitable as input preparation for downstream
preference-optimization experiments. This repository demonstrates validation,
aggregation, disagreement checks, and export; it does not claim that RLHF or DPO
training was run.

## Model Fine-tuning

- `finetune/lora_sft.py` provides an optional Transformers/Datasets/PEFT LoRA
  entry point.
- `data/sft_sample.jsonl` demonstrates the expected supervised data contract.
- Heavy dependencies are isolated in the `finetune` extra so the evaluation
  pipeline remains reproducible on a CPU-only environment.

No trained adapter or post-training quality gain is claimed or used as evidence.

## Reproduce the evidence

~~~bash
python -m pip install -e '.[dev]'
make verify
~~~

Expected: Ruff passes, 11 tests pass with at least 85% statement coverage, and
the benchmark reports 10/10 rubric preference wins, 10/10 prompt checks, and
3/3 preference exports.

## Measurement boundary

The authored regression suite demonstrates deterministic mechanics and catches
changes to this codebase. It is not an independent model benchmark, a
representative human-label study, a safety certification, or evidence of
financial return in a customer deployment.
