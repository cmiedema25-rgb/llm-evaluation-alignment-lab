# LLM Evaluation & Alignment Lab

[![CI](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)

A practical Python project for evaluating language-model responses, creating
pairwise preference labels, regression-testing prompts, and preparing
alignment/fine-tuning datasets.

## Reviewer proof in 60 seconds

The repository retains its benchmark output and provides one offline command to
reproduce every headline result. No API key, model download, or GPU is needed.

| Verifiable outcome | Retained evidence | Reproduce it |
| --- | --- | --- |
| Intended response ranked higher in 10/10 authored pairs | [`evidence/benchmark-report.json`](evidence/benchmark-report.json) | `make benchmark` |
| Prompt requirements passed in 10/10 cases | [`evidence/benchmark-report.json`](evidence/benchmark-report.json) | `make benchmark` |
| 3/3 pairwise labels exported to chosen/rejected records | [`evidence/benchmark-report.json`](evidence/benchmark-report.json) | `make benchmark` |
| 0 conflicting prompts in the three-label sample | [`evidence/benchmark-report.json`](evidence/benchmark-report.json) | `make benchmark` |
| 11 tests passed with 89.23% coverage (85% floor) | [`evidence/VERIFICATION.md`](evidence/VERIFICATION.md) | `make verify` |

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
make verify
```

These are deterministic results on ten authored regression cases and three
sample annotations. They prove the pipeline runs as documented; they do not
measure a production model, annotation quality at scale, or customer ROI. See
the [verification record](evidence/VERIFICATION.md) and [claim-to-code
map](docs/PROOF_OF_SKILLS.md).

## Skills demonstrated

- **Natural Language Processing:** token normalization, lexical relevance scoring, response-length and structure analysis.
- **Prompt Engineering:** reusable prompt test cases with required concepts, forbidden content, and length constraints.
- **Data Annotation & Labeling:** validated pairwise annotations with preference labels, rationales, dimensions, and reviewer metadata.
- **RLHF & Alignment:** preference aggregation, disagreement analysis, win-rate reporting, and export to chosen/rejected records used by preference-optimization workflows.
- **Model Fine-tuning:** an optional Hugging Face/PEFT LoRA training entry point and dataset-preparation path. The core CI does not require a GPU or claim that a model was trained in CI.
- **Software Quality:** typed Python, pytest, Ruff, examples, and GitHub Actions.

## Architecture

```text
src/llm_eval_lab/
├── models.py       # typed evaluation and annotation records
├── rubric.py       # deterministic response scoring
├── preferences.py  # pairwise labels, agreement, DPO-style export
├── prompt_suite.py # prompt regression tests
├── benchmark.py    # reproducible evaluation and alignment report
└── cli.py          # command-line evaluation utility

data/evaluation_cases.jsonl
data/sample_preferences.jsonl
evidence/benchmark-report.json
examples/run_evaluation.py
finetune/lora_sft.py
tests/
```

## Quick start

```bash
git clone https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab.git
cd llm-evaluation-alignment-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make verify
```

Run the deterministic evaluator:

```bash
python examples/run_evaluation.py
```

Run the complete retained benchmark:

```bash
llm-eval-lab benchmark data/evaluation_cases.jsonl \
  --preferences data/sample_preferences.jsonl \
  --report evidence/benchmark-report.json
```

## Evaluation rubric

The core evaluator scores four dimensions on a 0–1 scale:

1. **Relevance** — normalized lexical overlap between prompt and answer.
2. **Instruction following** — optional required terms and maximum word count.
3. **Safety** — deterministic checks for configured unsafe instruction patterns.
4. **Clarity** — sentence/structure heuristics that penalize empty or excessively fragmented output.

The aggregate score is a weighted mean. Because the implementation is deterministic, the same input produces the same result, which makes it suitable for regression testing.

## Pairwise preference annotation

`PairwiseAnnotation` stores a prompt, two candidate responses, the preferred side (`A`, `B`, or `tie`), evaluation dimensions, rationale, and reviewer identifier. The preference module can:

- validate records;
- calculate A/B/tie win rates;
- measure reviewer disagreement by prompt;
- export non-tie records as `prompt`, `chosen`, `rejected` JSONL suitable for downstream preference-training experiments.

Sample records are included in `data/sample_preferences.jsonl`.

## Prompt regression testing

Prompt cases define expected properties rather than one brittle exact string. A case can require concepts, reject forbidden terms, and enforce a maximum response length. This supports testing prompt revisions against stable behavioral requirements.

## Optional LoRA fine-tuning pipeline

`finetune/lora_sft.py` provides a small supervised fine-tuning entry point using Transformers, Datasets, and PEFT. Heavy ML dependencies are optional:

```bash
python -m pip install -e ".[finetune]"
python finetune/lora_sft.py --model sshleifer/tiny-gpt2 --dataset data/sft_sample.jsonl --output-dir outputs/lora
```

This script demonstrates model loading, tokenizer setup, LoRA configuration, dataset tokenization, training arguments, and adapter saving. Use an appropriate model, dataset, hardware, and license terms for real training.

## Verification

GitHub Actions installs the package, checks Ruff lint and formatting, enforces
the test coverage floor, and regenerates the benchmark on Python 3.11, 3.12,
and 3.13. A separate CodeQL workflow analyzes the Python source. The retained
report exposes both aggregate and per-case scores so a reviewer can inspect the
claim without relying on README prose.

## Scope and integrity

This repository is a portfolio implementation. Metrics shown by the benchmark
come only from the checked-in synthetic data; no production ROI, client
deployment, human-label agreement study, or completed large-model training run
is claimed. The optional LoRA entry point demonstrates plumbing, not a retained
trained adapter or a model-quality improvement.

## License

MIT.
