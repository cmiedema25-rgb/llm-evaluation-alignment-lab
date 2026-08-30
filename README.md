# LLM Evaluation & Alignment Lab

[![CI](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)

A practical Python portfolio project for evaluating language-model responses, creating pairwise preference labels, regression-testing prompts, and preparing alignment/fine-tuning datasets.

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
└── cli.py          # command-line evaluation utility

data/sample_preferences.jsonl
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
pytest -q
```

Run the deterministic evaluator:

```bash
python examples/run_evaluation.py
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

The GitHub Actions workflow installs the package, runs Ruff, and executes pytest on Python 3.11 and 3.12. This provides an externally visible check that the core evaluation and annotation code is executable.

## Scope and integrity

This repository is a portfolio implementation. Metrics shown by the included examples are produced from the included sample data; no fabricated production ROI, client deployment, or completed large-model training run is claimed.

## License

MIT.
