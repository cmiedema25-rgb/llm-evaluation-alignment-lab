# LLM Evaluation & Alignment Lab

[![CI](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)

Offline tooling for scoring LLM responses, regression-testing prompts, labeling pairwise preferences, and exporting chosen/rejected pairs for alignment experiments.

Uses a deterministic rubric (no API key required) so the same inputs always produce the same scores — useful for CI and prompt change detection.

## Install

```bash
git clone https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab.git
cd llm-evaluation-alignment-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make verify
```

## Quick usage

```bash
# Score sample cases
python examples/run_evaluation.py

# Full benchmark + preference export report
llm-eval-lab benchmark data/evaluation_cases.jsonl \
  --preferences data/sample_preferences.jsonl \
  --report evidence/benchmark-report.json
```

## What it does

**Response scoring** — weighted 0–1 scores for relevance, instruction following, safety pattern checks, and clarity.

**Prompt regression** — cases can require concepts, forbid terms, and cap length so prompt edits don’t silently drift.

**Preference labeling** — store A/B/tie labels with rationales, compute win rates and reviewer disagreement, export non-ties as `prompt` / `chosen` / `rejected` JSONL.

**Optional LoRA path** — `finetune/lora_sft.py` wires Transformers + PEFT for small SFT runs when you install `.[finetune]`.

## Layout

```text
src/llm_eval_lab/
├── models.py
├── rubric.py
├── preferences.py
├── prompt_suite.py
├── benchmark.py
└── cli.py
data/
evidence/
examples/
finetune/
tests/
```

## Tests & CI

`make verify` runs lint, tests (coverage floor), and regenerates the benchmark. GitHub Actions covers Python 3.11–3.13 plus CodeQL.

Fixture data under `data/` is synthetic — swap in your own cases for real eval work.

## License

MIT.
