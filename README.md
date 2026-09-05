# LLM Evaluation & Alignment Lab

[![CI](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/cmiedema25-rgb/llm-evaluation-alignment-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)

Offline toolkit for people who score LLM outputs, write preference rationales, and ship chosen/rejected pairs for alignment experiments.

Labeling and evaluation quality usually fail in quiet ways: thin rationales, missing evidence, silent prompt drift, and exports that drop the judgment trail. This lab keeps those checks local and deterministic—no API key—so the same cases always produce the same scores and CI can catch regressions.

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

# Summarize pairwise labels / run rationale quality gates
llm-eval-lab summarize data/sample_preferences.jsonl
llm-eval-lab check-annotations data/sample_preferences.jsonl

# Export clean prompt/chosen/rejected JSONL (add --with-metadata for audit fields)
llm-eval-lab export-preferences data/sample_preferences.jsonl \
  --output /tmp/prefs-export.jsonl

# Full benchmark + preference report
llm-eval-lab benchmark data/evaluation_cases.jsonl \
  --preferences data/sample_preferences.jsonl \
  --report evidence/benchmark-report.json
```

## What it does

**Response scoring** — weighted 0–1 scores for relevance, instruction following, safety pattern checks, and clarity.

**Prompt regression** — cases can require concepts, forbid terms, and cap length so prompt edits don’t silently drift.

**Preference labeling** — store A/B/tie labels with evidence-based rationales, writing-quality dimensions (helpfulness, correctness, instruction following, clarity, structure, tone, safety, completeness), confidence, and reviewer id. Compute win rates and reviewer disagreement; gate thin rationales that lack comparative judgment or response evidence.

**Preference export** — non-ties become `prompt` / `chosen` / `rejected` JSONL. Optional metadata keeps rationale, dimensions, reviewer, and confidence for audits and rater calibration.

**Optional LoRA path** — `finetune/lora_sft.py` wires Transformers + PEFT for small SFT runs when you install `.[finetune]`.

## Preference annotation shape

```json
{
  "prompt_id": "p4",
  "prompt": "Rewrite … for a non-expert …",
  "response_a": "…",
  "response_b": "…",
  "preference": "A",
  "rationale": "Prefer A … cites jargon removed … B still confuses a non-expert.",
  "dimensions": ["clarity", "tone", "instruction_following", "helpfulness"],
  "reviewer": "reviewer-1",
  "confidence": 0.97
}
```

Sample fixtures under `data/` are synthetic but written like production RLHF notes: specific evidence, dimension coverage, one intentional tie, and mixed reviewers.

## Layout

```text
src/llm_eval_lab/
├── models.py          # cases, pairwise annotations, writing dimensions
├── rubric.py          # deterministic response scoring
├── preferences.py     # load, summarize, quality gates, export
├── prompt_suite.py    # required/forbidden/length checks
├── benchmark.py       # retained offline benchmark
└── cli.py
data/                  # evaluation cases + sample preference labels
evidence/              # regenerated benchmark report
examples/
finetune/
tests/
```

## Tests & CI

`make verify` runs lint, tests (coverage floor), and regenerates the benchmark. GitHub Actions covers Python 3.11–3.13 plus CodeQL.

Fixture data under `data/` is synthetic—swap in your own cases and labels for real eval work.

## License

MIT.
