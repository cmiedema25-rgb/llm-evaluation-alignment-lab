# Verification Record

Date: 2026-09-05 UTC

Environment: CPython 3.13 on Linux.

## Commands

~~~bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest --cov=llm_eval_lab --cov-report=term-missing --cov-fail-under=85 -q
llm-eval-lab check-annotations data/sample_preferences.jsonl
llm-eval-lab benchmark data/evaluation_cases.jsonl \
  --preferences data/sample_preferences.jsonl \
  --report evidence/benchmark-report.json
~~~

## Observed results

| Check | Result |
| --- | ---: |
| Automated tests | 15 passed |
| Python statement coverage | 90.37% (85% floor) |
| Authored evaluation pairs | 14 |
| Intended-response rubric wins | 14/14 |
| Prompt requirement checks | 14/14 passed |
| Pairwise annotations validated | 8 |
| Annotation quality gates | 8/8 passing |
| Chosen/rejected records exported | 7/7 non-ties |
| Prompts with conflicting labels | 0 |

The cases are intentionally small, deterministic, and checked in. The results
verify the repository's behavior but do not claim production model quality,
large-scale inter-annotator agreement, safety certification, or customer ROI.
