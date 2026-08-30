# Verification Record

Date: 2026-08-30 UTC

Environment: CPython 3.12 on Linux.

## Commands

~~~bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest --cov=llm_eval_lab --cov-report=term-missing --cov-fail-under=85 -q
llm-eval-lab benchmark data/evaluation_cases.jsonl \
  --preferences data/sample_preferences.jsonl \
  --report evidence/benchmark-report.json
~~~

## Observed results

| Check | Result |
| --- | ---: |
| Automated tests | 11 passed |
| Python statement coverage | 89.23% (85% floor) |
| Authored evaluation pairs | 10 |
| Intended-response rubric wins | 10/10 |
| Prompt requirement checks | 10/10 passed |
| Pairwise annotations validated | 3 |
| Chosen/rejected records exported | 3/3 |
| Prompts with conflicting labels | 0 |

The cases are intentionally small, deterministic, and checked in. The results
verify the repository's behavior but do not claim production model quality,
large-scale inter-annotator agreement, safety certification, or customer ROI.
