# Changelog

## 1.1.0 — 2026-09-05

### Preference labeling & writing quality

- Expanded `data/sample_preferences.jsonl` with evidence-based rationales, writing-quality dimensions, confidence scores, a deliberate tie, and a second reviewer.
- Added rationale quality gates (`rationale_quality_issues`, `annotation_quality_report`, CLI `check-annotations`) that flag thin notes lacking comparative judgment or response evidence.
- Documented shared writing-quality dimensions (`WRITING_QUALITY_DIMENSIONS`) and optional `confidence` on `PairwiseAnnotation`.
- Export now supports clean DPO-style triples plus `--with-metadata` audit fields (`export-preferences` CLI).

### Evaluation coverage

- Grew `data/evaluation_cases.jsonl` with realistic prompts (jargon rewrite, safe refusal, rater checklist, release note) while keeping preferred/rejected ranking deterministic.
- Benchmark schema `1.1` tracks non-ties, annotation quality pass counts, and dimension frequency; ties no longer fail the suite.

### Docs

- README reframed around labeling/evaluation failure modes and the annotation schema; added CHANGELOG.
