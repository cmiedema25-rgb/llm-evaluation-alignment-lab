from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .benchmark import run_benchmark, write_report
from .preferences import (
    annotation_quality_report,
    export_preference_records,
    load_jsonl,
    preference_summary,
)
from .rubric import evaluate_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM evaluation and alignment utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="score one prompt/response pair")
    evaluate.add_argument("--prompt", required=True)
    evaluate.add_argument("--response", required=True)
    evaluate.add_argument("--required-term", action="append", default=[])
    evaluate.add_argument("--max-words", type=int)

    summarize = subparsers.add_parser("summarize", help="summarize pairwise JSONL labels")
    summarize.add_argument("path")

    export = subparsers.add_parser(
        "export-preferences",
        help="export non-tie labels as prompt/chosen/rejected JSONL",
    )
    export.add_argument("path")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--with-metadata",
        action="store_true",
        help="include rationale, dimensions, reviewer, and confidence",
    )

    quality = subparsers.add_parser(
        "check-annotations",
        help="run rationale quality gates on pairwise JSONL labels",
    )
    quality.add_argument("path")

    benchmark = subparsers.add_parser(
        "benchmark", help="run the retained evaluation and alignment benchmark"
    )
    benchmark.add_argument("cases", type=Path)
    benchmark.add_argument("--preferences", type=Path, required=True)
    benchmark.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_response(
            args.prompt,
            args.response,
            required_terms=args.required_term,
            max_words=args.max_words,
        )
        print(json.dumps(result.__dict__, indent=2))
        return 0

    if args.command == "benchmark":
        report = run_benchmark(args.cases, args.preferences)
        print(json.dumps(report, indent=2))
        if args.report is not None:
            write_report(report, args.report)
        return 0 if report["passed"] else 1

    if args.command == "export-preferences":
        annotations = load_jsonl(args.path)
        records = export_preference_records(annotations, include_metadata=args.with_metadata)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"exported": len(records), "output": str(args.output)}, indent=2))
        return 0

    if args.command == "check-annotations":
        annotations = load_jsonl(args.path)
        report = annotation_quality_report(annotations)
        print(json.dumps(report, indent=2))
        return 0 if int(report["flagged"]) == 0 else 1

    annotations = load_jsonl(args.path)
    print(json.dumps(preference_summary(annotations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
