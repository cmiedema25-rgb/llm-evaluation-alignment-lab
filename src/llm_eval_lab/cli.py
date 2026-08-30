from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .benchmark import run_benchmark, write_report
from .preferences import load_jsonl, preference_summary
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

    annotations = load_jsonl(args.path)
    print(json.dumps(preference_summary(annotations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
