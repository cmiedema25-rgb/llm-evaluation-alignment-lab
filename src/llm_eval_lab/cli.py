from __future__ import annotations

import argparse
import json

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        result = evaluate_response(
            args.prompt,
            args.response,
            required_terms=args.required_term,
            max_words=args.max_words,
        )
        print(json.dumps(result.__dict__, indent=2))
        return

    annotations = load_jsonl(args.path)
    print(json.dumps(preference_summary(annotations), indent=2))


if __name__ == "__main__":
    main()
