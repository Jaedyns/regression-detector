import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv

from detector.evaluate import CHECK_NAMES, CaseResult, run_suite
from detector.golden import load_golden
from detector.prompt_config import PromptConfig


def print_report(results: list[CaseResult]) -> None:
    n = len(results)
    print(f"\n{sum(r.passed for r in results)}/{n} cases passed every check\n")
    for name in CHECK_NAMES:
        checks = [c for r in results for c in r.checks if c.name == name]
        if checks:
            print(f"  {name:<22} {sum(c.passed for c in checks):>3}/{len(checks)}")
    failures = [(r, c) for r in results for c in r.checks if not c.passed]
    if failures:
        print("\nFailures:")
        for r, c in failures:
            print(f"  {r.case_id:<34} {c.name:<22} {c.detail}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run every check on the golden dataset once.")
    parser.add_argument("--prompt", default="v1", help="Prompt version in prompts/ (default: v1)")
    parser.add_argument("--case", action="append", help="Only run this case id (repeatable)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (faster, cheaper)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("No Claude API key. Put ANTHROPIC_API_KEY=... in the .env file in this folder.")

    cases = load_golden()
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
        if not cases:
            sys.exit(f"No golden case matches {args.case}")

    config = PromptConfig.load(args.prompt)
    print(f"prompt {config.version} ({config.model}), {len(cases)} cases, judge {'off' if args.no_judge else 'on'}")
    print_report(run_suite(cases, config, anthropic.Anthropic(), judge=not args.no_judge))
