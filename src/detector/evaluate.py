import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from detector.classifier import Classification, classify_email
from detector.evaluators import CheckResult, check_category, check_format, check_required_details
from detector.golden import GoldenCase, load_golden
from detector.judge import check_unsupported_claims
from detector.prompt_config import PromptConfig

CHECK_NAMES = ["category", "required_details", "format", "no_unsupported_claims"]


class CaseResult(BaseModel):
    case_id: str
    repeat: int = 0
    output: Classification | None
    error: str | None = None
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)


def evaluate_case(case: GoldenCase, config: PromptConfig, client: anthropic.Anthropic, judge: bool = True) -> CaseResult:
    try:
        output = classify_email(case.email, config, client)
    except Exception as e:  # a broken run is a failed case, not a crashed suite
        failed = [CheckResult(name=n, passed=False, detail="classifier error") for n in CHECK_NAMES]
        return CaseResult(case_id=case.id, output=None, error=f"{type(e).__name__}: {e}", checks=failed)

    checks = [check_category(case, output), check_required_details(case, output), check_format(case, output)]
    if judge:
        try:
            checks.append(check_unsupported_claims(case, output, client))
        except Exception as e:
            checks.append(CheckResult(name="no_unsupported_claims", passed=False, detail=f"judge error: {e}"))
    return CaseResult(case_id=case.id, output=output, checks=checks)


def run_suite(cases: list[GoldenCase], config: PromptConfig, client: anthropic.Anthropic,
              judge: bool = True, repeats: int = 1, workers: int = 8) -> list[CaseResult]:
    def run(task: tuple[GoldenCase, int]) -> CaseResult:
        case, repeat = task
        return evaluate_case(case, config, client, judge).model_copy(update={"repeat": repeat})

    tasks = [(case, r) for r in range(repeats) for case in cases]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(run, tasks))


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
