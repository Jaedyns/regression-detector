import argparse
import os
import sys
from collections import defaultdict

import anthropic
from dotenv import load_dotenv

from detector.baseline import BASELINE_PATH, load_baseline, save_baseline
from detector.compare import Comparison, compare
from detector.evaluate import CHECK_NAMES, CaseResult, run_suite
from detector.golden import load_golden
from detector.prompt_config import PromptConfig
from detector.report import render_markdown
from detector.stats import RunStats, summarize
from detector.storage import load_run, save_run


def print_report(results: list[CaseResult], stats: RunStats) -> None:
    print(f"\nAll checks passed in {stats.all_pass_rate:.0%} of {len(results)} trials "
          f"({stats.n_cases} cases x {stats.repeats} repeats)\n")
    for name in CHECK_NAMES:
        if name in stats.check_rates:
            print(f"  {name:<22} {stats.check_rates[name]:>6.1%}")

    failures: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in results:
        for c in r.checks:
            if not c.passed:
                failures[(r.case_id, c.name)].append(c.detail)
    if failures:
        print("\nFailures (failed/repeats, latest reason):")
        for (case_id, check), details in sorted(failures.items()):
            print(f"  {case_id:<34} {check:<22} {len(details)}/{stats.repeats}  {details[-1]}")


def print_comparison(comparison: Comparison, baseline_version: str) -> None:
    print(f"\nCompared with baseline (prompt {baseline_version}):")
    for c in comparison.improvements:
        print(f"  improved    {c.describe()}")
    for c in comparison.regressions:
        print(f"  REGRESSION  {c.describe()}")
    if comparison.new_cases:
        print(f"  new cases with no baseline: {', '.join(comparison.new_cases)}")
    if comparison.missing_cases:
        print(f"  baseline cases not run: {len(comparison.missing_cases)}")
    if not comparison.regressions and not comparison.improvements:
        print("  no change beyond noise")
    print("\nREGRESSION DETECTED" if comparison.regressed else "\nNo regression")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate a prompt version on the golden dataset and compare it to the baseline.")
    parser.add_argument("--prompt", help="Prompt version in prompts/ (default: the one in prompts/active.yaml)")
    parser.add_argument("--repeats", type=int, default=5, help="Runs per case, to average out non-determinism (default: 5)")
    parser.add_argument("--case", action="append", help="Only run this case id (repeatable)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (faster, cheaper)")
    parser.add_argument("--save-baseline", action="store_true", help="Make this run the new baseline")
    parser.add_argument("--markdown", metavar="FILE", help="Also append a Markdown report to FILE (CI passes $GITHUB_STEP_SUMMARY)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("No Claude API key. Put ANTHROPIC_API_KEY=... in the .env file in this folder.")

    cases = load_golden()
    if args.case:
        if args.save_baseline:
            sys.exit("A baseline must cover the whole dataset. Drop --case to save one.")
        cases = [c for c in cases if c.id in set(args.case)]
        if not cases:
            sys.exit(f"No golden case matches {args.case}")
    if args.save_baseline and args.no_judge:
        sys.exit("A baseline must include the judge. Drop --no-judge to save one.")

    config = PromptConfig.load(args.prompt)
    print(f"prompt {config.version} ({config.model}), {len(cases)} cases x {args.repeats} repeats, "
          f"judge {'off' if args.no_judge else 'on'}")

    results = run_suite(cases, config, anthropic.Anthropic(), judge=not args.no_judge, repeats=args.repeats)
    stats = summarize(results, config.version, config.model, args.repeats)
    run_id = save_run(results, config.version, config.model, args.repeats)
    print_report(results, stats)
    print(f"\nSaved as run {run_id} in runs/results.db")

    if args.save_baseline:
        save_baseline(stats, run_id)
        print(f"Saved as baseline: {BASELINE_PATH.relative_to(BASELINE_PATH.parents[1])}")
        return

    baseline = load_baseline()
    comparison = compare(baseline.stats, stats) if baseline else None
    if args.markdown:
        with open(args.markdown, "a") as f:
            f.write(render_markdown(results, stats, baseline.stats if baseline else None, comparison))

    if baseline is None:
        print("\nNo baseline yet. Run again with --save-baseline to create one.")
        return
    if baseline.stats.repeats != args.repeats:
        print(f"\nNote: baseline used {baseline.stats.repeats} repeats, this run used {args.repeats}. "
              "Per-case changes are less reliable.")
    print_comparison(comparison, baseline.stats.prompt_version)
    if comparison.regressed:
        sys.exit(1)


def promote_main() -> None:
    parser = argparse.ArgumentParser(description="Make a stored run the baseline, without re-running it.")
    parser.add_argument("run_id", type=int, help="Run id printed by `evaluate` (stored in runs/results.db)")
    args = parser.parse_args()

    run = load_run(args.run_id)
    if run is None:
        sys.exit(f"No run {args.run_id} in runs/results.db")

    stats = summarize(run.results, run.prompt_version, run.model, run.repeats)
    golden_ids = {c.id for c in load_golden()}
    missing = golden_ids - stats.case_rates.keys()
    if missing:
        sys.exit(f"Run {run.run_id} skipped {len(missing)} golden cases. A baseline must cover the whole dataset.")
    if "no_unsupported_claims" not in stats.check_rates:
        sys.exit(f"Run {run.run_id} ran without the judge. A baseline must include it.")

    previous = load_baseline()
    save_baseline(stats, run.run_id, sha=run.git_sha)
    old = f"prompt {previous.stats.prompt_version} ({previous.stats.all_pass_rate:.0%})" if previous else "none"
    print(f"Baseline was: {old}")
    print(f"Baseline now: prompt {run.prompt_version} ({stats.all_pass_rate:.0%}), run {run.run_id} "
          f"from {run.started_at} at commit {run.git_sha}")
