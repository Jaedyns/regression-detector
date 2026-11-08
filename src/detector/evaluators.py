import re

from pydantic import BaseModel

from detector.classifier import Classification
from detector.golden import GoldenCase


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


def check_category(case: GoldenCase, output: Classification) -> CheckResult:
    passed = output.category == case.category
    return CheckResult(
        name="category",
        passed=passed,
        detail="" if passed else f"expected {case.category}, got {output.category}",
    )


def check_required_details(case: GoldenCase, output: Classification) -> CheckResult:
    summary = output.summary.lower()
    missing = [d for d in case.required_details if d.lower() not in summary]
    return CheckResult(
        name="required_details",
        passed=not missing,
        detail="" if not missing else f"missing {missing}",
    )


MAX_SUMMARY_CHARS = 300
# A sentence break is end punctuation, whitespace, then a capital. "v4.12.0" and "$1,188" don't match.
SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z]")


def check_format(case: GoldenCase, output: Classification) -> CheckResult:
    summary = output.summary.strip()
    problems = []
    if not summary:
        problems.append("summary is empty")
    if len(summary) > MAX_SUMMARY_CHARS:
        problems.append(f"summary is {len(summary)} chars (max {MAX_SUMMARY_CHARS})")
    if SENTENCE_BREAK.search(summary):
        problems.append("summary is more than one sentence")
    return CheckResult(name="format", passed=not problems, detail="; ".join(problems))
