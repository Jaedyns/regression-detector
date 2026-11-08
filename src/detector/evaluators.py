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
