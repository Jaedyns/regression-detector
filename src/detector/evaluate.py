from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import BaseModel

from detector.classifier import Classification, classify_email
from detector.evaluators import CheckResult, check_category, check_format, check_required_details
from detector.golden import GoldenCase
from detector.judge import check_unsupported_claims
from detector.prompt_config import PromptConfig

CHECK_NAMES = ["category", "required_details", "format", "no_unsupported_claims"]


class CaseResult(BaseModel):
    case_id: str
    output: Classification | None
    error: str | None = None
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)


def evaluate_case(case: GoldenCase, config: PromptConfig, client: anthropic.Anthropic, judge: bool = True) -> CaseResult:
    try:
        output = classify_email(case.email, config, client)
    except (anthropic.APIError, Exception) as e:  # a broken run is a failed case, not a crashed suite
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
              judge: bool = True, workers: int = 8) -> list[CaseResult]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda c: evaluate_case(c, config, client, judge), cases))
