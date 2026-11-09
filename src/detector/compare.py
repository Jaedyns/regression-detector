from typing import Literal

from pydantic import BaseModel

from detector.stats import RunStats

# 33 cases x 5 repeats = 165 trials per check. At a ~95% pass rate the standard error of the
# difference between two runs is about 2.4 points, so a 5-point drop is ~2 standard errors:
# unlikely to be run-to-run noise.
CHECK_TOLERANCE = 0.05
# Per case with 5 repeats: 2 more failures out of 5 than the baseline.
CASE_TOLERANCE = 0.4


class Change(BaseModel):
    scope: Literal["check", "case"]
    check: str
    case_id: str | None = None
    before: float
    after: float

    def describe(self) -> str:
        where = self.check if self.scope == "check" else f"{self.case_id} / {self.check}"
        return f"{where}: {self.before:.0%} -> {self.after:.0%}"


class Comparison(BaseModel):
    regressions: list[Change]
    improvements: list[Change]
    new_cases: list[str]
    missing_cases: list[str]

    @property
    def regressed(self) -> bool:
        return bool(self.regressions)


def compare(baseline: RunStats, current: RunStats,
            check_tolerance: float = CHECK_TOLERANCE, case_tolerance: float = CASE_TOLERANCE) -> Comparison:
    regressions: list[Change] = []
    improvements: list[Change] = []

    def record(change: Change, tolerance: float) -> None:
        # Small epsilon so a drop of exactly the tolerance (e.g. 1.0 -> 0.6) counts despite float rounding.
        delta = change.after - change.before
        if delta <= -tolerance + 1e-9:
            regressions.append(change)
        elif delta >= tolerance - 1e-9:
            improvements.append(change)

    for check, before in baseline.check_rates.items():
        if check in current.check_rates:
            record(Change(scope="check", check=check, before=before, after=current.check_rates[check]), check_tolerance)

    shared = baseline.case_rates.keys() & current.case_rates.keys()
    for case_id in sorted(shared):
        for check, before in baseline.case_rates[case_id].items():
            after = current.case_rates[case_id].get(check)
            if after is not None:
                record(Change(scope="case", check=check, case_id=case_id, before=before, after=after), case_tolerance)

    return Comparison(
        regressions=regressions,
        improvements=improvements,
        new_cases=sorted(current.case_rates.keys() - baseline.case_rates.keys()),
        missing_cases=sorted(baseline.case_rates.keys() - current.case_rates.keys()),
    )
