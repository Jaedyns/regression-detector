from collections import defaultdict

from pydantic import BaseModel

from detector.evaluate import CHECK_NAMES, CaseResult


class RunStats(BaseModel):
    prompt_version: str
    model: str
    repeats: int
    n_cases: int
    all_pass_rate: float
    check_rates: dict[str, float]
    case_rates: dict[str, dict[str, float]]


def _rate(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def summarize(results: list[CaseResult], prompt_version: str, model: str, repeats: int) -> RunStats:
    by_check: dict[str, list[bool]] = defaultdict(list)
    by_case: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for c in r.checks:
            by_check[c.name].append(c.passed)
            by_case[r.case_id][c.name].append(c.passed)
    return RunStats(
        prompt_version=prompt_version,
        model=model,
        repeats=repeats,
        n_cases=len(by_case),
        all_pass_rate=_rate([r.passed for r in results]),
        check_rates={n: _rate(by_check[n]) for n in CHECK_NAMES if n in by_check},
        case_rates={case: {n: _rate(flags) for n, flags in checks.items()} for case, checks in sorted(by_case.items())},
    )
