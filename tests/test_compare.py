from detector.compare import compare
from detector.stats import RunStats


def stats(check_rates=None, case_rates=None):
    check_rates = check_rates or {"category": 0.95}
    case_rates = case_rates or {"a": {"category": 1.0}}
    return RunStats(prompt_version="v", model="m", repeats=5, n_cases=len(case_rates),
                    all_pass_rate=0.9, check_rates=check_rates, case_rates=case_rates)


def test_identical_runs_have_no_changes():
    c = compare(stats(), stats())
    assert not c.regressed and not c.improvements


def test_noise_within_tolerance_is_ignored():
    c = compare(stats({"category": 0.95}), stats({"category": 0.92}))
    assert not c.regressed


def test_check_drop_beyond_tolerance_is_regression():
    c = compare(stats({"category": 0.95}), stats({"category": 0.85}))
    assert c.regressed
    assert c.regressions[0].scope == "check" and c.regressions[0].check == "category"


def test_check_rise_beyond_tolerance_is_improvement():
    c = compare(stats({"category": 0.85}), stats({"category": 0.95}))
    assert not c.regressed and len(c.improvements) == 1


def test_case_drop_of_two_in_five_is_regression():
    c = compare(stats(case_rates={"a": {"category": 1.0}}), stats(case_rates={"a": {"category": 0.6}}))
    assert [ch.case_id for ch in c.regressions] == ["a"]


def test_case_drop_of_one_in_five_is_noise():
    c = compare(stats(case_rates={"a": {"category": 1.0}}), stats(case_rates={"a": {"category": 0.8}}))
    assert not c.regressed


def test_new_and_missing_cases_are_reported_not_compared():
    c = compare(stats(case_rates={"a": {"category": 1.0}}), stats(case_rates={"b": {"category": 0.0}}))
    assert c.new_cases == ["b"] and c.missing_cases == ["a"] and not c.regressed


def test_describe_is_readable():
    c = compare(stats(case_rates={"a": {"category": 1.0}}), stats(case_rates={"a": {"category": 0.2}}))
    assert c.regressions[0].describe() == "a / category: 100% -> 20%"
