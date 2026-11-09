from detector.classifier import Classification
from detector.compare import compare
from detector.evaluate import CaseResult
from detector.evaluators import CheckResult
from detector.report import render_markdown
from detector.stats import summarize


def results(category_passes: list[bool], summary: str = "Bill rose from $29 to $35.") -> list[CaseResult]:
    return [
        CaseResult(case_id="billing-x", repeat=i, output=Classification(category="billing", summary=summary),
                   checks=[CheckResult(name="category", passed=p, detail="" if p else "expected billing, got general")])
        for i, p in enumerate(category_passes)
    ]


def test_regression_report_has_verdict_table_and_failing_case():
    base = summarize(results([True] * 5), "v1", "m", 5)
    now_results = results([True, False, False, False, True])
    now = summarize(now_results, "v2", "m", 5)
    md = render_markdown(now_results, now, base, compare(base, now))
    assert md.startswith("## ❌ Regression detected")
    assert "| `category` | 100.0% | 40.0% | -60.0 pts |" in md
    assert "### Regressions" in md
    assert "| `billing-x` | `category` | 3/5 |" in md


def test_clean_report_has_no_failure_section():
    rs = results([True] * 5)
    s = summarize(rs, "v2", "m", 5)
    md = render_markdown(rs, s, s, compare(s, s))
    assert md.startswith("## ✅ No regression")
    assert "Failing cases" not in md


def test_pipes_in_output_do_not_break_the_table():
    rs = results([False], summary="Plan A | Plan B confusion.")
    s = summarize(rs, "v2", "m", 1)
    assert "Plan A \\| Plan B" in render_markdown(rs, s, None, None)
