from collections import defaultdict

from detector.compare import Comparison
from detector.evaluate import CHECK_NAMES, CaseResult
from detector.stats import RunStats


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(results: list[CaseResult], stats: RunStats,
                    baseline: RunStats | None, comparison: Comparison | None) -> str:
    if comparison is None:
        verdict = "No baseline to compare against"
    elif comparison.regressed:
        verdict = "❌ Regression detected"
    else:
        verdict = "✅ No regression"

    lines = [
        f"## {verdict}",
        "",
        f"Prompt **{stats.prompt_version}** (`{stats.model}`), {stats.n_cases} cases × {stats.repeats} repeats"
        + (f", compared with baseline prompt **{baseline.prompt_version}**." if baseline else "."),
        "",
        "| Check | Baseline | This run | Change |",
        "|---|---:|---:|---:|",
    ]
    for name in CHECK_NAMES:
        if name not in stats.check_rates:
            continue
        now = stats.check_rates[name]
        before = baseline.check_rates.get(name) if baseline else None
        change = f"{(now - before) * 100:+.1f} pts" if before is not None else ""
        lines.append(f"| `{name}` | {f'{before:.1%}' if before is not None else '—'} | {now:.1%} | {change} |")

    if comparison and comparison.regressions:
        lines += ["", "### Regressions", ""] + [f"- {_cell(c.describe())}" for c in comparison.regressions]
    if comparison and comparison.improvements:
        lines += ["", "### Improvements", ""] + [f"- {_cell(c.describe())}" for c in comparison.improvements]

    failures: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for r in results:
        for c in r.checks:
            if not c.passed:
                failures[(r.case_id, c.name)].append(r)
    if failures:
        lines += ["", "### Failing cases", "", "| Case | Check | Failed | Latest output |", "|---|---|---:|---|"]
        for (case_id, check), rs in sorted(failures.items()):
            latest = rs[-1]
            output = (f"`{latest.output.category}`: {latest.output.summary}" if latest.output
                      else f"error: {latest.error}")
            lines.append(f"| `{case_id}` | `{check}` | {len(rs)}/{stats.repeats} | {_cell(output)} |")

    return "\n".join(lines) + "\n"
