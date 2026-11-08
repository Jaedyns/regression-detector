import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from detector.prompt_config import PROMPTS_DIR, Category, PromptConfig

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "golden" / "emails.yaml"


class GoldenCase(BaseModel):
    id: str
    category: Category
    difficulty: Literal["clear", "ambiguous", "edge"]
    email: str
    required_details: list[str]
    notes: str | None = None


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    return [GoldenCase.model_validate(row) for row in yaml.safe_load(path.read_text())]


def find_problems(cases: list[GoldenCase]) -> list[str]:
    problems = []
    dupes = [i for i, n in Counter(c.id for c in cases).items() if n > 1]
    problems += [f"duplicate id: {i}" for i in dupes]

    for c in cases:
        for detail in c.required_details:
            if detail.lower() not in c.email.lower():
                problems.append(f"{c.id}: required detail {detail!r} is not in the email")
        if c.difficulty != "clear" and not c.notes:
            problems.append(f"{c.id}: {c.difficulty} cases need a note explaining the label")

    few_shot = {
        ex.email.strip().lower()
        for p in PROMPTS_DIR.glob("classifier_*.yaml")
        for ex in PromptConfig.load(p.stem.removeprefix("classifier_")).few_shot_examples
    }
    problems += [f"{c.id}: duplicates a few-shot example" for c in cases if c.email.strip().lower() in few_shot]
    return problems


def main() -> None:
    cases = load_golden()
    by_cat = Counter(c.category for c in cases)
    by_diff = Counter(c.difficulty for c in cases)
    print(f"{len(cases)} cases")
    print("  by category:   " + ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items())))
    print("  by difficulty: " + ", ".join(f"{k} {v}" for k, v in sorted(by_diff.items())))

    problems = find_problems(cases)
    for p in problems:
        print(f"  PROBLEM  {p}")
    if problems:
        sys.exit(1)
    print("dataset OK")
