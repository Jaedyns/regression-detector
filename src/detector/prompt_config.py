from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

Category = Literal["billing", "technical", "account", "general"]


class FewShotExample(BaseModel):
    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    version: str
    created: date
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    system_prompt: str
    few_shot_examples: list[FewShotExample] = []

    @classmethod
    def load(cls, version: str) -> "PromptConfig":
        path = PROMPTS_DIR / f"classifier_{version}.yaml"
        return cls.model_validate(yaml.safe_load(path.read_text()))
