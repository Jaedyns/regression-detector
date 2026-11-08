import anthropic
from pydantic import BaseModel

from detector.prompt_config import Category, PromptConfig


class Classification(BaseModel):
    category: Category
    summary: str


class ClassifierRefused(Exception):
    pass


def _build_messages(email: str, config: PromptConfig) -> list[dict]:
    messages = []
    for ex in config.few_shot_examples:
        messages.append({"role": "user", "content": ex.email})
        messages.append({
            "role": "assistant",
            "content": Classification(category=ex.category, summary=ex.summary).model_dump_json(),
        })
    messages.append({"role": "user", "content": email})
    return messages


def classify_email(email: str, config: PromptConfig, client: anthropic.Anthropic | None = None) -> Classification:
    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=config.model,
        max_tokens=1024,
        system=config.system_prompt,
        messages=_build_messages(email, config),
        output_config={"effort": config.effort},
        output_format=Classification,
    )
    # A refusal has no parsed output; the eval pipeline must count it as a failure, not crash on None.
    if response.stop_reason == "refusal":
        raise ClassifierRefused(getattr(response.stop_details, "explanation", None) or "refused")
    return response.parsed_output
