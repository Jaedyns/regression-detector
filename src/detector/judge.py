import anthropic
from pydantic import BaseModel

from detector.classifier import Classification
from detector.golden import GoldenCase

# Pinned separately from prompts/: the grader must stay fixed while the classifier changes,
# or a score shift could come from the judge instead of the prompt under test.
JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_EFFORT = "medium"

JUDGE_SYSTEM = """\
You check summaries of customer-support emails for unsupported claims.

An unsupported claim is anything the summary states as fact that the email does not say:
- invented details (amounts, dates, names, causes, product areas)
- something the customer only reported or heard, stated as confirmed fact
  (the customer says a colleague mentioned a discount; the summary says a discount was offered)
- promises, decisions or outcomes attributed to the company that the email does not contain

These are NOT unsupported claims: rewording, shortening, leaving details out, or a neutral
description of what the customer is asking for.

List each unsupported claim as a short quote or paraphrase from the summary. If there are
none, return an empty list."""


class JudgeVerdict(BaseModel):
    unsupported_claims: list[str]


def judge_prompt(email: str, summary: str) -> str:
    return f"<email>\n{email}\n</email>\n\n<summary>\n{summary}\n</summary>"


class JudgeRefused(Exception):
    pass


def judge_summary(email: str, summary: str, client: anthropic.Anthropic) -> JudgeVerdict:
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=2048,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt(email, summary)}],
        output_config={"effort": JUDGE_EFFORT},
        output_format=JudgeVerdict,
    )
    if response.stop_reason == "refusal":
        raise JudgeRefused(getattr(response.stop_details, "explanation", None) or "refused")
    return response.parsed_output


def check_unsupported_claims(case: GoldenCase, output: Classification, client: anthropic.Anthropic):
    from detector.evaluators import CheckResult

    verdict = judge_summary(case.email, output.summary, client)
    return CheckResult(
        name="no_unsupported_claims",
        passed=not verdict.unsupported_claims,
        detail="; ".join(verdict.unsupported_claims),
    )
