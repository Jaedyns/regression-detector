from pydantic import BaseModel

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
