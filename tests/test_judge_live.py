import os

import anthropic
import pytest
from dotenv import load_dotenv

from detector.judge import judge_summary

load_dotenv()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"),
]

FREE_MONTH = "My colleague said you're giving everyone a free month because of last week's outage. How do I claim mine?"

# (email, summary, should_flag). Hand-written so the right verdict is known in advance.
CALIBRATION = [
    pytest.param(FREE_MONTH,
                 "Customer is owed the free month the company promised after last week's outage.",
                 True, id="hearsay-stated-as-fact"),
    pytest.param("refund",
                 "Customer wants a refund for the $49 Pro plan charge.",
                 True, id="invented-amount-and-plan"),
    pytest.param("The iOS app crashes on launch since I updated to 4.12.0.",
                 "App crashes on launch after the 4.12.0 update, caused by a memory leak.",
                 True, id="invented-cause"),
    pytest.param(FREE_MONTH,
                 "Customer heard from a colleague that a free month is being offered after last week's outage and asks how to claim it.",
                 False, id="hearsay-attributed"),
    pytest.param("Why did our bill go from $29 to $35 this month? Nobody told us about a price change.",
                 "Customer asks why their bill rose from $29 to $35 without notice.",
                 False, id="faithful-reword"),
    pytest.param("refund",
                 "Customer is requesting a refund.",
                 False, id="faithful-minimal"),
]


@pytest.fixture(scope="module")
def client():
    return anthropic.Anthropic()


@pytest.mark.parametrize("email,summary,should_flag", CALIBRATION)
def test_judge_calibration(client, email, summary, should_flag):
    verdict = judge_summary(email, summary, client)
    assert bool(verdict.unsupported_claims) == should_flag, verdict.unsupported_claims
