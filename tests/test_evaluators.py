from detector.classifier import Classification
from detector.evaluators import MAX_SUMMARY_CHARS, check_category, check_format, check_required_details
from detector.golden import GoldenCase

CASE = GoldenCase(
    id="t",
    category="billing",
    difficulty="clear",
    email="Why did our bill go from $29 to $35 this month?",
    required_details=["$29", "$35"],
)


def out(category="billing", summary="Customer's bill rose from $29 to $35 without notice."):
    return Classification(category=category, summary=summary)


def test_category_pass():
    assert check_category(CASE, out()).passed


def test_category_fail_names_both_labels():
    r = check_category(CASE, out(category="technical"))
    assert not r.passed
    assert "billing" in r.detail and "technical" in r.detail


def test_details_pass_case_insensitive():
    case = CASE.model_copy(update={"required_details": ["BILL"]})
    assert check_required_details(case, out()).passed


def test_details_fail_lists_missing():
    r = check_required_details(CASE, out(summary="Customer's bill went up."))
    assert not r.passed
    assert "$29" in r.detail and "$35" in r.detail


def test_details_empty_list_passes():
    case = CASE.model_copy(update={"required_details": []})
    assert check_required_details(case, out(summary="Refund requested.")).passed


def test_format_pass():
    assert check_format(CASE, out()).passed


def test_format_version_numbers_are_not_sentence_breaks():
    assert check_format(CASE, out(summary="App crashes on launch after updating to 4.12.0 on iOS 18.1.")).passed


def test_format_fails_two_sentences():
    r = check_format(CASE, out(summary="Bill went up. Customer is upset."))
    assert not r.passed and "more than one sentence" in r.detail


def test_format_fails_empty():
    assert not check_format(CASE, out(summary="   ")).passed


def test_format_fails_too_long():
    assert not check_format(CASE, out(summary="a" * (MAX_SUMMARY_CHARS + 1))).passed
