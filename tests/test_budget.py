import pytest

from search.budget import filter_by_budget, parse_budget


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10만원 이하로 보여줘", {"max_price": 100000, "mode": "under"}),
        ("15만원 이내로", {"max_price": 150000, "mode": "under"}),
        ("3만5천원 이하", {"max_price": 35000, "mode": "under"}),
        ("100000원 이하", {"max_price": 100000, "mode": "under"}),
        ("50,000원 안으로 찾아줘", {"max_price": 50000, "mode": "under"}),
        ("20만원 미만", {"max_price": 200000, "mode": "under"}),
        ("더 저렴한 거 있어?", {"max_price": None, "mode": "cheaper"}),
        ("더 싼 거 보여줘", {"max_price": None, "mode": "cheaper"}),
    ],
)
def test_parse_budget_extracts_condition(text, expected):
    assert parse_budget(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "오늘 날씨 어때?",
        "이 옷 예쁘다",
        "가격 알려줘",
    ],
)
def test_parse_budget_returns_none_for_unmatched_text(text):
    assert parse_budget(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3만원보다 더 싸게", {"max_price": 30000, "mode": "cheaper"}),
        ("10만원보다 더 저렴한 거 있어?", {"max_price": 100000, "mode": "cheaper"}),
    ],
)
def test_parse_budget_keeps_amount_with_cheaper_keyword(text, expected):
    assert parse_budget(text) == expected


def test_filter_by_budget_filters_even_in_cheaper_mode_when_amount_present():
    products = [
        {"name": "A", "price": 150000},
        {"name": "B", "price": 50000},
        {"name": "C", "price": 200000},
    ]
    condition = {"max_price": 100000, "mode": "cheaper"}

    result = filter_by_budget(products, condition)

    assert [p["name"] for p in result] == ["B"]


def test_filter_by_budget_filters_and_sorts_ascending():
    products = [
        {"name": "A", "price": 150000},
        {"name": "B", "price": 50000},
        {"name": "C", "price": 90000},
        {"name": "D", "price": 200000},
    ]
    condition = {"max_price": 100000, "mode": "under"}

    result = filter_by_budget(products, condition)

    assert [p["name"] for p in result] == ["B", "C"]
    assert [p["price"] for p in result] == [50000, 90000]


def test_filter_by_budget_with_none_condition_only_sorts():
    products = [
        {"name": "A", "price": 150000},
        {"name": "B", "price": 50000},
    ]

    result = filter_by_budget(products, None)

    assert [p["name"] for p in result] == ["B", "A"]


def test_filter_by_budget_with_cheaper_mode_only_sorts_no_filter():
    products = [
        {"name": "A", "price": 150000},
        {"name": "B", "price": 50000},
    ]
    condition = {"max_price": None, "mode": "cheaper"}

    result = filter_by_budget(products, condition)

    assert [p["name"] for p in result] == ["B", "A"]


def test_filter_by_budget_includes_boundary_price():
    products = [{"name": "A", "price": 100000}]
    condition = {"max_price": 100000, "mode": "under"}

    result = filter_by_budget(products, condition)

    assert [p["name"] for p in result] == ["A"]
