"""자연어 예산 조건 파싱 및 가격 필터링/정렬."""

import re

_MAN_CHEON_PATTERN = re.compile(r"(\d+)\s*만\s*(?:(\d+)\s*천)?\s*원")
_PLAIN_WON_PATTERN = re.compile(r"(\d[\d,]*)\s*원")
_UNDER_KEYWORDS = ("이하", "이내", "안으로", "미만")
_CHEAPER_KEYWORDS = ("더 저렴", "더 싸", "더 싼")


def parse_budget(text: str) -> dict | None:
    """자연어 문장에서 예산 조건을 추출한다.

    Returns:
        {"max_price": int|None, "mode": "under"|"cheaper"} 또는 패턴이 없으면 None.
    """
    max_price = _extract_won_amount(text)
    if max_price is not None and any(kw in text for kw in _UNDER_KEYWORDS):
        return {"max_price": max_price, "mode": "under"}
    if any(kw in text for kw in _CHEAPER_KEYWORDS):
        # "3만원보다 더 싸게"처럼 "더 싸게" 계열 키워드와 명시적 금액이 함께
        # 오면 mode는 "cheaper"지만 파싱된 max_price도 그대로 넘긴다 (금액이
        # 없으면 None 유지 — 순수 "더 저렴한 거 있어?" 케이스).
        return {"max_price": max_price, "mode": "cheaper"}
    return None


def filter_by_budget(products: list[dict], condition: dict | None) -> list[dict]:
    """예산 조건에 맞는 상품만 남기고 가격 오름차순으로 정렬한다."""
    max_price = condition.get("max_price") if condition else None
    if max_price is not None:
        filtered = [p for p in products if p["price"] <= max_price]
    else:
        filtered = list(products)
    return sorted(filtered, key=lambda p: p["price"])


def _extract_won_amount(text: str) -> int | None:
    match = _MAN_CHEON_PATTERN.search(text)
    if match:
        man = int(match.group(1))
        cheon = int(match.group(2)) if match.group(2) else 0
        return man * 10_000 + cheon * 1_000

    match = _PLAIN_WON_PATTERN.search(text)
    if match:
        return int(match.group(1).replace(",", ""))

    return None
