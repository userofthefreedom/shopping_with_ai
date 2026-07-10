"""네이버 쇼핑 검색 API 연동."""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

from detection.color import search_query_terms

load_dotenv()

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://openapi.naver.com/v1/search/shop.json"
_TAG_PATTERN = re.compile(r"</?b>")


class NaverAPIError(Exception):
    """네이버 쇼핑 API 호출 실패 (키 누락, HTTP 오류, 타임아웃, 응답 파싱 실패 등)."""


def search_naver(query: str, display: int = 20) -> list[dict]:
    """상품명으로 네이버 쇼핑 API를 검색해 Product 리스트를 반환한다.

    Product = {name, price, image_url, purchase_url, source}
    """
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise NaverAPIError(
            "NAVER_CLIENT_ID/NAVER_CLIENT_SECRET이 설정되지 않았습니다. .env 파일을 확인하세요."
        )

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": display, "sort": "sim"}

    try:
        response = requests.get(_SEARCH_URL, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        items = response.json().get("items", [])
    except requests.RequestException:
        logger.exception("네이버 쇼핑 API 호출 실패 (query=%s)", query)
        raise NaverAPIError("상품 정보를 가져오지 못했습니다.")
    except (ValueError, TypeError, KeyError):
        # response.json() 파싱 실패(잘못된 JSON) 등 응답 전체가 기대와 다른
        # 경우 — 조용히 넘기지 않고 명시적으로 실패 처리한다.
        logger.exception("네이버 쇼핑 API 응답 파싱 실패 (query=%s)", query)
        raise NaverAPIError("상품 정보를 가져오지 못했습니다.")

    products = []
    for item in items:
        try:
            products.append(_parse_item(item))
        except (ValueError, TypeError, KeyError):
            # 항목 하나가 예상과 다른 형태(title/lprice가 null 등)라고 해서
            # 이미 정상 파싱된 나머지 항목까지 통째로 버리지 않는다.
            logger.exception("네이버 쇼핑 API 항목 파싱 실패, 건너뜀 (query=%s)", query)
    return products


def _parse_item(item: dict) -> dict:
    lprice = item.get("lprice") or 0
    return {
        "name": _TAG_PATTERN.sub("", item.get("title", "")),
        "price": int(lprice),
        "image_url": item.get("image", ""),
        "purchase_url": item.get("link", ""),
        "source": item.get("mallName", "naver"),
    }


def search_naver_variants(category: str, color: str, subtype: str | None = None) -> list[dict]:
    """카테고리 동의어(+CLIP 추정 subtype)별로 네이버 검색을 병렬로 수행해
    결과를 구매링크 기준으로 합친다.

    DeepFashion2 카테고리 하나(예: short_sleeved_shirt)가 와이셔츠부터
    그래픽 티셔츠까지 아우르는데, 번역어 하나로만 검색하면 네이버 랭킹이
    한쪽(주로 정장 셔츠)으로 쏠려 실제로 다른 스타일인 상품을 후보군에서
    놓친다(Phase 10). 동의어를 함께 검색해 후보군을 넓히되, 검색어별 호출을
    병렬화해 순차 호출로 인한 응답 지연(Phase 12에서 측정된 cold 경로 지연의
    주 원인)을 줄인다(Phase 13).
    """
    color_ko = color if color.endswith("색") else f"{color}색"
    terms = search_query_terms(category, subtype)

    with ThreadPoolExecutor(max_workers=min(8, len(terms))) as executor:
        future_to_term = {
            executor.submit(search_naver, f"{color_ko} {term}"): term for term in terms
        }
        results_per_term = []
        for future in future_to_term:
            try:
                results_per_term.append(future.result())
            except NaverAPIError:
                # 동의어 중 하나의 검색이 실패해도 나머지 동의어의 성공한
                # 결과까지 통째로 버리지 않는다.
                logger.exception(
                    "동의어 검색 실패, 해당 검색어만 건너뜀 (term=%s)", future_to_term[future]
                )

    seen_urls = set()
    merged = []
    for products in results_per_term:
        for product in products:
            if product["purchase_url"] in seen_urls:
                continue
            seen_urls.add(product["purchase_url"])
            merged.append(product)
    return merged
