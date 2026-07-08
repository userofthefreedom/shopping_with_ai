"""네이버 쇼핑 검색 API 연동."""

import logging
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://openapi.naver.com/v1/search/shop.json"
_TAG_PATTERN = re.compile(r"</?b>")


class NaverAPIError(Exception):
    """네이버 쇼핑 API 호출 실패 (키 누락, HTTP 오류, 타임아웃 등)."""


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
    except requests.RequestException:
        logger.exception("네이버 쇼핑 API 호출 실패 (query=%s)", query)
        raise NaverAPIError("상품 정보를 가져오지 못했습니다.")

    items = response.json().get("items", [])
    return [
        {
            "name": _TAG_PATTERN.sub("", item.get("title", "")),
            "price": int(item.get("lprice", 0)),
            "image_url": item.get("image", ""),
            "purchase_url": item.get("link", ""),
            "source": item.get("mallName", "naver"),
        }
        for item in items
    ]
