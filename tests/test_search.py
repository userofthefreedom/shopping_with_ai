from unittest.mock import Mock, patch

import pytest
import requests

from search.naver_api import NaverAPIError, search_naver

_FAKE_RESPONSE_JSON = {
    "items": [
        {
            "title": "네이비 <b>레드</b> 체크 <b>반팔 셔츠</b>",
            "link": "https://smartstore.naver.com/main/products/1",
            "image": "https://shopping-phinf.pstatic.net/main_1/1.jpg",
            "lprice": "33200",
            "mallName": "아리킴",
        },
        {
            "title": "빈폴레이디스 <b>반소매 셔츠</b>",
            "link": "https://search.shopping.naver.com/catalog/2",
            "image": "https://shopping-phinf.pstatic.net/main_2/2.jpg",
            "lprice": "106570",
            "mallName": "네이버",
        },
    ]
}


def _fake_env(monkeypatch):
    monkeypatch.setenv("NAVER_CLIENT_ID", "test-id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "test-secret")


def test_search_naver_parses_response_into_products(monkeypatch):
    _fake_env(monkeypatch)
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = _FAKE_RESPONSE_JSON

    with patch("search.naver_api.requests.get", return_value=mock_response) as mock_get:
        products = search_naver("빨간색 반팔 셔츠")

    assert mock_get.called
    assert products == [
        {
            "name": "네이비 레드 체크 반팔 셔츠",
            "price": 33200,
            "image_url": "https://shopping-phinf.pstatic.net/main_1/1.jpg",
            "purchase_url": "https://smartstore.naver.com/main/products/1",
            "source": "아리킴",
        },
        {
            "name": "빈폴레이디스 반소매 셔츠",
            "price": 106570,
            "image_url": "https://shopping-phinf.pstatic.net/main_2/2.jpg",
            "purchase_url": "https://search.shopping.naver.com/catalog/2",
            "source": "네이버",
        },
    ]


def test_search_naver_raises_when_credentials_missing(monkeypatch):
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)

    with pytest.raises(NaverAPIError):
        search_naver("빨간색 반팔 셔츠")


def test_search_naver_raises_on_http_error(monkeypatch):
    _fake_env(monkeypatch)
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 error")

    with patch("search.naver_api.requests.get", return_value=mock_response):
        with pytest.raises(NaverAPIError):
            search_naver("빨간색 반팔 셔츠")


def test_search_naver_raises_on_timeout(monkeypatch):
    _fake_env(monkeypatch)

    with patch("search.naver_api.requests.get", side_effect=requests.Timeout("timed out")):
        with pytest.raises(NaverAPIError):
            search_naver("빨간색 반팔 셔츠")
