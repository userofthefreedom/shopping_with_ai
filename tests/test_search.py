import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from PIL import Image

from search import clip_search
from search.clip_search import classify_subtype, embed_image, index_products, search_similar
from search.naver_api import NaverAPIError, search_naver, search_naver_variants

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


def test_search_naver_raises_on_malformed_json(monkeypatch):
    _fake_env(monkeypatch)
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.side_effect = ValueError("Expecting value: line 1 column 1")

    with patch("search.naver_api.requests.get", return_value=mock_response):
        with pytest.raises(NaverAPIError):
            search_naver("빨간색 반팔 셔츠")


def test_search_naver_treats_blank_lprice_as_zero(monkeypatch):
    _fake_env(monkeypatch)
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "items": [
            {
                "title": "가격 미정 상품",
                "link": "https://example.com/item/1",
                "image": "https://example.com/1.jpg",
                "lprice": "",
                "mallName": "네이버",
            }
        ]
    }

    with patch("search.naver_api.requests.get", return_value=mock_response):
        products = search_naver("빨간색 반팔 셔츠")

    assert products[0]["price"] == 0


def test_search_naver_skips_malformed_item_but_keeps_other_items(monkeypatch):
    _fake_env(monkeypatch)
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "items": [
            {
                "title": None,  # _TAG_PATTERN.sub()가 TypeError를 던짐
                "link": "https://example.com/item/broken",
                "image": "https://example.com/broken.jpg",
                "lprice": "10000",
                "mallName": "네이버",
            },
            {
                "title": "정상 상품",
                "link": "https://example.com/item/ok",
                "image": "https://example.com/ok.jpg",
                "lprice": "20000",
                "mallName": "네이버",
            },
        ]
    }

    with patch("search.naver_api.requests.get", return_value=mock_response):
        products = search_naver("빨간색 반팔 셔츠")

    assert len(products) == 1  # 잘못된 항목 1건은 건너뛰고 정상 항목만 반환
    assert products[0]["name"] == "정상 상품"


def test_search_naver_variants_keeps_successful_synonyms_when_one_fails():
    def fake_search_naver(query):
        if "티셔츠" in query:
            raise NaverAPIError("타임아웃")
        return [
            {
                "name": "셔츠 상품",
                "price": 20000,
                "image_url": "https://example.com/shirt.jpg",
                "purchase_url": "https://example.com/item/shirt",
                "source": "네이버",
            }
        ]

    with patch("search.naver_api.search_naver", side_effect=fake_search_naver):
        merged = search_naver_variants("short_sleeved_shirt", "파란")

    # "반팔 티셔츠" 검색만 실패해도 "반팔 셔츠" 검색 결과는 유실되지 않는다
    assert [p["purchase_url"] for p in merged] == ["https://example.com/item/shirt"]


def test_search_naver_variants_merges_synonym_queries_deduped():
    calls = []

    def fake_search_naver(query):
        calls.append(query)
        if "티셔츠" in query:
            return [
                {
                    "name": "티셔츠 상품",
                    "price": 10000,
                    "image_url": "https://example.com/tee.jpg",
                    "purchase_url": "https://example.com/item/tee",
                    "source": "네이버",
                }
            ]
        return [
            {
                "name": "셔츠 상품",
                "price": 20000,
                "image_url": "https://example.com/shirt.jpg",
                "purchase_url": "https://example.com/item/shirt",
                "source": "네이버",
            },
            {
                "name": "중복 상품(티셔츠 검색에서도 잡힘)",
                "price": 10000,
                "image_url": "https://example.com/tee.jpg",
                "purchase_url": "https://example.com/item/tee",
                "source": "네이버",
            },
        ]

    with patch("search.naver_api.search_naver", side_effect=fake_search_naver):
        merged = search_naver_variants("short_sleeved_shirt", "파란")

    assert sorted(calls) == sorted(["파란색 반팔 셔츠", "파란색 반팔 티셔츠"])
    assert [p["purchase_url"] for p in merged] == [
        "https://example.com/item/shirt",
        "https://example.com/item/tee",
    ]


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_search, "_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(clip_search, "_client", None)
    monkeypatch.setattr(clip_search, "_collection", None)
    yield


def test_embed_image_returns_normalized_512d_vector():
    image = Image.new("RGB", (64, 64), color=(100, 150, 200))

    embedding = embed_image(image)

    assert embedding.shape == (512,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-3


def test_index_products_and_search_similar_orders_by_similarity(isolated_chroma):
    products = [
        {
            "name": "레드 셔츠",
            "price": 10000,
            "image_url": "https://example.com/red.jpg",
            "purchase_url": "https://example.com/item/red",
        },
        {
            "name": "블루 셔츠",
            "price": 20000,
            "image_url": "https://example.com/blue.jpg",
            "purchase_url": "https://example.com/item/blue",
        },
    ]

    def fake_get(url, timeout=5):
        color = (220, 20, 20) if "red" in url else (20, 20, 220)
        response = Mock()
        response.raise_for_status = Mock()
        response.content = _png_bytes(color)
        return response

    with patch("search.clip_search.requests.get", side_effect=fake_get):
        indexed_count = index_products(products, category="short_sleeved_shirt", color="빨간")

    assert indexed_count == 2

    query_embedding = embed_image(Image.new("RGB", (64, 64), color=(200, 30, 30)))
    results = search_similar(query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0]["name"] == "레드 셔츠"
    assert results[0]["source"] == "naver"


def test_search_similar_returns_empty_list_when_collection_empty(isolated_chroma):
    query_embedding = embed_image(Image.new("RGB", (64, 64), color=(128, 128, 128)))

    assert search_similar(query_embedding, top_k=5) == []


def test_classify_subtype_returns_none_for_category_without_candidates():
    image = Image.new("RGB", (64, 64), color=(20, 20, 60))

    assert classify_subtype(image, "unknown_category") is None


def test_classify_subtype_returns_none_when_confidence_below_threshold():
    image = Image.new("RGB", (64, 64), color=(20, 20, 60))

    assert classify_subtype(image, "long_sleeved_outwear", min_confidence=0.99) is None


def test_classify_subtype_returns_one_of_candidate_labels_when_threshold_disabled():
    image = Image.new("RGB", (64, 64), color=(20, 20, 60))

    result = classify_subtype(image, "long_sleeved_outwear", min_confidence=0.0)

    expected_labels = {ko for _, ko in clip_search._SUBTYPE_CANDIDATES["long_sleeved_outwear"]}
    assert result in expected_labels


def test_index_products_skips_products_with_empty_purchase_url(isolated_chroma):
    products = [
        {
            "name": "정상 상품",
            "price": 10000,
            "image_url": "https://example.com/red.jpg",
            "purchase_url": "https://example.com/item/red",
        },
        {
            "name": "구매링크 없는 상품 1",
            "price": 20000,
            "image_url": "https://example.com/a.jpg",
            "purchase_url": "",
        },
        {
            "name": "구매링크 없는 상품 2",
            "price": 30000,
            "image_url": "https://example.com/b.jpg",
            "purchase_url": "",
        },
    ]

    def fake_get(url, timeout=5):
        response = Mock()
        response.raise_for_status = Mock()
        response.content = _png_bytes((220, 20, 20))
        return response

    with patch("search.clip_search.requests.get", side_effect=fake_get):
        indexed_count = index_products(products, category="short_sleeved_shirt", color="빨간")

    assert indexed_count == 1  # 빈 purchase_url 상품 2건은 제외


def test_index_products_skips_already_indexed_purchase_url(isolated_chroma):
    product = {
        "name": "레드 셔츠",
        "price": 10000,
        "image_url": "https://example.com/red.jpg",
        "purchase_url": "https://example.com/item/red",
    }

    def fake_get(url, timeout=5):
        response = Mock()
        response.raise_for_status = Mock()
        response.content = _png_bytes((220, 20, 20))
        return response

    with patch("search.clip_search.requests.get", side_effect=fake_get) as mock_get:
        first_count = index_products([product], category="short_sleeved_shirt", color="빨간")
        assert first_count == 1
        assert mock_get.call_count == 1

        second_count = index_products([product], category="short_sleeved_shirt", color="빨간")
        assert second_count == 1
        assert mock_get.call_count == 1  # 재다운로드 없음
