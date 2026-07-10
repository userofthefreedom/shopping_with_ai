from unittest.mock import patch

import numpy as np
import pytest

from search import text_search
from search.text_search import (
    embed_text,
    index_product_texts,
    is_local_search_sufficient,
    search_similar_text,
)


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(text_search, "_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(text_search, "_client", None)
    monkeypatch.setattr(text_search, "_collection", None)
    yield


def test_embed_text_returns_normalized_768d_vector():
    embedding = embed_text("빨간색 반팔 셔츠")

    assert embedding.shape == (768,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-3


def test_index_product_texts_and_search_similar_text_orders_by_similarity(isolated_chroma):
    products = [
        {
            "name": "빨간색 반팔 셔츠",
            "price": 10000,
            "image_url": "https://example.com/red.jpg",
            "purchase_url": "https://example.com/item/red",
        },
        {
            "name": "긴바지 청바지",
            "price": 20000,
            "image_url": "https://example.com/jeans.jpg",
            "purchase_url": "https://example.com/item/jeans",
        },
    ]

    indexed_count = index_product_texts(products, category="short_sleeved_shirt", color="빨간")

    assert indexed_count == 2

    results = search_similar_text("빨간 티셔츠", top_k=2)

    assert len(results) == 2
    assert results[0]["name"] == "빨간색 반팔 셔츠"
    assert results[0]["source"] == "naver"
    assert 0.0 <= results[0]["similarity"] <= 1.0
    assert results[0]["similarity"] > results[1]["similarity"]


def test_index_product_texts_skips_products_with_empty_purchase_url(isolated_chroma):
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

    indexed_count = index_product_texts(products, category="short_sleeved_shirt", color="빨간")

    assert indexed_count == 1  # 빈 purchase_url 상품 2건은 제외(id 충돌 방지)


def test_index_product_texts_skips_already_indexed_purchase_url(isolated_chroma):
    product = {
        "name": "빨간색 반팔 셔츠",
        "price": 10000,
        "image_url": "https://example.com/red.jpg",
        "purchase_url": "https://example.com/item/red",
    }

    with patch("search.text_search.embed_text", wraps=embed_text) as mock_embed:
        first_count = index_product_texts(
            [product], category="short_sleeved_shirt", color="빨간"
        )
        assert first_count == 1
        assert mock_embed.call_count == 1

        second_count = index_product_texts(
            [product], category="short_sleeved_shirt", color="빨간"
        )
        assert second_count == 1
        assert mock_embed.call_count == 1  # 재임베딩 없음


def test_search_similar_text_returns_empty_list_when_collection_empty(isolated_chroma):
    assert search_similar_text("빨간 티셔츠", top_k=5) == []


def test_is_local_search_sufficient_requires_min_count_and_similarity():
    high_sim_results = [{"similarity": 0.9}] * text_search._LOCAL_SEARCH_MIN_COUNT

    assert is_local_search_sufficient(high_sim_results) is True
    assert is_local_search_sufficient(high_sim_results[:-1]) is False  # 개수 부족

    low_sim_results = [{"similarity": 0.1}] * text_search._LOCAL_SEARCH_MIN_COUNT
    assert is_local_search_sufficient(low_sim_results) is False  # 유사도 부족
