from unittest.mock import patch

import numpy as np
import pytest

from search import text_search
from search.text_search import embed_text, index_product_texts, search_similar_text


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
