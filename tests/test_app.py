from unittest.mock import patch

from PIL import Image

import app
from search.naver_api import NaverAPIError


def _sample_image():
    return Image.new("RGB", (100, 100), color=(200, 30, 30))


def _sample_detection():
    return {"bbox": (10, 10, 60, 60), "category": "short_sleeved_shirt", "confidence": 0.9}


def _sample_products():
    return [
        {
            "name": "A 셔츠",
            "price": 50000,
            "image_url": "https://example.com/a.jpg",
            "purchase_url": "https://example.com/item/a",
            "source": "네이버",
        },
        {
            "name": "B 셔츠",
            "price": 150000,
            "image_url": "https://example.com/b.jpg",
            "purchase_url": "https://example.com/item/b",
            "source": "네이버",
        },
    ]


def test_lock_chat_input_for_analysis_disables_and_sets_placeholder():
    update = app._lock_chat_input_for_analysis()

    assert update["interactive"] is False
    assert update["placeholder"] == app._ANALYZING_PLACEHOLDER


def test_lock_chat_input_for_response_disables_and_sets_placeholder():
    update = app._lock_chat_input_for_response()

    assert update["interactive"] is False
    assert update["placeholder"] == app._THINKING_PLACEHOLDER


def test_unlock_chat_input_reenables_and_restores_placeholder():
    update = app._unlock_chat_input()

    assert update["interactive"] is True
    assert update["placeholder"] == app._CHAT_PLACEHOLDER


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

    with patch("app.search_naver", side_effect=fake_search_naver):
        merged = app._search_naver_variants("short_sleeved_shirt", "파란")

    assert calls == ["파란색 반팔 셔츠", "파란색 반팔 티셔츠"]
    assert [p["purchase_url"] for p in merged] == [
        "https://example.com/item/shirt",
        "https://example.com/item/tee",
    ]


def test_on_image_upload_no_detection_shows_message_and_resets_state():
    with patch("app.detect_products", return_value=[]):
        bbox_image, text, warning, gallery, links_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "인식된 상품이 없습니다" in warning
    assert gallery == []
    assert links_html == ""
    assert state["candidate_products"] == []
    assert chatbot[0]["role"] == "assistant"


def test_on_image_upload_happy_path_builds_gallery_and_state():
    products = _sample_products()
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.search_naver", return_value=products),
        patch("app.index_products", return_value=2) as mock_index,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=products),
    ):
        bbox_image, text, warning, gallery, links_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "빨간색 반팔 셔츠" in text
    assert warning == ""
    assert mock_index.called
    assert gallery == [
        ("https://example.com/a.jpg", "A 셔츠 - 50,000원"),
        ("https://example.com/b.jpg", "B 셔츠 - 150,000원"),
    ]
    assert "구매하러 가기" in links_html
    assert state["candidate_products"] == products
    assert state["color"] == "빨간"


def test_on_image_upload_naver_failure_continues_gracefully():
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.search_naver", side_effect=NaverAPIError("boom")),
        patch("app.index_products") as mock_index,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=[]),
    ):
        bbox_image, text, warning, gallery, links_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "상품 정보를 가져오지 못했습니다" in warning
    assert "유사 상품을 찾지 못했습니다" in warning
    assert not mock_index.called
    assert gallery == []
    assert state["candidate_products"] == []


def test_on_image_upload_no_similar_products_shows_message():
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.search_naver", return_value=[]),
        patch("app.index_products") as mock_index,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=[]),
    ):
        _, text, warning, gallery, links_html, state, _ = app.on_image_upload(_sample_image())

    assert "유사 상품을 찾지 못했습니다" in warning
    assert "상품 정보를 가져오지 못했습니다" not in warning
    assert not mock_index.called
    assert gallery == []


def test_on_chat_submit_applies_budget_filter():
    state = {
        "detected_item": _sample_detection(),
        "color": "빨간",
        "candidate_products": _sample_products(),
        "history": [],
    }
    captured_context = {}

    def fake_generate_response(user_message, context):
        captured_context.update(context)
        return "네, 더 저렴한 상품을 알려드릴게요."

    with patch("app.generate_response", side_effect=fake_generate_response):
        chatbot_messages, cleared_input, new_state, gallery, links_html = app.on_chat_submit(
            "10만원 이하로 보여줘", [], state
        )

    assert cleared_input == ""
    assert [p["name"] for p in captured_context["candidate_products"]] == ["A 셔츠"]
    assert [p["name"] for p in new_state["candidate_products"]] == ["A 셔츠"]
    assert gallery == [("https://example.com/a.jpg", "A 셔츠 - 50,000원")]
    assert chatbot_messages[-2] == {"role": "user", "content": "10만원 이하로 보여줘"}
    assert chatbot_messages[-1]["role"] == "assistant"


def test_on_chat_submit_generate_response_failure_falls_back():
    state = {
        "detected_item": _sample_detection(),
        "color": "빨간",
        "candidate_products": _sample_products(),
        "history": [],
    }

    with patch("app.generate_response", side_effect=RuntimeError("model crashed")):
        chatbot_messages, _, new_state, _, _ = app.on_chat_submit("이거 뭐야?", [], state)

    assert chatbot_messages[-1] == {
        "role": "assistant",
        "content": "죄송해요, 답변을 생성하지 못했습니다.",
    }
    assert new_state["history"][-1] == ("이거 뭐야?", "죄송해요, 답변을 생성하지 못했습니다.")
