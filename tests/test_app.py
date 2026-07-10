from unittest.mock import patch

from PIL import Image

import app
from search import text_search
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


def test_build_recognition_badge_escapes_description():
    result = app._build_recognition_badge("<script>alert(1)</script>")

    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert 'class="recognition-badge"' in result


def test_build_recognition_badge_empty_description_returns_empty_string():
    assert app._build_recognition_badge("") == ""
    assert app._build_recognition_badge(None) == ""


def test_build_warning_banner_empty_returns_empty_string():
    assert app._build_warning_banner([]) == ""


def test_build_warning_banner_escapes_and_wraps_each_warning():
    result = app._build_warning_banner(["<script>alert(1)</script>", "두번째 경고"])

    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert result.count("warning-item") == 2
    assert "두번째 경고" in result


def test_build_product_cards_escapes_name_and_source():
    products = [
        {
            "name": "<script>alert(1)</script>",
            "price": 10000,
            "image_url": "https://example.com/a.jpg",
            "purchase_url": "https://example.com/item/a",
            "source": "<img src=x onerror=alert(1)>",
        }
    ]

    result = app._build_product_cards(products)

    assert "<script>" not in result
    assert "<img src=x" not in result
    assert "&lt;script&gt;" in result
    assert "&lt;img src=x onerror=alert(1)&gt;" in result


def test_build_product_cards_rejects_javascript_scheme_url():
    products = [
        {
            "name": "정상 상품",
            "price": 10000,
            "image_url": "https://example.com/a.jpg",
            "purchase_url": "javascript:alert(1)",
            "source": "naver",
        }
    ]

    result = app._build_product_cards(products)

    assert "javascript:" not in result
    assert "정상 상품" not in result  # 스킴이 안전하지 않아 카드 자체를 생략


def test_build_product_cards_empty_products_shows_placeholder():
    result = app._build_product_cards([])

    assert "product-empty" in result


def test_build_product_cards_renders_grid_with_image_name_and_price():
    products = _sample_products()

    result = app._build_product_cards(products)

    assert result.count("product-card") >= 2
    assert 'src="https://example.com/a.jpg"' in result
    assert "A 셔츠" in result
    assert "50,000원" in result


def test_is_freshness_request_detects_keywords():
    assert app._is_freshness_request("최신 상품 있어?") is True
    assert app._is_freshness_request("오늘 신상 뭐 있어") is True
    assert app._is_freshness_request("이거 뭐야?") is False
    assert app._is_freshness_request("10만원 이하로 보여줘") is False


def test_empty_state_returns_fresh_history_list_each_call():
    first = app._empty_state()
    second = app._empty_state()

    assert first["history"] is not second["history"]

    first["history"].append(("이거 뭐야?", "답변"))

    assert second["history"] == []


def test_on_image_upload_no_detection_shows_message_and_resets_state():
    with patch("app.detect_products", return_value=[]):
        bbox_image, info_html, warning, products_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "인식된 상품이 없습니다" in warning
    assert "warning-banner" in warning
    assert info_html == ""
    assert "product-empty" in products_html
    assert state["candidate_products"] == []
    assert chatbot[0]["role"] == "assistant"


def test_on_image_upload_handles_unexpected_exception_in_recognition_step():
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", side_effect=ValueError("bbox 영역이 비어 있습니다")),
    ):
        bbox_image, info_html, warning, products_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "문제가 발생했습니다" in warning
    assert info_html == ""
    assert state["candidate_products"] == []
    assert chatbot[0]["role"] == "assistant"


def test_on_image_upload_happy_path_builds_cards_and_state():
    products = _sample_products()
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.classify_subtype", return_value=None),
        patch("app.search_similar_text", return_value=[]),
        patch("app.search_naver_variants", return_value=products),
        patch("app.index_products", return_value=2) as mock_index,
        patch("app.index_product_texts", return_value=2) as mock_index_text,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=products),
    ):
        bbox_image, info_html, warning, products_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "빨간색 반팔 셔츠" in info_html
    assert "recognition-badge" in info_html
    assert warning == ""
    assert mock_index.called
    assert mock_index_text.called
    assert "A 셔츠" in products_html
    assert "B 셔츠" in products_html
    assert "구매하러 가기" in products_html
    assert state["candidate_products"] == products
    assert state["color"] == "빨간"


def test_on_image_upload_uses_local_text_search_and_skips_naver_when_sufficient():
    local_results = [
        {
            "name": f"로컬 상품 {i}",
            "price": 10000 + i,
            "image_url": f"https://example.com/{i}.jpg",
            "purchase_url": f"https://example.com/item/{i}",
            "source": "naver",
            "similarity": 0.95,
        }
        for i in range(text_search._LOCAL_SEARCH_MIN_COUNT)
    ]
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.classify_subtype", return_value=None),
        patch("app.search_similar_text", return_value=local_results),
        patch("app.search_naver_variants") as mock_search_naver_variants,
        patch("app.index_products") as mock_index,
        patch("app.index_product_texts") as mock_index_text,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=local_results),
    ):
        _, info_html, warning, products_html, state, _ = app.on_image_upload(_sample_image())

    assert not mock_search_naver_variants.called  # 로컬 검색이 충분해 네이버 호출 스킵
    assert not mock_index.called  # 이미 색인된 결과라 재색인 불필요
    assert not mock_index_text.called
    assert state["candidate_products"] == local_results


def test_on_image_upload_naver_failure_continues_gracefully():
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.classify_subtype", return_value=None),
        patch("app.search_similar_text", return_value=[]),
        patch("app.search_naver_variants", side_effect=NaverAPIError("boom")),
        patch("app.index_products") as mock_index,
        patch("app.index_product_texts") as mock_index_text,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=[]),
    ):
        bbox_image, info_html, warning, products_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "상품 정보를 가져오지 못했습니다" in warning
    assert "유사 상품을 찾지 못했습니다" in warning
    assert not mock_index.called
    assert not mock_index_text.called
    assert "product-empty" in products_html
    assert state["candidate_products"] == []


def test_on_image_upload_no_similar_products_shows_message():
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.classify_subtype", return_value=None),
        patch("app.search_similar_text", return_value=[]),
        patch("app.search_naver_variants", return_value=[]),
        patch("app.index_products") as mock_index,
        patch("app.index_product_texts") as mock_index_text,
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=[]),
    ):
        _, info_html, warning, products_html, state, _ = app.on_image_upload(_sample_image())

    assert "유사 상품을 찾지 못했습니다" in warning
    assert "상품 정보를 가져오지 못했습니다" not in warning
    assert not mock_index.called
    assert not mock_index_text.called
    assert "product-empty" in products_html


def test_on_image_upload_uses_subtype_in_description_when_classified():
    products = _sample_products()
    detection = {"bbox": (10, 10, 60, 60), "category": "long_sleeved_outwear", "confidence": 0.9}
    with (
        patch("app.detect_products", return_value=[detection]),
        patch("app.detect_color", return_value="남색"),
        patch("app.classify_subtype", return_value="패딩"),
        patch("app.search_similar_text", return_value=[]),
        patch("app.search_naver_variants", return_value=products) as mock_search_naver_variants,
        patch("app.index_products", return_value=2),
        patch("app.index_product_texts", return_value=2),
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=products),
    ):
        _, info_html, warning, products_html, state, chatbot = app.on_image_upload(
            _sample_image()
        )

    assert "남색 패딩" in info_html
    assert state["subtype"] == "패딩"
    mock_search_naver_variants.assert_called_once_with(detection["category"], "남색", "패딩")


def test_on_image_upload_reranks_scoped_products_by_visual_similarity():
    products = _sample_products()  # a, b
    irrelevant_global_hit = {
        "name": "무관한 전역 상품",
        "price": 5000,
        "image_url": "https://example.com/irrelevant.jpg",
        "purchase_url": "https://example.com/item/irrelevant",
        "source": "naver",
    }
    # 전역 컬렉션 검색 결과: 무관한 상품이 먼저, 스코핑된 b가 그다음, a는 없음
    global_similar = [irrelevant_global_hit, products[1]]
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.classify_subtype", return_value=None),
        patch("app.search_similar_text", return_value=[]),
        patch("app.search_naver_variants", return_value=products),
        patch("app.index_products", return_value=2),
        patch("app.index_product_texts", return_value=2),
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=global_similar),
    ):
        _, _, warning, products_html, state, _ = app.on_image_upload(_sample_image())

    # 무관한 전역 상품은 제외되고, 스코핑된 b만 남는다 (a는 전역 검색에 없어 탈락)
    assert warning == ""
    assert [p["purchase_url"] for p in state["candidate_products"]] == [
        "https://example.com/item/b"
    ]
    assert "B 셔츠" in products_html
    assert "무관한 전역 상품" not in products_html
    assert "A 셔츠" not in products_html


def test_on_image_upload_falls_back_to_scoped_when_visual_intersection_empty():
    products = _sample_products()  # a, b
    unrelated_global_hits = [
        {
            "name": "무관한 전역 상품",
            "price": 5000,
            "image_url": "https://example.com/irrelevant.jpg",
            "purchase_url": "https://example.com/item/irrelevant",
            "source": "naver",
        }
    ]
    with (
        patch("app.detect_products", return_value=[_sample_detection()]),
        patch("app.detect_color", return_value="빨간"),
        patch("app.classify_subtype", return_value=None),
        patch("app.search_similar_text", return_value=[]),
        patch("app.search_naver_variants", return_value=products),
        patch("app.index_products", return_value=2),
        patch("app.index_product_texts", return_value=2),
        patch("app.embed_image", return_value=object()),
        patch("app.search_similar", return_value=unrelated_global_hits),
    ):
        _, _, warning, products_html, state, _ = app.on_image_upload(_sample_image())

    # 교집합이 비므로 스코핑된 naver 결과(products) 그대로 사용
    assert warning == ""
    assert [p["purchase_url"] for p in state["candidate_products"]] == [
        "https://example.com/item/a",
        "https://example.com/item/b",
    ]
    assert "A 셔츠" in products_html
    assert "B 셔츠" in products_html


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
        chatbot_messages, cleared_input, new_state, products_html = app.on_chat_submit(
            "10만원 이하로 보여줘", [], state
        )

    assert cleared_input == ""
    assert [p["name"] for p in captured_context["candidate_products"]] == ["A 셔츠"]
    assert [p["name"] for p in new_state["candidate_products"]] == ["A 셔츠"]
    assert "A 셔츠" in products_html
    assert "B 셔츠" not in products_html
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
        chatbot_messages, _, new_state, _ = app.on_chat_submit("이거 뭐야?", [], state)

    assert chatbot_messages[-1] == {
        "role": "assistant",
        "content": "죄송해요, 답변을 생성하지 못했습니다.",
    }
    assert new_state["history"][-1] == ("이거 뭐야?", "죄송해요, 답변을 생성하지 못했습니다.")


def test_on_chat_submit_freshness_keyword_triggers_live_refresh():
    state = {
        "detected_item": _sample_detection(),
        "color": "빨간",
        "subtype": None,
        "candidate_products": _sample_products(),
        "history": [],
    }
    fresh_products = [
        {
            "name": "신상 셔츠",
            "price": 99000,
            "image_url": "https://example.com/new.jpg",
            "purchase_url": "https://example.com/item/new",
            "source": "네이버",
        }
    ]

    with (
        patch("app.search_naver_variants", return_value=fresh_products) as mock_search_naver_variants,
        patch("app.index_products", return_value=1) as mock_index,
        patch("app.index_product_texts", return_value=1) as mock_index_text,
        patch("app.generate_response", return_value="최신 상품으로 갱신했어요."),
    ):
        chatbot_messages, cleared_input, new_state, products_html = app.on_chat_submit(
            "최신 상품 있어?", [], state
        )

    assert mock_search_naver_variants.called
    assert mock_index.called
    assert mock_index_text.called
    assert new_state["candidate_products"] == fresh_products
    assert "신상 셔츠" in products_html


def test_on_chat_submit_freshness_refresh_failure_logs_exception():
    state = {
        "detected_item": _sample_detection(),
        "color": "빨간",
        "subtype": None,
        "candidate_products": _sample_products(),
        "history": [],
    }

    with (
        patch("app.search_naver_variants", side_effect=NaverAPIError("boom")),
        patch("app.generate_response", return_value="응답"),
        patch("app.logger") as mock_logger,
    ):
        app.on_chat_submit("최신 상품 있어?", [], state)

    assert mock_logger.exception.called


def test_on_chat_submit_freshness_refresh_failure_notifies_user_in_response():
    state = {
        "detected_item": _sample_detection(),
        "color": "빨간",
        "subtype": None,
        "candidate_products": _sample_products(),
        "history": [],
    }

    with (
        patch("app.search_naver_variants", side_effect=NaverAPIError("boom")),
        patch("app.generate_response", return_value="42,800원입니다."),
    ):
        chatbot_messages, _, _, _ = app.on_chat_submit("최신 상품 있어?", [], state)

    assert "실시간 재고 확인에 실패" in chatbot_messages[-1]["content"]


def test_on_chat_submit_without_freshness_keyword_does_not_call_naver():
    state = {
        "detected_item": _sample_detection(),
        "color": "빨간",
        "subtype": None,
        "candidate_products": _sample_products(),
        "history": [],
    }

    with (
        patch("app.search_naver_variants") as mock_search_naver_variants,
        patch("app.generate_response", return_value="42,800원입니다."),
    ):
        app.on_chat_submit("얼마야?", [], state)

    assert not mock_search_naver_variants.called
