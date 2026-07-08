from chat.llm import _build_system_prompt


def test_build_system_prompt_includes_detected_item_and_color():
    context = {
        "detected_item": {"category": "short_sleeved_shirt", "confidence": 0.9},
        "color": "빨간",
        "candidate_products": [],
        "history": [],
    }

    prompt = _build_system_prompt(context)

    assert "빨간색 반팔 셔츠" in prompt
    assert "없음" in prompt  # 후보 상품 없음 안내


def test_build_system_prompt_lists_candidate_products_with_price():
    context = {
        "detected_item": {"category": "skirt", "confidence": 0.8},
        "color": "검정",
        "candidate_products": [
            {"name": "A 치마", "price": 30000, "source": "네이버"},
            {"name": "B 치마", "price": 45000, "source": "쇼핑몰"},
        ],
        "history": [],
    }

    prompt = _build_system_prompt(context)

    assert "A 치마 - 30,000원 (네이버)" in prompt
    assert "B 치마 - 45,000원 (쇼핑몰)" in prompt


def test_build_system_prompt_handles_no_detected_item():
    context = {"detected_item": None, "color": None, "candidate_products": [], "history": []}

    prompt = _build_system_prompt(context)

    assert "인식된 상품: 없음" in prompt
