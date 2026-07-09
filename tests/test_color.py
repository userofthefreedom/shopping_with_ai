import pytest
from PIL import Image

from detection.color import describe_item, detect_color, search_query_terms


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ((255, 0, 0), "빨간"),
        ((0, 255, 0), "초록"),
        ((0, 0, 255), "파란"),
        ((255, 255, 0), "노란"),
        ((0, 0, 0), "검정"),
        ((255, 255, 255), "흰색"),
        ((128, 128, 128), "회색"),
        ((101, 67, 33), "갈색"),
        ((25, 25, 70), "남색"),  # 어두운 네이비: hue만으론 "파란"과 같은 대역
        ((75, 82, 74), "초록"),  # 카키/국방색: 실제 사진에서 채도가 낮게 측정됨
    ],
)
def test_detect_color_maps_solid_patch_to_korean_name(rgb, expected):
    image = Image.new("RGB", (40, 40), color=rgb)

    color = detect_color(image, (0, 0, 40, 40))

    assert color == expected


def test_detect_color_raises_for_empty_bbox():
    image = Image.new("RGB", (40, 40), color=(255, 0, 0))

    with pytest.raises(ValueError):
        detect_color(image, (10, 10, 10, 10))


def test_describe_item_combines_category_and_color():
    assert describe_item("short_sleeved_shirt", "빨간") == "빨간색 반팔 셔츠"


def test_describe_item_does_not_double_saek_suffix():
    assert describe_item("skirt", "갈색") == "갈색 치마"


def test_describe_item_falls_back_to_raw_category_when_unknown():
    assert describe_item("unknown_category", "파란") == "파란색 unknown_category"


def test_search_query_terms_returns_synonyms_for_ambiguous_shirt_category():
    assert search_query_terms("short_sleeved_shirt") == ["반팔 셔츠", "반팔 티셔츠"]


def test_search_query_terms_returns_single_translation_for_other_categories():
    assert search_query_terms("skirt") == ["치마"]


def test_search_query_terms_falls_back_to_raw_category_when_unknown():
    assert search_query_terms("unknown_category") == ["unknown_category"]
