"""탐지된 상품 영역의 색상 인식 및 한국어 색상명/설명 매핑."""

import colorsys

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

_CATEGORY_KO = {
    "short_sleeved_shirt": "반팔 셔츠",
    "long_sleeved_shirt": "긴팔 셔츠",
    "short_sleeved_outwear": "반팔 아우터",
    "long_sleeved_outwear": "긴팔 아우터",
    "vest": "조끼",
    "sling": "민소매 상의",
    "shorts": "반바지",
    "trousers": "긴바지",
    "skirt": "치마",
    "short_sleeved_dress": "반팔 원피스",
    "long_sleeved_dress": "긴팔 원피스",
    "vest_dress": "조끼 원피스",
    "sling_dress": "민소매 원피스",
}

# DeepFashion2의 상의 카테고리는 와이셔츠부터 그래픽 티셔츠까지 폭넓게 아우르는데,
# `_CATEGORY_KO`의 단일 번역만 네이버 검색어로 쓰면 랭킹이 칼라 있는 정장 셔츠
# 쪽으로 쏠려 실제로는 다른 스타일(티셔츠 등)인 상품을 후보군에서 놓친다.
# 애매한 카테고리는 동의어를 함께 검색해 후보군을 넓히기 위한 목록.
_CATEGORY_SEARCH_SYNONYMS = {
    "short_sleeved_shirt": ["반팔 셔츠", "반팔 티셔츠"],
    "long_sleeved_shirt": ["긴팔 셔츠", "긴팔 티셔츠"],
    "short_sleeved_outwear": ["반팔 아우터", "반팔 가디건"],
    "long_sleeved_outwear": ["긴팔 아우터", "긴팔 자켓"],
}


def detect_color(image: Image.Image, bbox: tuple) -> str:
    """bbox 영역의 대표 색상을 K-means로 추출해 한국어 색상명으로 반환한다 (예: "빨간")."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    crop = image.convert("RGB").crop((x1, y1, x2, y2))
    if crop.width == 0 or crop.height == 0:
        raise ValueError(f"bbox 영역이 비어 있습니다: {bbox}")

    dominant_rgb = _dominant_color(crop)
    return _rgb_to_korean_name(dominant_rgb)


def describe_item(category: str, color: str, subtype: str | None = None) -> str:
    """카테고리(또는 CLIP 제로샷으로 추정한 subtype) + 색상을 한국어 설명으로 합친다.

    subtype이 주어지면(예: "패딩") `_CATEGORY_KO`의 뭉뚱그린 번역("긴팔 아우터")
    대신 더 구체적인 subtype을 사용한다. subtype이 없으면 기존 동작과 동일.
    """
    category_ko = subtype or _CATEGORY_KO.get(category, category)
    if not color:
        return category_ko
    color_ko = color if color.endswith("색") else f"{color}색"
    return f"{color_ko} {category_ko}"


def search_query_terms(category: str, subtype: str | None = None) -> list[str]:
    """네이버 쇼핑 검색에 사용할 카테고리 동의어 목록을 반환한다.

    애매한 카테고리는 여러 동의어(예: "셔츠"/"티셔츠")를, 그 외에는 `_CATEGORY_KO`
    번역 하나만 담은 리스트를 반환한다. subtype이 주어지면(예: "패딩") 동의어
    목록에 추가로 포함해 검색 후보군을 넓힌다.
    """
    terms = list(_CATEGORY_SEARCH_SYNONYMS.get(category, [_CATEGORY_KO.get(category, category)]))
    if subtype and subtype not in terms:
        terms.append(subtype)
    return terms


def _dominant_color(crop: Image.Image) -> tuple:
    small = crop.resize((40, 40))
    pixels = np.array(small).reshape(-1, 3).astype(float)
    n_clusters = min(3, len(pixels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_idx = counts.argmax()
    r, g, b = kmeans.cluster_centers_[dominant_idx]
    return int(r), int(g), int(b)


def _rgb_to_korean_name(rgb: tuple) -> str:
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = h * 360

    if v < 0.2:
        return "검정"
    if s < 0.08:
        # 실제 사진(조명/그림자/원단 질감)은 합성 단색보다 채도가 낮게
        # 측정되는 경향이 있어, 카키색처럼 원래 채도가 낮은 색까지
        # 무채색으로 오분류하지 않도록 임계값을 보수적으로 낮게 둔다.
        return "흰색" if v > 0.85 else "회색"

    if hue_deg < 15 or hue_deg >= 345:
        return "빨간"
    if hue_deg < 45:
        return "갈색" if v < 0.55 else "주황"
    if hue_deg < 65:
        return "노란"
    if hue_deg < 170:
        return "초록"
    if hue_deg < 255:
        # 네이비는 hue만 보면 일반 파란과 같은 대역이라 어둡기(v)로 구분한다.
        return "남색" if v < 0.35 else "파란"
    if hue_deg < 280:
        return "남색"
    if hue_deg < 330:
        return "보라"
    return "분홍"
