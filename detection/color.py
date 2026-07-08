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


def detect_color(image: Image.Image, bbox: tuple) -> str:
    """bbox 영역의 대표 색상을 K-means로 추출해 한국어 색상명으로 반환한다 (예: "빨간")."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    crop = image.convert("RGB").crop((x1, y1, x2, y2))
    if crop.width == 0 or crop.height == 0:
        raise ValueError(f"bbox 영역이 비어 있습니다: {bbox}")

    dominant_rgb = _dominant_color(crop)
    return _rgb_to_korean_name(dominant_rgb)


def describe_item(category: str, color: str) -> str:
    """카테고리 + 색상을 "빨간색 반팔 셔츠" 같은 한국어 설명으로 합친다."""
    category_ko = _CATEGORY_KO.get(category, category)
    color_ko = color if color.endswith("색") else f"{color}색"
    return f"{color_ko} {category_ko}"


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
    if s < 0.15:
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
        return "파란"
    if hue_deg < 280:
        return "남색"
    if hue_deg < 330:
        return "보라"
    return "분홍"
