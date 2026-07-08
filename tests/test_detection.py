from PIL import Image
from ultralytics.utils import ASSETS

from detection.detect import detect_products

_DEEPFASHION2_CATEGORIES = {
    "short_sleeved_shirt",
    "long_sleeved_shirt",
    "short_sleeved_outwear",
    "long_sleeved_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeved_dress",
    "long_sleeved_dress",
    "vest_dress",
    "sling_dress",
}


def test_detect_products_returns_detections_for_real_image():
    image = Image.open(ASSETS / "zidane.jpg")

    detections = detect_products(image)

    assert len(detections) >= 1
    for detection in detections:
        assert set(detection.keys()) == {"bbox", "category", "confidence"}
        x1, y1, x2, y2 = detection["bbox"]
        assert x1 < x2
        assert y1 < y2
        assert detection["category"] in _DEEPFASHION2_CATEGORIES
        assert 0.0 <= detection["confidence"] <= 1.0


def test_detect_products_returns_empty_list_when_no_product_found():
    blank_image = Image.new("RGB", (640, 640), color=(128, 128, 128))

    detections = detect_products(blank_image)

    assert detections == []
