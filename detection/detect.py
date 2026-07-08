"""YOLOv8 기반 상품(패션 아이템) 탐지."""

import logging

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_MODEL_REPO = "Bingsu/adetailer"
_MODEL_FILE = "deepfashion2_yolov8s-seg.pt"
_CONF_THRESHOLD = 0.25

_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        weights_path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
        _model = YOLO(weights_path)
    return _model


def detect_products(image: Image.Image) -> list[dict]:
    """이미지에서 패션 상품을 탐지한다.

    Returns:
        [{"bbox": (x1, y1, x2, y2), "category": str, "confidence": float}, ...]
        상품이 하나도 탐지되지 않으면 빈 리스트.
    """
    try:
        model = _get_model()
        results = model.predict(source=image, conf=_CONF_THRESHOLD, verbose=False)
    except Exception:
        logger.exception("상품 탐지 중 오류 발생")
        raise

    detections = []
    boxes = results[0].boxes
    if boxes is None:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "category": model.names[cls_id],
                "confidence": confidence,
            }
        )

    return detections
