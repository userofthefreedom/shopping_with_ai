"""CLIP ViT-B/32 이미지 임베딩 및 ChromaDB 기반 유사 상품 검색."""

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

_MODEL_NAME = "openai/clip-vit-base-patch32"
_CHROMA_PATH = "data/chroma"
_COLLECTION_NAME = "product_images"

# DeepFashion2 13개 카테고리는 스펙이 고정한 요구사항이라 탐지 모델 자체는
# 바꾸지 않는다. 대신 이미 로드된 CLIP의 제로샷 이미지-텍스트 매칭으로 각
# 카테고리 내 세부 종류(예: "긴팔 아우터" 중 패딩/코트/가디건/자켓)를 추정해
# 설명/검색어를 더 구체적으로 만든다. CLIP 텍스트 인코더는 영어 중심으로
# 학습되어 한국어 제로샷 매칭은 신뢰하기 어려우므로, 영어 후보 라벨로 매칭한
# 뒤 한국어 표시명으로 매핑한다 (영어 라벨, 한국어 표시명) 튜플.
_SUBTYPE_CANDIDATES = {
    "short_sleeved_shirt": [
        ("t-shirt", "티셔츠"),
        ("polo shirt", "카라 티셔츠"),
        ("dress shirt", "셔츠"),
        ("blouse", "블라우스"),
    ],
    "long_sleeved_shirt": [
        ("dress shirt", "셔츠"),
        ("blouse", "블라우스"),
        ("knit sweater", "니트"),
        ("long sleeve t-shirt", "긴팔 티셔츠"),
    ],
    "short_sleeved_outwear": [
        ("vest jacket", "베스트"),
        ("short cardigan", "가디건"),
    ],
    "long_sleeved_outwear": [
        ("padding jacket", "패딩"),
        ("coat", "코트"),
        ("cardigan", "가디건"),
        ("blazer jacket", "자켓"),
    ],
    "vest": [
        ("vest", "조끼"),
        ("waistcoat", "베스트"),
    ],
    "sling": [
        ("tank top", "탱크톱"),
        ("camisole", "캐미솔"),
    ],
    "shorts": [
        ("denim shorts", "청반바지"),
        ("casual shorts", "반바지"),
    ],
    "trousers": [
        ("jeans", "청바지"),
        ("slacks", "슬랙스"),
        ("sweatpants", "조거팬츠"),
    ],
    "skirt": [
        ("mini skirt", "미니스커트"),
        ("pleated skirt", "플리츠 스커트"),
        ("long skirt", "롱스커트"),
    ],
    "short_sleeved_dress": [
        ("casual dress", "원피스"),
        ("shirt dress", "셔츠 원피스"),
    ],
    "long_sleeved_dress": [
        ("sweater dress", "니트 원피스"),
        ("coat dress", "코트 원피스"),
    ],
    "vest_dress": [
        ("sleeveless dress", "민소매 원피스"),
        ("pinafore dress", "멜빵 원피스"),
    ],
    "sling_dress": [
        ("slip dress", "슬립 원피스"),
        ("camisole dress", "캐미솔 원피스"),
    ],
}

_model = None
_processor = None
_client = None
_collection = None


def _get_model():
    global _model, _processor
    if _model is None:
        # 로컬 변수에 먼저 담고 마지막에 한 번에 전역 대입한다 — _model만
        # 먼저 대입하면, 동시에 들어온 다른 호출이 `if _model is None` 체크를
        # 통과해 아직 None인 _processor를 반환할 수 있다(레이스 컨디션).
        model = CLIPModel.from_pretrained(_MODEL_NAME)
        model.eval()
        processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model, _processor = model, processor
    return _model, _processor


def _get_collection():
    global _client, _collection
    if _collection is None:
        # _get_model()과 동일한 이유로 로컬 변수에 먼저 담고 한 번에 대입한다.
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        collection = client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        _client, _collection = client, collection
    return _collection


def embed_image(image: Image.Image) -> np.ndarray:
    """CLIP ViT-B/32로 이미지의 L2 정규화된 임베딩 벡터를 계산한다."""
    model, processor = _get_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    features = output.pooler_output
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze(0).numpy()


def classify_subtype(image_crop: Image.Image, category: str, min_confidence: float = 0.35) -> str | None:
    """CLIP 제로샷 이미지-텍스트 매칭으로 카테고리 내 세부 종류를 추정한다.

    `category`에 정의된 후보가 없거나, 최고 확률이 `min_confidence` 미만이면
    `None`을 반환한다 (호출 측은 기존 카테고리 번역으로 안전하게 폴백).
    """
    candidates = _SUBTYPE_CANDIDATES.get(category)
    if not candidates:
        return None

    model, processor = _get_model()
    english_labels = [label for label, _ in candidates]
    inputs = processor(
        text=english_labels, images=image_crop.convert("RGB"), return_tensors="pt", padding=True
    )
    with torch.no_grad():
        output = model(**inputs)

    probs = output.logits_per_image.softmax(dim=-1).squeeze(0)
    best_idx = int(probs.argmax().item())
    if probs[best_idx].item() < min_confidence:
        return None
    return candidates[best_idx][1]


def _download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def index_products(products: list[dict], category: str, color: str) -> int:
    """네이버 상품 이미지를 다운로드해 CLIP 임베딩으로 변환 후 ChromaDB에 저장한다.

    이미 색인된 `purchase_url`(id가 동일)인 상품은 재다운로드/재임베딩하지
    않고 건너뛴다. 남은 상품의 이미지 다운로드는 병렬로 수행한다
    (Phase 8에서 순차 다운로드가 응답 시간의 99% 이상을 차지하는 병목으로
    확인됨). CLIP 임베딩 자체는 스레드 세이프성을 보장할 수 없어 병렬화하지
    않는다 — 다운로드에 비해 이미 충분히 빠르다.

    Returns:
        색인된(이미 있었던 것 포함) 상품 수. 이미지 다운로드/임베딩 실패한
        상품은 제외.
    """
    valid_products = [p for p in products if p.get("purchase_url")]
    skipped = len(products) - len(valid_products)
    if skipped:
        # purchase_url이 빈 상품은 전부 같은 해시로 id가 겹쳐 서로 덮어쓴다
        # (다른 상품인데 하나만 색인되는 결과) — 색인 전에 걸러낸다.
        logger.warning("purchase_url이 없는 상품 %d건을 색인에서 제외함", skipped)

    collection = _get_collection()
    id_to_product = {_make_id(product): product for product in valid_products}

    existing_ids = set()
    if id_to_product:
        existing = collection.get(ids=list(id_to_product.keys()))
        existing_ids = set(existing["ids"])

    to_fetch = {
        product_id: product
        for product_id, product in id_to_product.items()
        if product_id not in existing_ids
    }

    downloaded = {}
    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(8, len(to_fetch))) as executor:
            future_to_id = {
                executor.submit(_download_image, product["image_url"]): product_id
                for product_id, product in to_fetch.items()
            }
            for future in as_completed(future_to_id):
                product_id = future_to_id[future]
                try:
                    downloaded[product_id] = future.result()
                except Exception:
                    logger.exception(
                        "상품 이미지 다운로드 실패: %s",
                        to_fetch[product_id].get("image_url"),
                    )

    ids, embeddings, metadatas = [], [], []
    for product_id, image in downloaded.items():
        product = to_fetch[product_id]
        try:
            embedding = embed_image(image)
        except Exception:
            logger.exception("상품 이미지 임베딩 실패: %s", product.get("image_url"))
            continue

        ids.append(product_id)
        embeddings.append(embedding.tolist())
        metadatas.append(
            {
                "name": product["name"],
                "price": product["price"],
                "image_url": product["image_url"],
                "purchase_url": product["purchase_url"],
                "category": category,
                "color": color,
            }
        )

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    return len(existing_ids) + len(ids)


def search_similar(image_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """이미지 임베딩과 코사인 유사도가 높은 상위 top_k개의 Product를 반환한다."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[np.asarray(image_embedding).tolist()],
        n_results=min(top_k, count),
    )
    metadatas = result["metadatas"][0]
    return [
        {
            "name": m["name"],
            "price": m["price"],
            "image_url": m["image_url"],
            "purchase_url": m["purchase_url"],
            "source": "naver",
        }
        for m in metadatas
    ]


def _make_id(product: dict) -> str:
    return hashlib.sha256(product["purchase_url"].encode()).hexdigest()[:16]
