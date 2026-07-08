"""CLIP ViT-B/32 이미지 임베딩 및 ChromaDB 기반 유사 상품 검색."""

import hashlib
import io
import logging

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

_model = None
_processor = None
_client = None
_collection = None


def _get_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(_MODEL_NAME)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    return _model, _processor


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
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


def index_products(products: list[dict], category: str, color: str) -> int:
    """네이버 상품 이미지를 다운로드해 CLIP 임베딩으로 변환 후 ChromaDB에 저장한다.

    Returns:
        성공적으로 색인된 상품 수 (이미지 다운로드/임베딩 실패한 상품은 건너뜀).
    """
    collection = _get_collection()
    ids, embeddings, metadatas = [], [], []

    for product in products:
        try:
            response = requests.get(product["image_url"], timeout=5)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            embedding = embed_image(image)
        except Exception:
            logger.exception("상품 이미지 색인 실패: %s", product.get("image_url"))
            continue

        ids.append(_make_id(product))
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
    return len(ids)


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
