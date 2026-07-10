"""KO-SRoBERTa 텍스트 임베딩 기반 로컬 상품 검색.

네이버 쇼핑 API 키워드 검색은 카테고리/색상 표현이 네이버 랭킹과 정확히
맞아떨어지지 않으면 후보군 자체가 빈약해지고(Phase 10), 매 요청마다 실시간
네트워크 호출이 필요해 응답 시간의 병목이 된다(Phase 8/9). 상품 텍스트를
KO-SRoBERTa로 임베딩해 로컬 ChromaDB에 저장해두면, 같은/유사한 쿼리는 네트워크
없이 의미 기반으로 검색할 수 있다 (원본 스펙 문서의 "벡터 DB에서의 상품 검색"
요구사항).
"""

import logging

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from search.clip_search import _make_id

logger = logging.getLogger(__name__)

_MODEL_NAME = "jhgan/ko-sroberta-multitask"
_CHROMA_PATH = "data/chroma"
_COLLECTION_NAME = "product_text"

# 로컬 텍스트 벡터 검색 결과를 "네이버 실시간 검색 없이 써도 될 만큼
# 충분하다"고 판단하는 기준. 실제 네이버 검색 결과로 실측한 결과(Phase 12,
# TEST_RESULT.md 참고) 관련 있는 쿼리는 유사도 0.618~0.722, 무관한 쿼리는
# 0.430~0.483으로 뚜렷이 갈렸다 — CLIP 이미지 유사도(Phase 10, 0.54~0.60에
# 몰려 못 갈렸음)와 달리 KO-SRoBERTa 텍스트 유사도는 분리가 잘 되어 0.6을
# 그 사이 값으로 채택.
_LOCAL_SEARCH_MIN_COUNT = 5
_LOCAL_SEARCH_MIN_SIMILARITY = 0.6

_model = None
_client = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(_MODEL_NAME, device=device)
    return _model


def _get_collection():
    global _client, _collection
    if _collection is None:
        # 로컬 변수에 먼저 담고 한 번에 대입해 TOCTOU 레이스를 피한다
        # (clip_search.py의 _get_model()/_get_collection()과 동일한 패턴).
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        collection = client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        _client, _collection = client, collection
    return _collection


def embed_text(text: str) -> np.ndarray:
    """KO-SRoBERTa로 텍스트의 L2 정규화된 768차원 임베딩 벡터를 계산한다."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return np.asarray(vector)


def index_product_texts(products: list[dict], category: str, color: str) -> int:
    """상품명을 KO-SRoBERTa로 임베딩해 ChromaDB `product_text` 컬렉션에 저장한다.

    `search/clip_search.py`의 `index_products`(Phase 9)와 동일하게, 이미
    색인된 `purchase_url`(id가 동일)인 상품은 재임베딩하지 않고 건너뛴다.
    이미지 다운로드가 없어 병렬화는 필요하지 않다.

    Returns:
        색인된(이미 있었던 것 포함) 상품 수. 임베딩 실패한 상품은 제외.
    """
    collection = _get_collection()
    valid_products = [p for p in products if p.get("purchase_url")]
    skipped = len(products) - len(valid_products)
    if skipped:
        # purchase_url이 빈 상품은 전부 같은 해시로 id가 겹쳐 서로 덮어쓴다
        # (clip_search.index_products와 동일한 이유로 색인 전에 제외).
        logger.warning("purchase_url이 없는 상품 %d건을 텍스트 색인에서 제외함", skipped)
    id_to_product = {_make_id(product): product for product in valid_products}

    existing_ids = set()
    if id_to_product:
        existing = collection.get(ids=list(id_to_product.keys()))
        existing_ids = set(existing["ids"])

    to_embed = {
        product_id: product
        for product_id, product in id_to_product.items()
        if product_id not in existing_ids
    }

    ids, embeddings, metadatas = [], [], []
    for product_id, product in to_embed.items():
        try:
            embedding = embed_text(product["name"])
        except Exception:
            logger.exception("상품 텍스트 임베딩 실패: %s", product.get("name"))
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


def search_similar_text(query: str, top_k: int = 5) -> list[dict]:
    """쿼리와 의미적으로 유사한 상위 top_k개의 Product를 반환한다.

    반환되는 각 dict는 SPEC §2의 Product 계약에 `similarity`(코사인 유사도,
    1에 가까울수록 유사)를 추가로 포함한다 — 호출 측(3_1542353.py)이 결과가
    "충분히 신뢰할만한지" 판단해 네이버 실시간 검색으로 폴백할지 정하는 데
    사용한다.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_embedding = embed_text(query)
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, count),
    )
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "name": m["name"],
            "price": m["price"],
            "image_url": m["image_url"],
            "purchase_url": m["purchase_url"],
            "source": "naver",
            "similarity": 1 - dist,
        }
        for m, dist in zip(metadatas, distances)
    ]


def is_local_search_sufficient(results: list[dict]) -> bool:
    """로컬 텍스트 벡터 검색 결과만으로 네이버 실시간 검색을 건너뛰어도 될지 판단한다."""
    if len(results) < _LOCAL_SEARCH_MIN_COUNT:
        return False
    return results[0]["similarity"] >= _LOCAL_SEARCH_MIN_SIMILARITY
