"""AI 쇼핑 어시스턴트 Gradio 앱: 업로드 -> 탐지 -> 색상 -> 검색 -> 챗봇/예산 필터 -> 추천."""

import functools
import html
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
from prometheus_client import Counter, Histogram
from PIL import ImageDraw

from chat.llm import generate_response
from detection.color import describe_item, detect_color
from detection.detect import detect_products
from search.budget import filter_by_budget, parse_budget
from search.clip_search import classify_subtype, embed_image, index_products, search_similar
from search.naver_api import NaverAPIError, search_naver_variants
from search.text_search import index_product_texts, is_local_search_sufficient, search_similar_text

logger = logging.getLogger(__name__)

_TOP_K = 5
# search_similar는 카테고리/색상 스코핑이 없는 전역 product_images 컬렉션을
# 검색한다. 스코핑을 유지하면서도 시각 유사도로 재정렬하기 위해, top_k보다
# 넉넉한 풀을 가져와 scoped_products의 purchase_url과 교집합한다.
_RERANK_POOL_SIZE = 50

_FRESHNESS_KEYWORDS = ("최신", "신상", "재고", "실시간", "현재", "오늘")

IMAGE_UPLOAD_DURATION = Histogram(
    "shopping_assistant_image_upload_seconds", "이미지 업로드부터 인식/추천 완료까지 소요 시간"
)
CHAT_RESPONSE_DURATION = Histogram(
    "shopping_assistant_chat_response_seconds", "챗봇 응답 생성 소요 시간"
)
NAVER_API_CALLS = Counter(
    "shopping_assistant_naver_api_calls_total", "네이버 쇼핑 API 호출 결과", ["result"]
)


def _timed(histogram):
    """함수의 모든 반환 경로(조기 반환 포함)에 대해 소요 시간을 히스토그램에 기록한다."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.observe(time.perf_counter() - start)

        return wrapper

    return decorator


def _search_naver_with_metrics(category, color, subtype):
    """search_naver_variants를 감싸 성공/실패 카운터를 남긴다. 실패 시 NaverAPIError를
    그대로 재발생시켜 호출부의 기존 except 처리를 그대로 유지한다."""
    try:
        products = search_naver_variants(category, color, subtype)
        NAVER_API_CALLS.labels(result="success").inc()
        return products
    except NaverAPIError:
        NAVER_API_CALLS.labels(result="failure").inc()
        raise

_THEME = gr.themes.Base(
    font=[gr.themes.LocalFont("Pretendard"), "-apple-system", "BlinkMacSystemFont", "sans-serif"],
).set(
    body_background_fill="#FAFAF8",
    body_background_fill_dark="#18181B",
    body_text_color="#18181B",
    body_text_color_dark="#F4F4F5",
    body_text_color_subdued="#71717A",
    body_text_color_subdued_dark="#A1A1AA",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#212124",
    border_color_primary="#E5E3DE",
    border_color_primary_dark="#3F3F46",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#212124",
    block_border_color="#E5E3DE",
    block_border_color_dark="#3F3F46",
    block_label_text_color="#71717A",
    block_label_text_color_dark="#A1A1AA",
    block_radius="12px",
    block_shadow="none",
    input_border_color="#E5E3DE",
    input_border_color_dark="#3F3F46",
    button_primary_background_fill="#18181B",
    button_primary_background_fill_dark="#F4F4F5",
    button_primary_background_fill_hover="#3F3F46",
    button_primary_background_fill_hover_dark="#D4D4D8",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#18181B",
)

_CSS = """
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

.gradio-container, .gradio-container * {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.gradio-container { max-width: 1120px !important; }

.app-header {
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 16px;
    margin-bottom: 4px;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}
.app-header p {
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 0.92rem;
}

.recognition-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    background: var(--body-text-color);
    color: var(--background-fill-primary);
    font-size: 0.85rem;
    font-weight: 600;
}

.warning-banner {
    border-left: 3px solid #C0392B;
    background: #FBEAE8;
    padding: 10px 14px;
    border-radius: 4px;
}
.warning-item {
    font-size: 0.85rem;
    color: #7A2E27;
}
.warning-item strong { color: #C0392B; }

.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
}
.product-card {
    display: block;
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    overflow: hidden;
    background: var(--background-fill-primary);
    text-decoration: none;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.product-card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}
.product-card-image {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    display: block;
    background: #F1F0EC;
}
.product-card-body { padding: 10px 12px 14px; }
.product-card-name {
    font-size: 0.85rem;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.4em;
}
.product-card-price {
    font-size: 0.95rem;
    font-weight: 700;
    color: #C0392B;
    margin-top: 4px;
}
.product-card-source {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    margin-top: 2px;
}
.product-empty {
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
    padding: 24px 0;
    text-align: center;
}
"""


def _empty_state() -> dict:
    """호출마다 새 리스트를 담은 빈 state를 만든다. 모듈 전역 dict를 그대로
    재사용하면 `history` 리스트가 세션 간에 공유되어 한 세션의 대화 기록이
    다른(새) 세션의 '빈 상태'에 섞여 들어간다."""
    return {
        "detected_item": None,
        "color": None,
        "subtype": None,
        "candidate_products": [],
        "history": [],
    }


_CHAT_PLACEHOLDER = "예: 이거 뭐야? / 얼마야? / 더 저렴한 거 있어?"
_ANALYZING_PLACEHOLDER = "이미지 분석 중입니다. 잠시만 기다려주세요..."
_THINKING_PLACEHOLDER = "답변 생성 중입니다. 잠시만 기다려주세요..."


def _draw_bbox(image, bbox):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=(255, 0, 0), width=4)
    return overlay


def _build_recognition_badge(description):
    if not description:
        return ""
    return f'<div class="recognition-badge">인식된 상품: {html.escape(description)}</div>'


def _build_warning_banner(warnings):
    if not warnings:
        return ""
    items = "".join(
        f'<div class="warning-item"><strong>주의:</strong> {html.escape(w)}</div>' for w in warnings
    )
    return f'<div class="warning-banner">{items}</div>'


def _rerank_by_visual_similarity(embedding, scoped_products):
    """전역 CLIP 검색 결과를 scoped_products(카테고리/색상 필터가 이미 적용된
    후보)의 purchase_url과 교집합해 시각 유사도 순서만 가져온다. 교집합이
    비면(스코핑된 후보와 전역 컬렉션 히트가 겹치지 않으면) 스코핑된 후보를
    그대로 사용한다 — search_similar가 스코핑 없는 전역 컬렉션을 검색해
    무관한 결과로 스코핑된 후보를 덮어써 버리던 문제를 막는다."""
    if not scoped_products:
        return []
    scoped_urls = {p["purchase_url"] for p in scoped_products}
    similar_pool = search_similar(embedding, top_k=_RERANK_POOL_SIZE)
    reranked = [p for p in similar_pool if p["purchase_url"] in scoped_urls][:_TOP_K]
    return reranked if reranked else scoped_products


def _is_freshness_request(message):
    """'최신/신상/재고/실시간/현재/오늘' 등 신선도 키워드가 있으면 네이버 실시간
    재검색을 트리거한다 (원본 스펙의 "특정 키워드로 웹 검색 트리거" 요구사항)."""
    return any(keyword in message for keyword in _FRESHNESS_KEYWORDS)


def _index_products_everywhere(products, category, color):
    """새로 얻은 상품을 이미지(CLIP)/텍스트(KO-SRoBERTa) 컬렉션 양쪽에 병렬로
    색인한다 (Phase 13 이전엔 순차 실행돼 cold 경로 지연에 일조했음)."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_label = {
            executor.submit(index_products, products, category=category, color=color): "이미지",
            executor.submit(
                index_product_texts, products, category=category, color=color
            ): "텍스트",
        }
        for future in as_completed(future_to_label):
            try:
                future.result()
            except Exception:
                logger.exception("상품 %s 색인 실패", future_to_label[future])


def _lock_chat_input_for_analysis():
    """이미지 분석이 끝나기 전에 채팅을 보내 빈 state로 응답받는 것을 막는다."""
    return gr.update(interactive=False, placeholder=_ANALYZING_PLACEHOLDER)


def _lock_chat_input_for_response():
    """챗봇 응답 생성 중 중복 제출을 막는다."""
    return gr.update(interactive=False, placeholder=_THINKING_PLACEHOLDER)


def _unlock_chat_input():
    return gr.update(interactive=True, placeholder=_CHAT_PLACEHOLDER)


def _build_product_cards(products):
    """네이버 API가 반환한 상품명/이미지/구매링크는 신뢰할 수 없는 외부
    입력이라 그대로 gr.HTML()에 보간하면 Stored XSS가 된다. name/source는
    이스케이프, purchase_url은 http(s) 스킴만 허용해 javascript:/data:
    스킴을 거른다. image_url도 이스케이프해 <img src="...">에 안전하게
    보간한다(별도 스킴 검증 없이 <img src>로만 쓰여 실행 가능한 벡터가
    아님)."""
    if not products:
        return '<div class="product-empty">추천할 상품이 없습니다.</div>'
    cards = []
    for p in products:
        url = p["purchase_url"]
        if not url.startswith(("http://", "https://")):
            continue
        name = html.escape(p["name"])
        source = html.escape(p.get("source", ""))
        image_url = html.escape(p["image_url"])
        href = html.escape(url)
        cards.append(
            f'<a class="product-card" href="{href}" target="_blank" rel="noopener noreferrer">'
            f'<img class="product-card-image" src="{image_url}" alt="{name}" loading="lazy">'
            f'<div class="product-card-body">'
            f'<div class="product-card-name">{name}</div>'
            f'<div class="product-card-price">{p["price"]:,}원</div>'
            f'<div class="product-card-source">{source} · 구매하러 가기</div>'
            f"</div></a>"
        )
    if not cards:
        return '<div class="product-empty">추천할 상품이 없습니다.</div>'
    return f'<div class="product-grid">{"".join(cards)}</div>'


@_timed(IMAGE_UPLOAD_DURATION)
def on_image_upload(image):
    if image is None:
        return None, "", "", "", _empty_state(), []

    detections = detect_products(image)
    if not detections:
        message = "인식된 상품이 없습니다. 다른 사진을 업로드해보세요."
        chatbot = [{"role": "assistant", "content": message}]
        return (
            image,
            "",
            _build_warning_banner([message]),
            _build_product_cards([]),
            _empty_state(),
            chatbot,
        )

    detection = max(detections, key=lambda d: d["confidence"])
    bbox = detection["bbox"]
    color = detect_color(image, bbox)
    cropped = image.convert("RGB").crop(tuple(int(round(v)) for v in bbox))
    subtype = classify_subtype(cropped, detection["category"])
    description = describe_item(detection["category"], color, subtype)

    info_html = _build_recognition_badge(description)
    warnings = []

    # 로컬 텍스트 벡터(KO-SRoBERTa) 검색 우선 — 충분하면 네이버 실시간 호출을
    # 건너뛴다 (원본 스펙: "벡터 DB에서의 상품 검색"이 기본, 네이버는 신선도
    # 키워드가 있을 때만).
    local_results = search_similar_text(description, top_k=_TOP_K * 2)
    if is_local_search_sufficient(local_results):
        logger.info("로컬 텍스트 검색으로 충분 — 네이버 호출 스킵 (query=%s)", description)
        naver_products = local_results
    else:
        try:
            naver_products = _search_naver_with_metrics(detection["category"], color, subtype)
        except NaverAPIError:
            naver_products = []
            warnings.append("상품 정보를 가져오지 못했습니다.")

        if naver_products:
            _index_products_everywhere(naver_products, detection["category"], color)

    embedding = embed_image(cropped)
    candidate_products = _rerank_by_visual_similarity(embedding, naver_products)

    if not candidate_products:
        warnings.append("유사 상품을 찾지 못했습니다.")

    bbox_image = _draw_bbox(image, bbox)
    state = {
        "detected_item": detection,
        "color": color,
        "subtype": subtype,
        "candidate_products": candidate_products,
        "history": [],
    }
    chatbot = [
        {
            "role": "assistant",
            "content": f"'{description}'을(를) 인식했어요! 무엇이 궁금하신가요?",
        }
    ]

    return (
        bbox_image,
        info_html,
        _build_warning_banner(warnings),
        _build_product_cards(candidate_products),
        state,
        chatbot,
    )


def on_chat_submit(user_message, chatbot_messages, state):
    chatbot_messages = list(chatbot_messages or [])
    state = dict(state) if state else _empty_state()
    candidate_products = state.get("candidate_products", [])

    detected_item = state.get("detected_item")
    if detected_item and _is_freshness_request(user_message):
        color = state.get("color") or ""
        subtype = state.get("subtype")
        try:
            fresh_products = _search_naver_with_metrics(detected_item["category"], color, subtype)
        except NaverAPIError:
            logger.exception("챗봇 신선도 재검색 실패")
            fresh_products = []
        if fresh_products:
            _index_products_everywhere(fresh_products, detected_item["category"], color)
            candidate_products = fresh_products
            state["candidate_products"] = candidate_products

    condition = parse_budget(user_message)
    if condition is not None:
        candidate_products = filter_by_budget(candidate_products, condition)
        state["candidate_products"] = candidate_products

    context = {
        "detected_item": detected_item,
        "color": state.get("color"),
        "subtype": state.get("subtype"),
        "candidate_products": candidate_products,
        "history": state.get("history", []),
    }

    try:
        start = time.perf_counter()
        response = generate_response(user_message, context)
        CHAT_RESPONSE_DURATION.observe(time.perf_counter() - start)
    except Exception:
        logger.exception("챗봇 응답 생성 실패")
        response = "죄송해요, 답변을 생성하지 못했습니다."

    state.setdefault("history", []).append((user_message, response))
    chatbot_messages.append({"role": "user", "content": user_message})
    chatbot_messages.append({"role": "assistant", "content": response})

    return (
        chatbot_messages,
        "",
        state,
        _build_product_cards(candidate_products),
    )


with gr.Blocks(title="AI 쇼핑 어시스턴트") as demo:
    gr.HTML(
        '<div class="app-header"><h1>AI 쇼핑 어시스턴트</h1>'
        "<p>상품 사진을 올리면 비슷한 상품을 찾아드려요.</p></div>"
    )

    state = gr.State(_empty_state())

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="상품 사진 업로드")
            bbox_output = gr.Image(label="탐지 결과")
            detection_html = gr.HTML()
            warning_html = gr.HTML()
        with gr.Column():
            chatbot = gr.Chatbot(label="쇼핑 어시스턴트", elem_classes=["chat-panel"])
            chat_input = gr.Textbox(
                label="메시지 입력",
                placeholder=_CHAT_PLACEHOLDER,
            )

    gr.Markdown("## 추천 상품")
    products_html = gr.HTML(elem_classes=["product-section"])

    image_input.upload(
        fn=_lock_chat_input_for_analysis,
        outputs=[chat_input],
    ).then(
        fn=on_image_upload,
        inputs=[image_input],
        outputs=[bbox_output, detection_html, warning_html, products_html, state, chatbot],
    ).then(
        fn=_unlock_chat_input,
        outputs=[chat_input],
    )
    chat_input.submit(
        fn=_lock_chat_input_for_response,
        outputs=[chat_input],
    ).then(
        fn=on_chat_submit,
        inputs=[chat_input, chatbot, state],
        outputs=[chatbot, chat_input, state, products_html],
    ).then(
        fn=_unlock_chat_input,
        outputs=[chat_input],
    )


if __name__ == "__main__":
    demo.launch(theme=_THEME, css=_CSS)
