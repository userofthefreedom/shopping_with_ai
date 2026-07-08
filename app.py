"""AI 쇼핑 어시스턴트 Gradio 앱: 업로드 -> 탐지 -> 색상 -> 검색 -> 챗봇/예산 필터 -> 추천."""

import logging

import gradio as gr
from PIL import ImageDraw

from chat.llm import generate_response
from detection.color import describe_item, detect_color
from detection.detect import detect_products
from search.budget import filter_by_budget, parse_budget
from search.clip_search import embed_image, index_products, search_similar
from search.naver_api import NaverAPIError, search_naver

logger = logging.getLogger(__name__)

_TOP_K = 5

_EMPTY_STATE = {"detected_item": None, "color": None, "candidate_products": [], "history": []}


def _draw_bbox(image, bbox):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=(255, 0, 0), width=4)
    return overlay


def _build_gallery(products):
    return [(p["image_url"], f"{p['name']} - {p['price']:,}원") for p in products]


def _build_links_html(products):
    if not products:
        return ""
    items = "\n".join(
        f'<li><a href="{p["purchase_url"]}" target="_blank">{p["name"]} - {p["price"]:,}원 '
        f'({p.get("source", "")}) 구매하러 가기</a></li>'
        for p in products
    )
    return f"<ul>{items}</ul>"


def on_image_upload(image):
    if image is None:
        return None, "", [], "", dict(_EMPTY_STATE), []

    detections = detect_products(image)
    if not detections:
        message = "인식된 상품이 없습니다. 다른 사진을 업로드해보세요."
        chatbot = [{"role": "assistant", "content": message}]
        return image, message, [], "", dict(_EMPTY_STATE), chatbot

    detection = max(detections, key=lambda d: d["confidence"])
    bbox = detection["bbox"]
    color = detect_color(image, bbox)
    description = describe_item(detection["category"], color)

    status_lines = [f"인식된 상품: {description}"]

    try:
        naver_products = search_naver(description)
    except NaverAPIError:
        naver_products = []
        status_lines.append("상품 정보를 가져오지 못했습니다.")

    if naver_products:
        try:
            index_products(naver_products, category=detection["category"], color=color)
        except Exception:
            logger.exception("상품 색인 실패")

    cropped = image.convert("RGB").crop(tuple(int(round(v)) for v in bbox))
    embedding = embed_image(cropped)
    candidate_products = search_similar(embedding, top_k=_TOP_K) or naver_products

    if not candidate_products:
        status_lines.append("유사 상품을 찾지 못했습니다.")

    bbox_image = _draw_bbox(image, bbox)
    state = {
        "detected_item": detection,
        "color": color,
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
        "\n".join(status_lines),
        _build_gallery(candidate_products),
        _build_links_html(candidate_products),
        state,
        chatbot,
    )


def on_chat_submit(user_message, chatbot_messages, state):
    chatbot_messages = list(chatbot_messages or [])
    state = dict(state or _EMPTY_STATE)
    candidate_products = state.get("candidate_products", [])

    condition = parse_budget(user_message)
    if condition is not None:
        candidate_products = filter_by_budget(candidate_products, condition)
        state["candidate_products"] = candidate_products

    context = {
        "detected_item": state.get("detected_item"),
        "color": state.get("color"),
        "candidate_products": candidate_products,
        "history": state.get("history", []),
    }

    try:
        response = generate_response(user_message, context)
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
        _build_gallery(candidate_products),
        _build_links_html(candidate_products),
    )


with gr.Blocks(title="AI 쇼핑 어시스턴트") as demo:
    gr.Markdown("# AI 쇼핑 어시스턴트")

    state = gr.State(dict(_EMPTY_STATE))

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="상품 사진 업로드")
            bbox_output = gr.Image(label="탐지 결과")
            detection_text = gr.Textbox(label="인식 결과", interactive=False)
        with gr.Column():
            chatbot = gr.Chatbot(label="쇼핑 어시스턴트")
            chat_input = gr.Textbox(
                label="메시지 입력",
                placeholder="예: 이거 뭐야? / 얼마야? / 더 저렴한 거 있어?",
            )

    gr.Markdown("## 추천 상품")
    gallery = gr.Gallery(label="추천 상품", columns=5)
    links_html = gr.HTML()

    image_input.upload(
        fn=on_image_upload,
        inputs=[image_input],
        outputs=[bbox_output, detection_text, gallery, links_html, state, chatbot],
    )
    chat_input.submit(
        fn=on_chat_submit,
        inputs=[chat_input, chatbot, state],
        outputs=[chatbot, chat_input, state, gallery, links_html],
    )


if __name__ == "__main__":
    demo.launch()
