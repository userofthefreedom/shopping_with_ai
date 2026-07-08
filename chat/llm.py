"""Llama 3.2 Korean Bllossom 3B 기반 쇼핑 챗봇 응답 생성."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
_MAX_NEW_TOKENS = 300

_SYSTEM_PROMPT_TEMPLATE = """너는 친절한 한국어 AI 쇼핑 어시스턴트야. 사용자가 업로드한 상품 사진을 분석한 \
결과와 관련 상품 후보를 바탕으로 자연스러운 한국어로 답해야 해.

{item_description}

{candidates_description}

모르는 정보는 지어내지 말고, 위에 주어진 정보 범위 안에서만 답해."""

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return _model, _tokenizer


def generate_response(user_message: str, context: dict) -> str:
    """ChatContext(탐지 결과/색상/후보 상품/대화 이력)를 바탕으로 자연어 응답을 생성한다."""
    model, tokenizer = _get_model()

    messages = [{"role": "system", "content": _build_system_prompt(context)}]
    for user_turn, assistant_turn in context.get("history", []):
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": user_message})

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    input_len = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


def _build_system_prompt(context: dict) -> str:
    detected_item = context.get("detected_item")
    color = context.get("color")
    candidate_products = context.get("candidate_products") or []

    if detected_item:
        category = detected_item.get("category", "상품")
        item_description = f"인식된 상품: {color or ''}색 {category}".strip()
    else:
        item_description = "인식된 상품: 없음"

    if candidate_products:
        lines = [
            f"{i}. {p['name']} - {p['price']:,}원 ({p.get('source', '알 수 없음')})"
            for i, p in enumerate(candidate_products, start=1)
        ]
        candidates_description = "추천 후보 상품 목록:\n" + "\n".join(lines)
    else:
        candidates_description = "추천 후보 상품 목록: 없음 (아직 추천할 상품이 없음)"

    return _SYSTEM_PROMPT_TEMPLATE.format(
        item_description=item_description, candidates_description=candidates_description
    )
