"""Llama 3.2 Korean Bllossom 3B 기반 쇼핑 챗봇 응답 생성 (LangChain 기반)."""

import logging

import torch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline

from detection.color import describe_item

logger = logging.getLogger(__name__)

_MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
_MAX_NEW_TOKENS = 300

_SYSTEM_PROMPT_TEMPLATE = """너는 친절한 한국어 AI 쇼핑 어시스턴트야. 사용자가 업로드한 상품 사진을 분석한 \
결과와 관련 상품 후보를 바탕으로 자연스러운 한국어로 답해야 해.

{item_description}

{candidates_description}

모르는 정보는 지어내지 말고, 위에 주어진 정보 범위 안에서만 답해."""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder("history"),
        ("human", "{user_message}"),
    ]
)

_llm = None


def _get_model():
    """ChatHuggingFace로 감싼 LLM을 lazy singleton으로 로드한다.

    ChatHuggingFace는 내부적으로 tokenizer의 `apply_chat_template`을 그대로
    사용해 메시지 목록을 프롬프트 문자열로 변환하므로, LangChain 도입 전
    (`tokenizer.apply_chat_template` 직접 호출) 대비 실제 모델 입력 포맷이
    동등하게 유지된다.
    """
    global _llm
    if _llm is None:
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        text_generation_pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=terminators,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        _llm = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
    return _llm


def generate_response(user_message: str, context: dict) -> str:
    """ChatContext(탐지 결과/색상/후보 상품/대화 이력)를 바탕으로 자연어 응답을 생성한다."""
    llm = _get_model()

    history_messages = []
    for user_turn, assistant_turn in context.get("history", []):
        history_messages.append(HumanMessage(content=user_turn))
        history_messages.append(AIMessage(content=assistant_turn))

    chain = _PROMPT | llm
    result = chain.invoke(
        {
            "system_prompt": _build_system_prompt(context),
            "history": history_messages,
            "user_message": user_message,
        }
    )
    return result.content.strip()


def _build_system_prompt(context: dict) -> str:
    detected_item = context.get("detected_item")
    color = context.get("color")
    subtype = context.get("subtype")
    candidate_products = context.get("candidate_products") or []

    if detected_item:
        description = describe_item(detected_item.get("category", "상품"), color or "", subtype)
        item_description = f"인식된 상품: {description}"
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
