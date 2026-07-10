"""app.py의 Gradio Blocks를 FastAPI에 마운트하고 /health, /metrics를 추가한다.

배포(Phase 15) 전용 래핑 레이어. 로직/라우팅은 app.py 그대로 재사용하며 여기서는
헬스체크와 Prometheus 계측만 얹는다. 실행: `uvicorn server:app --host 0.0.0.0 --port 7860`.
"""

import gradio as gr
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app import _CSS, _THEME, demo

app = FastAPI(title="AI 쇼핑 어시스턴트")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health():
    return {"status": "ok"}


gr.mount_gradio_app(app, demo, path="/", theme=_THEME, css=_CSS)
