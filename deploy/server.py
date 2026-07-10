"""3_1542353.py의 Gradio Blocks를 FastAPI에 마운트하고 /health, /metrics를 추가한다.

배포(Phase 15) 전용 래핑 레이어. 로직/라우팅은 3_1542353.py 그대로 재사용하며 여기서는
헬스체크와 Prometheus 계측만 얹는다. 실행: `uvicorn server:app --host 0.0.0.0 --port 7860`.
"""

import importlib.util
import sys
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# 진입점 파일명(3_1542353.py)이 숫자로 시작해 식별자로 쓸 수 없어
# 경로 기반으로 로드한다 (일반 `from app import ...`는 SyntaxError).
_APP_PATH = Path(__file__).resolve().parent.parent / "3_1542353.py"
_spec = importlib.util.spec_from_file_location("app", _APP_PATH)
_app_module = importlib.util.module_from_spec(_spec)
sys.modules["app"] = _app_module
_spec.loader.exec_module(_app_module)

_CSS, _THEME, demo = _app_module._CSS, _app_module._THEME, _app_module.demo

app = FastAPI(title="AI 쇼핑 어시스턴트")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health():
    return {"status": "ok"}


gr.mount_gradio_app(app, demo, path="/", theme=_THEME, css=_CSS)
