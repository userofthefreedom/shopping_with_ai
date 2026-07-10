# Project Rules

## 프로젝트 개요
AI 쇼핑 어시스턴트: 사용자가 상품 사진을 업로드하면 AI가 상품을 인식하고,
관련 정보 및 유사 상품을 추천하며, 대화형 챗봇을 통해 쇼핑 경험을 지원하는
Gradio 기반 웹 서비스.

## Must Read Before Coding
- Read `docs/for_claude/PRD.md` before implementation. (무엇을 만들지)
- Read `docs/for_claude/SPEC.md` before implementation. (어떻게 만들지)
- Read `docs/for_claude/PHASE.md` before implementation. (지금 어떤 Phase인지)
- Read `docs/for_claude/PLAN.md` before implementation. (현재 Phase의 구현 계획)
- Read `docs/for_claude/PROGRESS.md` if it exists. (이전 세션에서 무엇까지 했는지)

## 문서 위치 규칙
- Claude와의 소통/작업 지시를 위해 만드는 모든 문서(PRD, SPEC, PHASE, PLAN, PROGRESS,
  TEST_RESULT 등)는 `docs/for_claude/`에 만든다.
- `docs/for_claude/`는 git에 커밋하지 않는다 (`.gitignore` 처리됨). 이 폴더의 문서는
  로컬 작업 기록용이며 저장소의 배포 대상 코드가 아니다.
- `CLAUDE.md`만 예외로 프로젝트 루트에 유지한다 (Claude Code가 세션 시작 시 자동으로
  읽는 규칙 파일이라 이동하면 자동 로드가 끊긴다).

## 기술 스택
- 언어/런타임: Python 3.10+ (Miniforge3 conda 가상환경 필수 — 이 프로젝트는 venv를
  쓰지 않고 conda 환경으로만 관리한다)
- 백엔드 라이브러리: PyTorch, ultralytics(YOLOv8), transformers, scikit-learn,
  ChromaDB, LangChain
- 프론트엔드: Gradio (단일 웹 UI)
- 벡터 DB: ChromaDB
- 외부 API: 네이버 쇼핑 검색 API
- 모델: YOLOv8(DeepFashion2 사전학습), CLIP ViT-B/32, KO-SRoBERTa,
  Llama 3.2 Korean Bllossom 3B

## 설치 / 실행 / 테스트 / 빌드 명령어
- conda 환경 활성화: `conda activate shopping_assistant`
- 설치: `pip install -r requirements.txt` (conda 환경 활성화 후 실행)
- 실행: `python 3_1542353.py` (localhost:7860)
- 테스트: `pytest` (Phase별 유닛 테스트는 `tests/` 아래에 작성)
- 별도 빌드 단계 없음 (Docker화는 Phase 9 배포 단계에서만 다룸)

## 코드 스타일과 컨벤션
- 모듈 단위로 분리: 탐지(detection), 색상(color), 검색(search/recommend),
  챗봇(chat), API 연동(integrations), UI(3_1542353.py)는 각각 별도 파일/패키지로 둔다.
- 함수/파일 이름은 영문, 사용자 노출 문자열(챗봇 응답, UI 라벨)은 한국어.
- 예외 상황은 조용히 무시하지 않고 명시적으로 처리하거나 로그를 남긴다.

## 인증키와 .env 관리
- 네이버 쇼핑 API Client ID/Secret은 `.env`에 저장하고 `os.environ`으로 읽는다.
- `.env`는 git에 커밋하지 않는다 (`.gitignore`에 포함).
- `.env.example`에 필요한 키 이름만 (값 없이) 기록해 둔다.

## Workflow Rules (Harness)
- 한 번에 하나의 Phase만 구현한다. (`PHASE.md` 기준, WIP = 1)
- 구현 전에 조사 → `PLAN.md` 작성 순서를 따른다. **승인 대기 없이** 계획 수립 후 바로
  구현으로 진행한다 — 사용자가 결과를 사후에 검토하는 방식으로 작업한다 (git
  commit/push만 예외, 아래 참고).
- 테스트(및 해당 Phase의 Acceptance Criteria 충족)와 실행 확인 없이 완료로 보지 않는다.
- `PLAN.md`에 없는 파일은 수정하지 않는다. 꼭 필요하면 이유를 설명하고 진행한다
  (승인 대기 불필요).
- 기존에 구현된 기능을 수정해야 할 때도 이유를 설명하고 진행한다 (승인 대기 불필요).
- **git commit/push는 사용자가 직접 한다.** Claude는 커밋/푸시하지 않는다. Phase
  완료 후 `PROGRESS.md`를 갱신하고 "커밋 준비 완료" 상태로 작업 트리에 남겨둔다.
- 요구사항(PRD)이나 설계(SPEC)가 바뀌면 해당 문서를 함께 갱신한다.

## Version 2 진행 방식 (Phase 9~ )
- Version 1(Phase 0~8)은 완료됨. Version 2는 성능 개선(Phase 9) → UI/UX 개선
  (Phase 10) → 배포(Phase 11) 순서로, 여전히 WIP=1을 지키며 하나씩 진행한다
  (`PRD.md`/`SPEC.md`/`PHASE.md`의 "Version 2" 섹션 참고).
- Harness/Multi-Agent Orchestration 방법론은 `C:\Users\SSAFY\Desktop\TIL\agent&harness\README.md`
  를 참고한다. Claude Code 안에서는 Plan Mode(Planner), 메인 세션(Coder),
  `Agent(subagent_type: fork)`(Researcher), `/code-review`(Reviewer), `/verify`
  (Verifier)로 역할을 매핑해 쓴다 — 새로운 도구를 만들 필요는 없다.
- 세 작업(성능/UI-UX/배포)은 서로 파일이 겹치거나 영향을 주므로 동시에 병렬로
  진행하지 않는다.
