# AI 쇼핑 어시스턴트

사진 한 장을 올리면 상품을 인식하고, 비슷한 상품을 찾아 가격을 비교해주고,
대화로 궁금한 걸 물어볼 수 있는 AI 쇼핑 어시스턴트입니다.

## 목차
- [1. 프로젝트 소개](#1-프로젝트-소개)
- [2. 빠른 시작](#2-빠른-시작)
- [3. 상세 설명](#3-상세-설명)
  - [3.1 동작 흐름](#31-동작-흐름)
  - [3.2 프로젝트 구조](#32-프로젝트-구조)
  - [3.3 기술 스택](#33-기술-스택)
  - [3.4 성능/품질 기준](#34-성능품질-기준)
- [4. 배포](#4-배포)
- [5. 보안/개인정보](#5-보안개인정보)
- [6. 개발 히스토리](#6-개발-히스토리)

## 1. 프로젝트 소개

사용자가 상품 사진을 업로드하면 AI가 상품을 인식하고, 관련 정보와 유사 상품을
추천하며, 대화형 챗봇으로 쇼핑 경험을 돕는 Gradio 기반 단일 웹 서비스입니다.
탐지부터 추천, 챗봇 응답까지 하나의 업로드 → 대화 흐름 안에서 전부 처리됩니다.

**핵심 기능 (사용 흐름)**
1. 상품 사진 업로드 → YOLOv8으로 상품 탐지 및 카테고리 분류 (예: "빨간색 니트 가디건")
2. 상품 정보 요청 → LLM 챗봇이 특징/스타일을 자연어로 설명
3. 가격 문의 → 네이버 쇼핑 API + CLIP 유사도 검색으로 유사 상품 가격 제공
4. 예산/저렴한 상품 요청 → 자연어 예산 필터링으로 조건에 맞는 추천만 남기기
5. 최종 선택 → 추천 카드에서 실제 쇼핑몰 구매 링크로 이동

**추가 기능**
- 옷 색상 인식 (탐지 영역 색상 추출 → 한국어 색상명 변환 → 검색 키워드에 반영)
- 자연어 예산 조건 파싱 ("10만원 이하", "더 저렴한" 등) 및 가격순 필터링/정렬
- CLIP 기반 이미지 유사도 검색, KO-SRoBERTa 기반 텍스트 유사도 검색(로컬 벡터
  검색 우선, 부족할 때만 네이버 실시간 검색)
- "최신/신상/재고/실시간/현재/오늘" 같은 신선도 키워드 감지 시 챗봇이 알아서
  네이버 실시간 재검색을 트리거

**대상 사용자**
- 일반 소비자 — 상품 사진을 올려 비슷한 상품을 찾고 싶은 사람
- 참고 대상 — 패션/이커머스 플랫폼, 개인화 추천 서비스 제공자

## 2. 빠른 시작

사전 요구사항: conda(Miniforge3 권장), Python 3.10. GPU(NVIDIA, CUDA 12.8
드라이버)가 있으면 훨씬 빠르지만 없어도 CPU로 동작은 합니다.

### 1) conda 환경 준비 (Miniforge3)
```bash
conda create -n shopping_assistant python=3.10 -y
conda activate shopping_assistant
```

> **Git Bash(MINGW64)에서 `conda: command not found`가 뜨는 경우**
> Git Bash 새 터미널에는 conda가 기본으로 초기화되어 있지 않습니다. 아래처럼
> conda의 bash 훅을 먼저 불러온 뒤 activate 하세요 (Miniforge3 설치 경로가
> 다르면 경로를 맞게 바꾸세요).
> ```bash
> source /c/Users/SSAFY/miniforge3/etc/profile.d/conda.sh
> conda activate shopping_assistant
> ```
> 매번 새 터미널마다 다시 치기 번거로우면 `~/.bashrc`에 위 `source` 줄을
> 한 번만 추가해두면 이후로는 Git Bash를 열 때마다 자동으로 conda가
> 잡힙니다:
> ```bash
> echo 'source /c/Users/SSAFY/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc
> ```
> 환경이 이미 만들어져 있다면(`conda env list`로 확인) `conda create` 줄은
> 건너뛰고 `conda activate shopping_assistant`부터 하면 됩니다 — 이미 있는
> 이름으로 다시 `create`해도 기존 환경을 덮어쓰지 않고 에러만 내고
> 끝나므로 안전합니다.

### 2) 패키지 설치
```bash
# PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 나머지 패키지
pip install -r requirements.txt
```

### 3) 환경 변수 설정
```bash
cp .env.example .env
# .env에 네이버 쇼핑 API Client ID/Secret 입력
```

### 4) 실행
```bash
python app.py
# http://localhost:7860 접속
```

### 5) 테스트
```bash
pytest
```

Docker로 실행하고 싶다면(로컬 Docker Compose로 앱+Prometheus+Grafana 모니터링
스택을 한 번에 띄우는 방법, AWS EC2 GPU 배포 가이드 포함) →
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) 참고.

## 3. 상세 설명

### 3.1 동작 흐름

```
사용자 이미지 업로드
        │
        ▼
YOLOv8 탐지 (bbox + 13개 패션 카테고리)
        │
        ▼
색상 인식 (K-means + HSV 규칙 → 한국어 색상명)
        │
        ▼
"카테고리 + 색상(+ 세부 종류)" 설명 생성  ──▶  CLIP 제로샷으로 세부 종류 추정
        │                                       (예: "긴팔 아우터" → "패딩")
        ▼
로컬 텍스트 벡터 검색 우선 (KO-SRoBERTa, ChromaDB)
        │
        ├─ 충분함 ──────────────────────────────────┐
        │                                          │
        └─ 부족함                                   │
              │                                    │
              ▼                                    │
      네이버 쇼핑 API 실시간 검색                     │
      (이미지/텍스트 컬렉션에 동시 색인)               │
              │                                    │
              ▼                                    │
      CLIP 이미지 유사도로 후보 재정렬 ◀──────────────┘
              │
              ▼
      (대화 중 예산 조건이 있으면) 자연어 예산 필터링/정렬
              │
              ▼
      LLM 챗봇(Llama 3.2 Bllossom)이 대화 맥락 반영해 자연어로 응답
              │
              ▼
      추천 상품 카드 + 실제 쇼핑몰 구매 링크
```

- **탐지 + 색상**: YOLOv8(DeepFashion2 사전학습)으로 bbox와 카테고리를 얻고,
  해당 영역을 K-means로 대표색을 뽑아 HSV 규칙으로 한국어 색상명(카키/남색 등
  포함)으로 변환합니다.
- **검색은 "로컬 우선"**: 매 요청마다 네이버 API를 부르면 느리고 요청 수도
  낭비되므로, 이미 색인된 상품을 KO-SRoBERTa 텍스트 임베딩으로 먼저 찾아보고
  (ChromaDB `product_text`), 결과가 충분하지 않을 때만 네이버 쇼핑 API를
  실시간으로 호출합니다. 이렇게 얻은 상품은 다음 요청을 위해 이미지(CLIP)/
  텍스트(KO-SRoBERTa) 컬렉션 양쪽에 병렬로 색인됩니다.
- **재정렬**: 텍스트 검색으로 좁힌 후보군을, 업로드한 이미지 자체와의 CLIP
  코사인 유사도로 다시 정렬해 시각적으로 더 비슷한 상품이 위로 오게 합니다.
- **대화**: "얼마야?", "더 저렴한 거 있어?" 같은 질문에는 예산 파서가 조건을
  뽑아 후보를 필터링하고, "최신/신상" 같은 신선도 키워드가 있으면 네이버를
  실시간으로 다시 호출합니다. 챗봇은 지금까지의 대화 맥락(어떤 상품 얘기
  중인지)을 유지한 채 자연어로 답합니다.

### 3.2 프로젝트 구조
```
ai_mini/
  app.py                     # Gradio 앱 진입점 — 업로드→탐지→검색→챗봇→추천 흐름 전체를 연결
  requirements.txt
  .env.example
  detection/                 # YOLOv8 탐지(detect.py), 색상 인식(color.py)
  search/                    # CLIP 이미지 유사도(clip_search.py),
                              # KO-SRoBERTa 텍스트 검색(text_search.py),
                              # 네이버 쇼핑 API(naver_api.py), 예산 필터링(budget.py)
  chat/                      # LLM 챗봇 (llm.py, LangChain 파이프라인)
  data/chroma/               # ChromaDB persist 디렉터리 (git ignore)
  tests/                     # 유닛 테스트
  deploy/                    # Docker/FastAPI/Prometheus/Grafana 배포 설정
  docs/                      # 공개 문서 (배포 가이드 등)
```

### 3.3 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| 백엔드 | Python, PyTorch, ultralytics, transformers, scikit-learn, ChromaDB, LangChain, FastAPI |
| 프론트엔드 | Gradio |
| 벡터 DB | ChromaDB (이미지 임베딩 `product_images`, 텍스트 임베딩 `product_text`) |
| 외부 API | 네이버 쇼핑 검색 API |
| 배포/모니터링 | Docker, Docker Compose, Prometheus, Grafana |

**모델별 역할**

| 모델 | 역할 |
| --- | --- |
| YOLOv8 (DeepFashion2 사전학습) | 상품 탐지 + 13개 패션 카테고리 분류 |
| CLIP ViT-B/32 | 이미지 임베딩 기반 유사 상품 검색, 제로샷 세부 종류(subtype) 분류 |
| KO-SRoBERTa | 상품 설명 텍스트 임베딩(로컬 벡터 검색 우선순위 판단) |
| Llama 3.2 Korean Bllossom 3B | 대화형 챗봇 응답 생성 |

### 3.4 성능/품질 기준
- 객체 탐지 정확도: mAP@0.5 기준 70% 이상 권장
- 추천 품질: 코사인 유사도 기반 상위 N개 상품 반환
- 실시간성: 이미지 업로드 후 3초 이내 결과 제공

## 4. 배포

로컬에서 무료로 Docker Compose 하나로 앱 + Prometheus + Grafana 모니터링
스택을 통째로 띄워볼 수 있습니다(`deploy/` 아래에 구성). 실제 운영 배포가
필요한 경우를 위한 AWS EC2 GPU 절차도 가이드로 정리해 두었습니다(비용이
발생하므로 실행은 하지 않았고 문서만 제공). 자세한 실행 명령/포트/확인 방법은
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)를 참고하세요.

## 5. 보안/개인정보
- 사용자 업로드 이미지, 검색 이력 등은 로컬 환경 또는 안전한 서버에 저장
- 네이버 API 키 등 민감 정보는 `.env`로 분리 관리하며 git에 커밋하지 않음

## 6. 개발 히스토리

이 프로젝트는 기본 기능부터 완성한 뒤(Version 1), 실사용 관점에서 발견한
문제들을 하나씩 찾아 고치는 방식(Version 2)으로 발전했습니다.

**Version 1 — 기본 파이프라인 구축**
탐지(YOLOv8) → 색상 인식 → 네이버 쇼핑 API 연동 → CLIP 유사도 검색 → 예산
필터링 → LLM 챗봇 → Gradio 통합까지, 핵심 기능을 하나씩 쌓아 end-to-end로
동작하는 서비스를 완성했습니다. 마지막 성능/품질 검증 단계에서 "이미지 업로드
후 3초 이내 응답"이라는 목표에 미달한다는 걸 발견했고, 그 원인(네이버 상품
이미지를 순차적으로 다운로드+임베딩하는 구조)을 다음 단계 과제로 남겼습니다.

**Version 2 — 실사용 검증에서 나온 문제를 하나씩 해결**
- **응답 속도**: Version 1에서 남긴 병목을 병렬 다운로드 + 이미 색인된 상품
  재다운로드 스킵으로 개선했습니다.
- **색상/추천 정확도**: 실제 사진(네이비 크롭 티셔츠, 카키 필드 재킷)으로
  직접 검증하다가 "남색"이 "파란색"으로, "카키"가 "회색"으로 오분류되는 걸
  발견해 HSV 색상 규칙을 단계적으로 보정했고, 카테고리를 그대로 검색어로
  쓰면 원본과 다른 스타일이 추천되는 편향도 발견해 동의어 검색으로 후보군을
  넓혔습니다.
- **채팅 타이밍 버그**: 이미지 분석이 끝나기 전에 채팅을 보내면 빈 상태로
  "정보 없음"이라 답하는 문제를 재현으로 확인하고, 분석 중 입력을 잠그는
  방식으로 해결했습니다.
- **원본 스펙 재검토**: 원본 요구사항 문서를 다시 검토하다가 Version 1이
  KO-SRoBERTa 기반 텍스트 벡터 검색 레이어를 빠뜨렸다는 걸 발견해, "로컬
  벡터 검색을 먼저 시도하고 부족할 때만 네이버 실시간 검색"하는 구조로
  아키텍처를 바꿨습니다. 이 과정에서 CLIP 제로샷 기반 세부 종류 분류(예:
  "아우터" → "패딩")도 함께 추가했습니다.
- **품질/보안 점검**: 배포 전 자체 코드 리뷰와 보안 리뷰를 진행해 Stored
  XSS, 세션 간 챗봇 기록이 섞이는 상태 오염 버그 등 실제 문제들을 여러 건
  발견하고 수정했습니다.
- **디자인 고도화**: 기능 위주였던 화면을 무신사·29CM 스타일의 미니멀
  디자인으로 다듬었습니다.
- **배포**: WSL2 + Docker Desktop 설치부터, FastAPI로 감싼 뒤 Prometheus/
  Grafana 모니터링까지 붙여 실제 GPU 패스스루가 되는 로컬 Docker 스택으로
  검증했고, 이후 루트가 복잡해 보인다는 피드백을 받아 배포 관련 파일들을
  `deploy/` 폴더 하나로 정리했습니다.

전체 Phase 단위의 상세 작업 기록은 이 저장소에는 포함되어 있지 않은 로컬
작업 노트(`docs/for_claude/`)에서 관리됩니다 — 여기서는 어떤 문제를 어떻게
풀었는지 흐름만 남겼습니다.
