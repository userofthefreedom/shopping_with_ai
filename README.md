# AI 쇼핑 어시스턴트

사용자가 상품 사진을 업로드하면 AI가 상품을 인식하고, 관련 정보와 유사 상품을
추천하며, 대화형 챗봇으로 쇼핑 경험을 돕는 어시스턴트 서비스입니다.

## 주요 사용자
- 일반 소비자
- 패션/이커머스 플랫폼, 개인화 추천 서비스 제공자

## 기능 흐름
1. 상품 사진 업로드 → YOLOv8으로 상품 탐지 및 카테고리 분류 (예: "빨간색 니트 가디건")
2. 상품 정보 요청 → LLM 챗봇이 특징/스타일을 자연어로 설명
3. 가격 문의 → 네이버 쇼핑 API + CLIP 유사도 검색으로 유사 상품 가격 제공
4. 예산/저렴한 상품 요청 → ChromaDB 벡터 검색 + 자연어 예산 필터링으로 추천
5. 최종 선택 → 실제 쇼핑몰 구매 링크로 연결

### 추가 기능
- 옷 색상 인식 (탐지 영역 색상 추출 → 한국어 색상명 변환 → 검색 키워드에 반영)
- 자연어 예산 조건 파싱 ("10만원 이하", "더 저렴한" 등) 및 가격순 필터링/정렬
- CLIP 기반 이미지 유사도 검색 (업로드 이미지와 유사한 상품 이미지 검색)

## 기술 스택
- 백엔드: Python, PyTorch, ultralytics(YOLOv8), transformers, scikit-learn,
  ChromaDB, LangChain
- 프론트엔드: Gradio (웹 UI)
- 벡터 DB: ChromaDB
- 외부 API: 네이버 쇼핑 검색 API
- 모델: YOLOv8(DeepFashion2 사전학습, 13개 패션 카테고리), CLIP ViT-B/32,
  KO-SRoBERTa, Llama 3.2 Korean Bllossom 3B

## 프로젝트 구조
```
ai_mini/
  app.py                     # Gradio 진입점
  requirements.txt
  .env.example
  detection/                 # YOLOv8 탐지, 색상 인식
  search/                    # CLIP/ChromaDB 유사도 검색, 네이버 API, 예산 필터링
  chat/                      # LLM 챗봇
  data/chroma/               # ChromaDB persist 디렉터리
  tests/                     # 유닛 테스트
```

## 시작하기

### 1. conda 환경 준비 (Miniforge3)
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

### 2. 패키지 설치
```bash
# PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 나머지 패키지
pip install -r requirements.txt
```

### 3. 환경 변수 설정
```bash
cp .env.example .env
# .env에 네이버 쇼핑 API Client ID/Secret 입력
```

### 4. 실행
```bash
python app.py
# http://localhost:7860 접속
```

### 5. 테스트
```bash
pytest
```

## 성능/품질 기준
- 객체 탐지 정확도: mAP@0.5 기준 70% 이상 권장
- 추천 품질: 코사인 유사도 기반 상위 N개 상품 반환
- 실시간성: 이미지 업로드 후 3초 이내 결과 제공

## 보안/개인정보
- 사용자 업로드 이미지, 검색 이력 등은 로컬 환경 또는 안전한 서버에 저장
- 네이버 API 키 등 민감 정보는 `.env`로 분리 관리하며 git에 커밋하지 않음
