# 배포 가이드

Docker + FastAPI + Prometheus/Grafana로 로컬에서 전체 스택을 구동하는 방법과,
(실행하지 않는) AWS EC2 GPU 배포 절차를 안내합니다.

## 구성

```
ai_mini/
  .dockerignore           # 빌드 컨텍스트 루트에 위치해야 해서 루트에 유지
  deploy/
    server.py              # app.py의 Gradio Blocks를 FastAPI에 마운트 (+ /health, /metrics)
    Dockerfile              # nvidia/cuda 베이스, Python 3.10, uvicorn deploy.server:app 실행
    docker-compose.yml       # app + prometheus + grafana
    prometheus/prometheus.yml
    grafana/provisioning/    # 데이터소스 + 대시보드 자동 프로비저닝
```

배포 관련 파일은 `deploy/` 아래 모아 두었습니다 — 아래 모든 `docker compose`
명령은 **프로젝트 루트에서 `cd deploy`한 뒤** 실행합니다.

- `app.py`의 로직/UI는 변경하지 않았습니다. `server.py`는 `app.py`의 `demo`
  (gr.Blocks)를 그대로 import해 FastAPI 위에 마운트만 합니다.
- 이미지 인식 소요 시간, 챗봇 응답 생성 시간, 네이버 API 호출 성공/실패 수를
  Prometheus 커스텀 메트릭으로 계측합니다 (`app.py`의
  `IMAGE_UPLOAD_DURATION`/`CHAT_RESPONSE_DURATION`/`NAVER_API_CALLS`).

## 사전 준비

1. **WSL2 + Docker Desktop 설치** (GPU 패스스루를 위해 WSL2 백엔드 필수). 최초
   설치 절차는 별도 안내 참고 (Windows 11 기준, `wsl --install` → Docker
   Desktop GUI 설치 → GPU 패스스루 확인까지).
2. 설치 확인:
   ```bash
   docker --version
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```
   컨테이너 안에서도 GPU 정보가 출력되면 준비 완료입니다.
3. `.env` 파일 (네이버 쇼핑 API 키)이 프로젝트 루트에 있어야 합니다 (`.env.example` 참고).

## 로컬 실행

```bash
cd deploy
docker compose build
docker compose up -d
```

- 최초 빌드/실행 시 CUDA 베이스 이미지 + PyTorch(cu128) + 각 모델 가중치를
  새로 받기 때문에 수 GB 다운로드가 발생하고, 환경에 따라 수십 분 걸릴 수
  있습니다.
- 모델 가중치는 호스트의 huggingface/torch 캐시 디렉터리를 컨테이너에
  볼륨으로 마운트해 재사용하므로, 로컬에서 이미 한 번 실행해본 적이 있다면
  재다운로드하지 않습니다.
- `data/chroma`는 볼륨 마운트로 영속화되어 컨테이너를 재시작해도 색인
  데이터가 유지됩니다.

### 확인

| 서비스 | 주소 |
| --- | --- |
| Gradio 앱 | http://localhost:7860 |
| 헬스체크 | http://localhost:7860/health |
| Prometheus 메트릭(raw) | http://localhost:7860/metrics |
| Prometheus UI | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin / admin) |

Grafana에는 Prometheus 데이터소스와 "AI 쇼핑 어시스턴트" 대시보드가 자동으로
프로비저닝되어 있습니다 (HTTP 요청 수, 이미지 업로드/챗봇 응답 처리 시간,
네이버 API 성공/실패 수).

컨테이너 안에서 GPU가 실제로 잡히는지 재확인하려면(`deploy/` 안에서):

```bash
docker compose exec app nvidia-smi
```

### 종료

`deploy/` 안에서:

```bash
docker compose down
```
(볼륨은 `-v` 옵션을 주지 않는 한 삭제되지 않습니다. `data/chroma`는 호스트
경로에 그대로 남습니다.)

## AWS EC2 GPU 배포 (선택, 비용 발생 — 실행하지 않음)

**이 프로젝트는 로컬 무료 검증까지만 실제로 수행했습니다.** 아래는 실제 서비스로
배포할 경우의 절차를 참고용으로 남긴 것이며, 실제 AWS 리소스를 생성하지
않았습니다. GPU 인스턴스(G4dn/G5/P4 등)는 AWS 프리티어 대상이 아니므로 과금이
발생합니다 — 진행 전 비용을 반드시 확인하세요.

1. **AMI 선택**: Deep Learning AMI (Ubuntu, NVIDIA 드라이버 + Docker + NVIDIA
   Container Toolkit 사전 설치) — 예: `ami-060449aa9aa36d665` (리전/시점에 따라
   최신 AMI ID 확인 필요).
2. **인스턴스 유형**: GPU 인스턴스(G4dn.xlarge 이상 권장). 프리티어(t2.micro/
   t3.micro)는 GPU가 없어 이 프로젝트를 못 돌립니다.
3. **보안 그룹 인바운드 규칙**: 7860(앱), 9090(Prometheus), 3001(Grafana)
   포트를 필요한 IP 범위로만 열어둡니다 (0.0.0.0/0 전체 공개는 지양).
4. **접속 및 배포**:
   ```bash
   ssh -i <key.pem> ubuntu@<EC2-퍼블릭-IP>
   git clone <repo-url>
   cd ai_mini
   cp .env.example .env   # 네이버 API 키 입력
   cd deploy
   docker compose up -d
   ```
5. **GPU 확인**: `docker compose exec app nvidia-smi`.
6. 운영 배포 시 추가로 고려할 사항(이번 범위 밖, 실행하지 않음): HTTPS(리버스
   프록시 + 인증서), Grafana 기본 비밀번호 교체, 이미지 태그/버전 관리,
   오토스케일링/로드밸런싱.

## 보안 참고
- `.env`는 `.dockerignore`/`.gitignore`에 모두 포함되어 이미지나 저장소에
  들어가지 않습니다. 컨테이너에는 `docker-compose.yml`의 `env_file`로만
  주입됩니다.
- `docs/for_claude/`, `problem/`(로컬 조사용 이미지) 등은 `.dockerignore`로
  이미지 빌드에서 제외됩니다.
