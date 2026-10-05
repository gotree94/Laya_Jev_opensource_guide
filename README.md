# Laya 가이드 — Jev를 오픈소스로 Self-Hosting 하기

> 기준 문서: GitHub `NandhaKishorM/laya`, HuggingFace `convaiinnovations/laya` <br>
> 참고 영상: 「Laya 진짜 추천합니다. Jev를 오픈소스로 쓰는법」 <br>
> 라이선스: **Apache-2.0** (상업적 사용 가능, Laya / Laya-MLX / Laya-CoreML / Ollaya 모두) <br>

* https://youtu.be/ZktfWXfOwIY

---

## 목차

1. [Laya란 무엇인가](#1-laya란-무엇인가)
2. [핵심 개념 정리](#2-핵심-개념-정리)
3. [Laya vs Jev 비교](#3-laya-vs-jev-비교)
4. [환경 설치](#4-환경-설치)
5. [빠른 시작 — Python SDK](#5-빠른-시작--python-sdk)
6. [CLI 사용법](#6-cli-사용법)
7. [질문 스키마 — choice / score / noul](#7-질문-스키마--choice--score--noul)
8. [자기호스팅 서버 (laya-serve)](#8-자기호스팅-서버-laya-serve)
9. [기존 Jev 클라이언트 마이그레이션](#9-기존-jev-클라이언트-마이그레이션)
10. [파인튜닝 — 핵심](#10-파인튜닝--핵심)
11. [성능 및 벤치마크](#11-성능-및-벤치마크)
12. [한계점 — 반드시 읽을 것](#12-한계점--반드시-읽을-것)
13. [활용 예제](#13-활용-예제)
14. [트러블슈팅](#14-트러블슈팅)
15. [참고 링크](#15-참고-링크)

---

## 1. Laya란 무엇인가

**Laya는 Jev를 대체하는 오픈 가중치 "판단(Decision) 모델"입니다. 실행 환경(게임 엔진 등)이 아닙니다.**

```
┌─────────────────────────────────────────────────────────────┐
│  Jev (TypeSafe)          │  Laya (오픈소스)                  │
│  ─────────────────────   │  ──────────────────────          │
│  폐쇄 가중치              │  가중치 + 코드 + 노트북 전부 공개    │
│  TypeSafe 클라우드만 운영  │  로컬 / 사내망 / 자체 GPU 에서 구동   │
│  $0.042 / 100만 토큰      │  $0 (자사 인프라 비용만)            │
│  POST /v1/systemone      │  POST /v1/systemone  (동일!)       │
└─────────────────────────────────────────────────────────────┘
```

### 설계 철학
- **비자기회귀(Non-autoregressive)**: 토큰을 하나씩 생성하지 않고 **1회 forward pass**로 모든 질문에 답함
- **텍스트를 만들지 않음** → 파싱 필요 없음, 환각(hallucination) 구조적으로 불가능
- 출력은 사람이 쓴 `"confidence": 0.9` 같은 **문자열이 아니라 실제 확률 분포**
- 강화학습(RLCD, Reinforcement Learning for Calibrated Decisions)으로 **교정(calibration)**된 확률 제공

### 질문 유형 3가지

| 유형 | 역할 | 반환값 |
|---|---|---|
| `choice` | 선택지 중 하나 고르기 | 선택된 라벨 + 선택지별 확률 |
| `score` | 순서 있는 등급에 위치 매기기 | 기대 등급 + 등급별 확률 |
| `noul` | 명제가 참인지 판단 | `P(true)` (0.0 ~ 1.0) |

---

## 2. 핵심 개념 정리

### 상태(State) 와 질문(Questions)

Laya는 **상태 1개 + 타입이 지정된 질문 여러 개**를 받아 한 번에 처리합니다.

```python
state = {                      # str, dict, list 모두 가능
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today."
}

questions = {
    "department": {"type": "choice", "instructions": "...", "criteria": {...}},
    "urgency":    {"type": "score",  "instructions": "...", "criteria": [...]},
    "churn_risk": {"type": "noul",   "instructions": "..."},
}
```

### Router — 자동 언어 라우팅

`Router()`는 요청의 **문자 스크립트와 언어**를 감지해 최적 체크포인트를 자동 선택합니다.

- 라우팅 판정만 하는 경우 **체크포인트를 다운로드하지 않음** (수 ms, 완전 오프라인)
- 사전 정의된 생성 함수 없이 순수 Python으로 동작 (의존성 0)
- `Router(default="multilingual")` 또는 `LAYA_DEFAULT_MODEL=multilingual`로 기본값 변경

### 체크포인트 3종

| 이름 | 인코더 | 파라미터 | 컨텍스트 | 용도 |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 | 영어 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 (최대 8,192) | 100+개 언어, 2배 빠름 |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | 파인튜닝된 판정 워크플로 |

> HF 저장소: `convaiinnovations/laya` (루트=영어, `multilingual/`, `typed-decisions/` 서브폴더)

**왜 Router가 필요한가**
- 영어 체크포인트는 비라틴 문자에서 붕괴 ( Khmer 언어에서 정확도 0.000인데 신뢰도 0.952 — **틀리면서도 확신**)
- 다국어 체크포인트는 영어에서 약간 저하
- Router는 forward pass **전에** `<0.5ms` 로 스크립트를 검사해서 잘못된 체크포인트를 쓰지 않음
- 51개 언어 중 영어+사용가능: `laya` 23/51, `laya-multilingual` 48/51, **Router 48/51**

---

## 3. Laya vs Jev 비교

### 성능 / 운영 비교

| 항목 | Laya | Jev |
|---|---|---|
| 공개 범위 | **가중치·코드·파인튜닝 노트북·평가 코드 전부 공개** | 가중치 비공개, 호스팅 API만 |
| 실행 위치 | 로컬 / 자체 서버 | TypeSafe 클라우드 |
| 비용 | 자체 호스팅 (한계 0) | 입력 100만 토큰당 $0.042 |
| 지연 (질문 1개, p50) | **32.8 ms** (T4) | 236 ~ 276 ms (제3자 측정) |
| 컨텍스트 | 512 (영어) / 1024, 다국어 8192까지 | 64k 토큰 |
| 선택지 개수 | 약 20개 넘으면 정확도 급락 (HTTP 상한 100) | 최대 255개 |
| 신뢰도 교정(ECE) | **0.081** (Jev 대비 3배 우수) | 0.246 |
| 라이선스 | Apache-2.0 | 상용 API |

### 정확도 벤치마크

**typed-decisions (2,000건, 4개 워크플로)**

| 모델 | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya-typed-decisions` (파인튜닝) | **0.766** | 0.471 | 0.062 | 0.213 | 0.242 |
| Jev 1.13.0 (발표값) | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 |
| teacher ceiling | 0.735 | — | — | — | — |
| `laya` (영어, 제로샷) | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| majority | 0.461 | — | — | — | — |
| random | 0.318 | — | — | — | — |

> 파인튜닝된 Laya가 Jev(0.727)와 teacher ceiling(0.735) 모두 넘습니다.
> 워크플로별: invoice 0.804 / security 0.766 / customer service 0.764 / agent-trace 0.730
> 원시 유형별: noul 0.857 / choice 0.733 / score 0.723

**Head-to-Head (Laya routed vs Jev 1.13.0)**

| Task | Jev | Laya (routed) | 차이 |
|---|---|---|---|
| typed-decisions | 0.727 | **0.766** | +0.039 |
| AG News (4-class) | 0.910 | **0.950** | +0.040 |
| DAIR Emotion (6-class) | 0.480 | **0.595** | +0.115 |
| Banking77 (선택지 72 vs 77) | **0.870** | 0.425 | **Jev 우위** |
| ECE (낮을수록 좋음) | 0.246 | **0.081** | 3× 우수 |
| p50 지연 1질문 | 236–276 ms | **32.8 ms** | 7.8× 빠름 |

**다국어 (MASSIVE intent, 20-class, random 0.05)**

| | `laya` | `laya-multilingual` |
|---|---|---|
| 영어 | **0.783** | 0.657 |
| 그 외 13개 언어 | 0.306 | **0.451** |
| XNLI 영어 | **0.860** | 0.843 |
| XNLI 그 외 14개 언어 | 0.521 | **0.731** |

---

## 4. 환경 설치

### 4.1 요구 사항

- **Python 3.10 이상** (huggingface_hub 1.x, transformers 5.x, torch 2.14가 요구)
- 첫 실행 시 HuggingFace Hub 접근 필요 (체크포인트 다운로드)
- 디바이스: CUDA / Apple Silicon MPS / XPU / CPU 모두 지원

### 4.2 Windows (PowerShell)

```powershell
# 1) 가상환경 생성 (Python 3.11 사용 예시)
py -3.11 -m venv .venv

# 2) 설치
.\.venv\Scripts\python.exe -m pip install laya

# 3) 설치 확인 (버전만 출력, 체크포인트 로드 안 함)
.\.venv\Scripts\python.exe -I -c "import laya; print(laya.__version__)"
```

> `-I` 는 현재 디렉터리를 import 경로에서 제외하므로, 로컬 사본이 설치를 가리는 것을 방지합니다.
> **설치와 실행은 반드시 같은 가상환경의 Python exe를 사용하세요.**

GitHub 개발 버전 설치:
```powershell
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/NandhaKishorM/laya.git"
```

### 4.3 macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install laya
.venv/bin/python -I -c "import laya; print(laya.__version__)"
```

> Debian/Ubuntu에서 `ensurepip` 오류 시: `sudo apt install python3-venv`

### 4.4 uv 사용 (권장)

```bash
uv venv --python 3.12
uv pip install laya
uv run python -I -c "import laya; print(laya.__version__)"

# GPU 백엔드 자동 선택
uv pip install laya --torch-backend=auto
# CPU 강제
uv pip install laya --torch-backend=cpu
# uv 프로젝트라면
uv add laya
```

### 4.5 선택적 extra (extras)

```bash
pip install "laya[serve]"        # HTTP 서버 (fastapi + uvicorn + python-multipart)
pip install "laya[mcp]"          # MCP 서버
pip install "laya[langchain]"    # LangChain / LangGraph 통합
pip install "laya[llamaindex]"   # LlamaIndex selector
pip install "laya[crewai]"       # CrewAI 라우팅
pip install "laya[onnx]"         # ONNX Runtime
pip install "laya[fast]"         # TileLang GPU 고속 경로
pip install "laya[structured]"   # pydantic (decide용)
```

### 4.6 디바이스별 PyTorch 선택

PyTorch는 OS/GPU에 맞는 빌드를 자동으로 고르지만, 직접 지정해야 할 때:

```powershell
# NVIDIA CUDA
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# Intel XPU (GPU 드버버 설치 후)
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/xpu
.\.venv\Scripts\python.exe -c "import torch; print(torch.xpu.is_available())"
```

디바이스 지정은 코드에서: `laya.load(..., device="cuda" | "mps" | "xpu" | "cpu")` 또는 `LAYA_DEVICE` 환경변수

### 4.7 Apple Silicon 최적화 (선택)

| 패키지 | 엔진 | 대상 | 공개된 속도 (M3 Max) |
|---|---|---|---|
| `laya-mlx` | MLX | Apple Silicon GPU | 7.39ms (다국어) / 13.42ms (영어) |
| `laya-coreml` | Core ML | Apple Neural Engine | 4.98ms (다국어) |

가중치는 FP16 변환본이 HuggingFace `aac6fef/laya-mlx` 등에 올라있습니다.

### 4.8 Ollaya — 결정 모델용 Ollama (선택)

```bash
# Rust 단일 바이너리, Linux/macOS/Windows (CPU + NVIDIA GPU)
ollaya pull laya:en
ollaya run laya:en
ollaya serve            # 데몬이 TypeSafe /v1/systemone 형식을 그대로 speaks
```

- 약 3MB ONNX 그래프만 배포하고, 원저자 HF 저장소의 커밋 고정 + sha256 검증된 원본 가중치를 읽음
- `TYPESAFE_BASE_URL=http://localhost:11435` 로 공식 SDK를 로컬로 향하게 함
- `kev`, `Decision 1.0`, `Qwen3Guard`, NLI 분류기 등 다른 결정 모델도 지원

### 4.9 Docker

```bash
# 저장소 클론 후
docker compose up                                  # compose.yaml
docker compose -f compose.cuda.yaml up              # GPU
docker compose -f compose.http.yaml up              # HTTP 모드
docker compose -f compose.modelscope.yaml up        # ModelScope 미러
```

NixOS / Nix 사용자:
```bash
nix run .#laya-serve
nix develop
```

---

## 5. 빠른 시작 — Python SDK

### 5.1 30초 minimal

```python
from laya import Router

router = Router()          # 첫 호출 시 선택된 체크포인트만 다운로드

result = router.predict(
    {"body": "We were billed twice. Please refund the duplicate."},
    {"billing": {"type": "noul", "instructions": "Does the user request a refund?"}},
)

print(result["answers"]["billing"]["noul"])    # P(true), 0.0 ~ 1.0
print(result["routing"]["model"])             # 'english'
```

### 5.2 전체 예시

```python
from laya import Router

router = Router(preload=True)      # 3개 체크포인트 전부 메모리에 상주 (서버용 권장)

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

res = router.predict(state, questions)
print("Department :", res["answers"]["department"]["choice"])       # -> billing (confidence 0.94)
print("Routing    :", res["routing"]["model"])                    # -> english
```

### 5.3 Agent 직접 사용 (Router 없이)

```python
import laya

agent    = laya.load("convaiinnovations/laya")                                # 영어
agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual")     # 다국어
agent_td = laya.load("convaiinnovations/laya", subfolder="typed-decisions") # 파인튜닝

result  = agent.predict(state, questions)
results = agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
result  = agent.predict_long(long_doc, questions, window=256)
```

### 5.4 체크포인트 강제 지정

```python
res_en = router.predict(state, questions)                                    # 자동
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया..."}, questions) # 자동 → multilingual
res_td = router.predict(state, questions, model="typed-decisions")          # 강제
```

### 5.5 라우팅만 (체크포인트 로드 없음, 수 ms)

```python
router = Router()

router.route({"body": "Der Kunde wurde zweimal belastet"}, questions).reason
# "Latin script but language looks like 'de', not English"

router.route({"body": "Esqueci minha senha"}).model
# "default"  ← 라틴 문자라도 짧으면 신호가 없음
```

### 5.6 이질적 배치 처리 (heterogeneous batch)

```python
requests = [
    {"state": "Please refund invoice 1",   "questions": questions},
    {"state": "تم خصم المبلغ مرتين",        "questions": questions},   # 아랍어 → multilingual
    {"state": "Please refund invoice 2",   "questions": questions},
]

# Router가 먼저 전체를 라우팅 → 체크포인트별 그룹핑 → 같은 스키마 그룹을 공유 forward pass로
results = Router(max_loaded=1).predict_batch(requests)
# 결과는 원래 입력 순서로 복원됨
```

각 항목은 `model`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len` 을 독립 지정 가능.
전체 라우팅만 필요하면 `route_batch(requests)`.

### 5.7 스키마 기반 decide

```python
schema = {
    "type": "object",
    "properties": {
        "dept":        {"type": "string", "enum": ["billing", "support"]},
        "urgency":     {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    }
}

agent.decide("...", schema=schema)
agent.decide_batch(ticket_texts, schema=schema)
router.decide_batch(states, schema=schema, return_details=True, batch_size=64)
```

### 5.8 프리셋 질문 세트

```python
import laya

agent.predict({"message": "..."}, laya.triage_questions())
agent.predict({"message": "..."}, laya.guard_questions())
agent.predict({"message": "..."}, laya.moderation_questions())
laya.router_questions()
```

### 5.9 토큰 예산 (token budget)

```python
# 긴 문서: 다국어 체크포인트는 기본 1024로 잘려나감 → 명시적으로 8192까지 올려야 함
result = router.predict(long_document, questions, model="multilingual", max_len=8192)
```

- `max_len` : state 에 사용할 수 있는 토큰
- `head_max_len` : 질문 + 선택지를 담는 head의 상한 (laya 192, 나머지 256)
- 실제 head 길이는 렌더링된 질문+선택지에 따라 달라짐 → **실사용 가능 길이는 `max_len - 실제_head_len - 1`**
- 선택지당 최소 약 4토큰 → 선택지 k ≈ `head_max_len / 4` 를 넘으면 head가 상한을 넘김
- 모든 표면(SDK/CLI/MCP/LangChain/서버)에서 per-call 지정 가능

### 5.10 선택지가 많을 때 — shortlist

```python
shortlist = laya.predict_shortlist(
    agent, state, questions,
    embed_fn=laya.embed_fn_from_agent(agent),
    k=20,
)
```

인코더 임베딩만 사용하므로 **추가 모델이 필요 없습니다.** 캐시는 `laya.cached_embed_fn(embed_fn)` (LRU 최대 4096).

### 5.11 Hook (전처리 / 마스킹 / 로깅)

```python
def log(ctx):
    print(ctx.model, ctx.results[0]["answers"], ctx.elapsed_ms)

agent = laya.load("convaiinnovations/laya", on_predict_end=log)

# 개인정보 마스킹 예시
def redact(ctx):
    for s in ctx.states:
        s["body"] = mask_pii(s["body"])

router.predict(state, questions, on_predict_start=redact)
```

지원 훅: `on_predict_start`, `on_predict_end`, `on_route`, `on_load`, `on_evict`, `on_error`
언제든 `ctx.skip(...)` 로 캐시된 답 반환 가능. `hooks`, `hooks_raise`, `hooks_timeout` 설정 가능.
HTTP 서버로는 callable 을 보낼 수 없어 해당 키는 **422로 거부**됩니다.

### 5.12 신뢰도 게이팅 (abstention)

```python
result = router.predict(state, questions, min_confidence=0.9)

answer = result["answers"]["department"]
answer["low_confidence"]            # True/False (게이트 발동 시에만 존재)
answer["abstention"]                # "passed" | "abstained" | "unevaluated"
answer["abstention_threshold"]      # 적용된 임계값
```

- 게이트는 `answer_confidence`(= max p) 를 읽음. 엔트로피 기반 `confidence` 가 아님
- `min_confidence` 미지정이면 응답이 **바이트 단위로 이전과 동일**
- `decide()` 는 게이트 미달 시 `None` 반환

### 5.13 배치 모드 성능 팁

```python
results = router.predict_batch(requests, batch_size=8, sort_by_length=True)
```

- forward pass 는 배치 내 가장 긴 길이에 맞춰 패딩됨 → 길이가 비슷한 것끼리 묶으면 패딩 낭비 감소
- 실측: Yelp 리뷰 128건, MPS, batch_size 8 → 5.35s → **3.04s** (1.77×), 라벨 64/64 동일
- `1 < batch_size < len(states)` 일 때만 의미 있음
- 배치 모양 변화로 임계값 근처에서 미세한 float 차이가 날 수 있음

### 5.14 TypeScript / JavaScript

```bash
# 서버 (Python)
pip install -e '.[serve]'
LAYA_HOST=127.0.0.1 LAYA_MODELS=english laya-serve

# TS 클라이언트
cd sdk/typescript
npm ci && npm run build
node examples/triage.mjs
```

| npm 패키지 | 용도 |
|---|---|
| `laya-client` | 자체 호스팅 Python `laya-serve` 로 HTTP 통신 |
| `laya-ts` | Python 서버 없이 JS 안에서 ONNX 런타임으로 직접 추론 (브라우저 포함) |

npm 릴리스는 저장소의 `laya-ts-v*` 태그에서 퍼블리시됩니다.

### 5.15 MCP 서버

```bash
pip install "laya[mcp]"
laya-mcp-server
# 또는
python -m laya.mcp.server
```

제공 도구: `laya_predict`, `laya_predict_batch`, `laya_route`, `laya_route_batch`, `laya_decide`, `laya_shortlist`, `laya_preset`, `laya_status`

클라이언트 설정:
```json
{
  "mcpServers": {
    "laya": {
      "command": "laya-mcp-server",
      "env": { "LAYA_DEVICE": "cpu" }
    }
  }
}
```

원격 모드 (클라이언트에 torch 불필요):
```bash
LAYA_BASE_URL=http://127.0.0.1:8000 laya-mcp-server
```
> 원격 모드에서는 `laya_shortlist` 사용 불가

### 5.16 LangChain / LangGraph

```python
from laya.integrations.langchain import LayaDecision, LayaGuardrail, LayaRouter

router = LayaRouter(
    criteria={"billing": "refunds", "tech": "bugs"},
    confidence_threshold=0.80,
    fallback="human_agent",
)
# conditional edge 로 사용
guardrail = LayaGuardrail(action="raise")     # 위반 시 LayaGuardrailError
decision  = LayaDecision(schema)              # 스키마 모양 값 반환
```

`batch()` / `abatch()` 는 `predict_batch` 를 사용 (공유 forward pass).

---

## 6. CLI 사용법

패키지 설치 시 `laya` 명령이 함께 설치됩니다.

```bash
# 1) 라우팅만 (체크포인트 다운로드 없음, 완전 오프라인, 수 ms)
laya "I was charged twice, please refund"

# 2) 전체 답변 (첫 실행 시 체크포인트 다운로드)
laya "Refactor this service" --predict

# 3) 언어 강제 지정
laya "Mein Konto wurde zweimal belastet" --lang de

# 4) 언어 힌트 (soft — 감지 안 되면 내장 감지로 폴백)
laya "My payment failed twice" --lang-guess en

# 5) 체크포인트 고정
laya "My payment failed twice" --model ml

# 6) 내장 프리셋 (triage / email / guard / moderation / router)
laya "My payment failed twice" --preset triage

# 7) 신뢰도 게이팅
laya "Refund my card" --predict --min-confidence 0.9

# 8) 파일 배치 처리 (한 줄에 하나씩)
laya --batch tickets.txt --predict --json
laya --batch tickets.txt --predict --batch-size 8 --sort-by-length
cat tickets.txt | laya --batch - --predict --json

# 9) 직접 만든 질문 세트 (JSON)
laya "Where is my card" --questions intents.json
laya "..." --questions q.json --head-max-len 384

# 10) 인터랙티브 모드
laya
```

### `--questions` JSON 형식

```json
{
  "state_key": "body",
  "questions": {
    "dept": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": { "billing": "refunds", "tech": "bugs" }
    }
  }
}
```

`state_key` 생략 가능(기본 `request`).

### 주요 플래그

| 플래그 | 설명 |
|---|---|
| `--predict` | 라우팅된 체크포인트를 로드해 실제 예측 |
| `--json` | JSONL 출력 |
| `--batch FILE` | 배치 처리 (`-` 이면 stdin) |
| `--batch-size N` | forward pass 크기 제한 |
| `--sort-by-length` | 길이相近끼리 그룹핑 (1 < batch_size < 요청수 필요) |
| `--max-len` / `--head-max-len` | per-call 토큰 예산 |
| `--min-confidence` | abstention 게이트 |
| `--lang` (결정적) / `--lang-guess` (부드러운 힌트) | 언어 지정 |
| `--model` | 체크포인트 고정 (이름/별칭/대소문자 무관하게 해석) |
| `--preset` | `triage`, `email`, `guard`, `moderation`, `router` |
| `--questions` | 커스텀 질문 세트 |

### 웹 GUI 플레이그라운드

```bash
pip install "laya[serve]"
python examples/server.py        # http://127.0.0.1:8000
```

2분할 GUI (폼/JSON 편집 → Ctrl+Enter 실행 → 전체 분포와 교정된 신뢰도 표시, curl/Python 복사)
+ `/predict`, `/predict/batch` JSON API. `GET /models` 로 체크포인트/별칭 목록 확인.

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "state": {"body": "We were billed twice for March. Please refund it today."},
  "questions": {
    "department": {"type":"choice","instructions":"Which department?",
      "criteria": {"billing":"invoices, payments, refunds","other":"everything else"}},
    "urgency": {"type":"score","instructions":"How urgent?",
      "criteria": ["not urgent","soon","critical"]}
  }
}' | python -m json.tool
```

옵션: `--no-preload` (지연 로드), `--device cuda|cpu|mps`

---

## 7. 질문 스키마 — choice / score / noul

### 공통 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `type` | str | `"choice"` / `"score"` / `"noul"` |
| `instructions` | str | 질문 문장 |
| `criteria` | dict/list | 선택지 또는 등급 정의 |
| `option_order` | list | 표시 순서만 바꾸기 (위치 편향 평균용) |
| `labels` | dict | `noul` 전용. 모델에 보이는 텍스트만 교체 |

### 7.1 choice — 선택지 중 하나

```python
{
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": {
        "billing":   "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales":     "pricing, new contracts",
        "other":     "everything else"
    }
}
```

응답:
```json
{
  "choice": "billing",
  "confidence": 0.86,
  "answer_confidence": 0.94,
  "probabilities": { "billing": 0.94, "technical": 0.03, "sales": 0.02, "other": 0.01 }
}
```

> `criteria` 키가 그대로 라벨로 사용됩니다. 라벨은 의미적이거나 불투명한 코드(A/B)가 안전합니다.

### 7.2 score — 순서 있는 등급

```python
{
    "type": "score",
    "instructions": "How urgent is this request?",
    "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
}
```

- **모든 등급에 설명(description)이 필요합니다.** `null` 등급은 422로 거부됩니다.
- 응답: 기대 등급(분포) + `probabilities` + `confidence` + `answer_confidence`
- ⚠️ `laya-multilingual` 은 score에서 위치 편향이 있음 (첫 등급을 잘 안 고름) → 영어 score는 `english` 체크포인트로 라우팅

### 7.3 noul — 참/거짓 확률

```python
{
    "type": "noul",
    "instructions": "Does the user threaten to cancel or leave?"
}
```

- 항상 `[false, true]` 두 슬롯을 평가 → 두 번째 슬롯 확률 = `P(true)`
- `criteria` 를 주면 `true`/`false` 키가 필수
- `labels` 로 모델에 보이는 텍스트만 교체 (의미는 그대로)
  ```python
  {"type": "noul", "instructions": "...", "labels": {"false": "A", "true": "B"}}
  ```
- 영어 체크포인트는 기본 `false:`/`true:` 페어에 강하게 끌릴 수 있으므로 판별이 필요하면 criteria 를 명시

### 응답 dict 전체 필드

| 필드 | 설명 |
|---|---|
| `choice` | 선택된 라벨 (choice) |
| `score` | 기대 등급 (score) |
| `noul` | `P(true)` 0.0~1.0 (noul) |
| `confidence` | 1 − 정규화 엔트로피 (**선택지 수에 따라 변함** — 게이팅에 비권장) |
| `answer_confidence` | max(p) — 보정된 값, 선택지 수에 불변 (**게이팅에 권장**) |
| `probabilities` | 라벨별 확률 dict (원래 옵션 순서 기준) |
| `low_confidence` | 게이트 발동 시에만 존재 |
| `abstention` | `passed` / `abstained` / `unevaluated` (min_confidence 설정 시에만) |
| `abstention_threshold` | 적용된 임계값 |
| `window` | `predict_long` 에서 `{index, token_start, token_end, count}` |

### 라우팅 메타데이터

```python
result["routing"]
# {
#   'model': 'multilingual',
#   'repo': 'convaiinnovations/laya/multilingual',
#   'reason': 'non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it'
# }
```

---

## 8. 자기호스팅 서버 (laya-serve)

### 설치 및 실행

```bash
pip install "laya[serve]"

# 전부 상주 + 0.0.0.0:8000 바인딩
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve

# 특정 체크포인트만
LAYA_HOST=127.0.0.1 LAYA_MODELS=english laya-serve
```

### 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/v1/systemone` | 단일 요청 (Jev 호환) |
| POST | `/v1/systemone/batch` | 다중 상태 (최대 64개) |
| GET | `/health` | 헬스체크 (`LAYA_API_KEY` 설정 시 내부 정보는 인증 필요, liveness는 개방) |
| GET | `/models` | 체크포인트 / 별칭 목록 |

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"body": "billed twice, refund please or we cancel"},
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}
}'
```

```bash
curl -s localhost:8000/v1/systemone/batch -H 'content-type: application/json' -d '{
  "states": [
    {"body": "billed twice, refund please"},
    {"body": "cannot login, getting 500 error"}
  ],
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}
}'
```

배치 응답은 원래 순서 배열 + 집계된 `total_usage` 를 반환합니다.

### 환경변수 전체 목록

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LAYA_HOST` | — | 바인드 호스트 |
| `LAYA_PORT` | — | 바인드 포트 |
| `LAYA_DEVICE` | auto | 디바이스 (`cuda`, `cuda:0`, `mps`, `cpu`, `xpu`) |
| `LAYA_PRELOAD` | 1 | 시작 시 체크포인트 상주 (0이면 지연 로드) |
| `LAYA_MODELS` | `english,multilingual` | 상주할 체크포인트 콤마 리스트 |
| `LAYA_THREADS` | torch 기본값 | CPU intra-op 스레드 상한 (물리 코어 이하로) |
| `LAYA_AUTO_TASK` | 0 | 1이면 질문 id가 typed-decisions 워크플로와 일치할 때 그 체크포인트로 라우팅 |
| `LAYA_DEFAULT_MODEL` | `english` | 언어 신호가 없을 때 폴백. 대부분 영어가 아니면 `multilingual` |
| `LAYA_MAX_LOADED` | 2 | 상주 체크포인트 수. `LAYA_AUTO_TASK` 쓰면 3으로 올릴 것 |
| `LAYA_IDLE_UNLOAD_SECONDS` | 0 | 유휴 시 언로드. 0=비활성, 300=5분 후 해제 |
| `LAYA_API_KEY` | (미설정) | 설정 시 `Authorization: Bearer <key>` 필수 |
| `LAYA_ROOT_PATH` | (미설정) | 리버스 프록시 접두사 (예: `/laya`) — 프록시가 접두사를 제거해 전달 |
| `LAYA_JEV_STRICT` | off | 엄격한 Jev 와이어 계약으로 응답 투영 (알 수 없는 필드를 거부하는 클라이언트용) |
| `LAYA_MAX_TOKEN_BUDGET` | (체크포인트별) | per-call `max_len`/`head_max_len` 상한 |
| `LAYA_MAX_BATCH_TOKENS` | (서버값) | 배치 토큰 상한. 초과 시 여러 forward pass로 분할 |
| `LAYA_BASE_URL` | (미설정) | 원격 laya-serve 주소 (MCP/클라이언트 HTTP 모드) |
| `LAYA_REMOTE_TIMEOUT` | 300s | 원격 서버 HTTP 대기 |
| `LAYA_CUDA_AMP` | — | `fp16` 또는 `bf16` |
| `LAYA_CPU_AMP` | — | `bf16` |
| `LAYA_MPS_AMP_MIN_ROWS` | 5 | MPS autocast 행 수 기준 (≥1로 clamp) |

### 운영 메모리 모드

| 모드 | 요청당 지연 | 모델 재로드 |
|---|---|---|
| `Router()` (지연, max_loaded=2) | 전환 시 감색만 (<1ms) | 언어 첫 등장 시 1회 |
| `Router(max_loaded=1)` | 언어 전환마다 7~10초 | 전환마다 1회 |
| `Router(preload=True)` | 32.8ms (GPU) / 193~464ms (CPU) | 없음 |

> **서버 운영 시 `preload=True` 가 정답입니다.** 재로드를 줄이는 게 아니라 없앱니다.
> `max_loaded=1` 로 두면 CPU에서 전환마다 7.4초, T4에서 10.3초 Median 재로드가 발생합니다.

기존 agent 를 VRAM 중복 없이 연결:
```python
router.attach("english", existing_agent)
router.unload()          # 메모리 해제
```

---

## 9. 기존 Jev 클라이언트 마이그레이션

Laya의 응답 페이로드는 Jev와 **스키마 동일** (choice/score/noul + `{input_tokens, output_tokens}` usage)입니다.
기존 Jev 클라이언트는 **baseUrl 만 바꾸면 됩니다.**

```python
# TypeSafe 공식 SDK
client = TypeSafeClient(base_url="http://localhost:8000")   # → laya-serve

# 또는 환경변수
TYPESAFE_BASE_URL=http://localhost:8000

# Ollaya 데몬을 쓸 경우
TYPESAFE_BASE_URL=http://localhost:11435
```

언어별 클라이언트:
- **Haskell**: `hs-jev` — baseUrl만 변경
- **PHP**: `marcreichel/laya-php` (laya-serve 대상)
- **JS/TS**: `laya-client` (npm)
- **.NET**: `laya-dotnet` (Python 사이드카 불필요, 단 NuGet 미출시)

### 포팅 시 반드시 알아야 할 3가지 차이

1. **선택지 개수**
   - 선택지는 체크포인트의 `head_max_len` 예산을 공유 (laya 192, 나머지 256 토큰) — Jev의 255 상한이 아님
   - HTTP 서버는 `MAX_CHOICE_OPTIONS = 100` 을 강제 (인퍼런스 전에 413)
   - 토큰 예산 초과 시 모든 선택지가 잘려 들어감 → **약 20개(짧은 설명 기준)부터 정확도 하락 시작**
   - 전혀 들어가지 않으면 422

2. **score 등급**
   - 모든 등급에 description 필요, `null` 등급은 422

3. **confidence 공식**
   - `confidence` 는 1 − 정규화 엔트로피 (Jev와 다른 공식) → **Jev의 임계값을 그대로 이식 불가**
   - `answer_confidence` (= max p) 사용을 권장. 이 값도 선택지 수에 따라 변하므로 선택지 수별로 임계값을 다르게 설정 (`fit_abstention_thresholds`)

### 추가 확인 필요 항목

- `score` 유형에서 `laya-multilingual` 위치 편향 (이슈 #131)
- `action.act_probability` 는 사용 불가 (이슈 #185) → confidence 로 게이팅 (AUROC 0.77)
- autocast dtype(bf16/fp16/fp32)에 따라 임계값이 달라짐 → **배포 dtype 으로 fit/측정할 것**

---

## 10. 파인튜닝 — 핵심

> 영상의 핵심 내용입니다. **제로샷 Laya 는 쓸 수 없습니다.** 도메인 파인튜닝이 정확도를 끌어올립니다.

### 10.1 영상 실험 결과 요약

한국어 쇼핑몰 고객 문의 6-way 분류 (배송 / 취소 / 교환·반품 / 상품문의 / 주문조회 / 결제) 

- MacBook Pro M4, 24GB RAM, GPU 없이 진행
- 합성 데이터 (온라인 쇼핑몰에서 있을 법한 한국어 문의 + 정답)
- **250개 시험 문제를 미리 따로 떼어 두고 모든 단계에서 동일 세트로 비교**

| 학습 데이터 | 학습 시간 | 정답률 | 맞힌 문제 |
|---|---|---|---|
| 학습 전 | – | 32.8% | 82 / 250 |
| 100건 | 4분 | 68.8% | 172 / 250 |
| 300건 | 15분 | 83.2% | – |
| 1,000건 | 48분 | 86.4% | 216 / 250 |
| **Jev (학습 없음)** | – | **100%** | 250 / 250 |

극단적으로 10,000건까지 확대 (검증 600개 세트로 재평가):

| | 정확도 |
|---|---|
| 학습 전 Laya | 35.8% |
| 10,000건 학습 후 | **99.7%** (598 / 600) |

- 학습 시간: 7시간 52분 (동일한 맥북)
- 지연: Laya 로컬 0.11초 vs Jev API 게이트웨이 경유 중앙값 0.5초
- 발표자 정리: *"Jev 는 범용 모델이라 학습 없이도 다른 분야에서 잘할 것이기 때문에 Laya 가 Jev 를 완전히 따라잡았다고 말할 수는 없다. 증명된 것은 '쓰고 싶은 특정 영역에 대해 학습시키면, 학습 건수와 데이터 정확도에 따라 매우 높은 정답률을 만들 수 있다'는 점."*

### 10.2 학습 방법 선택

| 방법 | 환경 | 비고 |
|---|---|---|
| `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | Kaggle 2×T4 (DDP) | 전체 루프 (데이터셋 구축 → RLCD 학습 → 온도 교정 → 평가 → Hub 푸시) |
| `notebooks/laya_finetune_typed_decisions_mps.py` | Apple Silicon (MPS + CPU 폴백) | 단일 프로세스, `--micro-batch` / `--grad-accum` 으로 DDP 대체 |

```bash
python notebooks/laya_finetune_typed_decisions_mps.py \
  --micro-batch 1 \
  --grad-accum 32
```

### 10.3 학습 루프 구성 (5단계)

```
1. 라우팅 경로 정의      → 내 서비스의 분류 항목 + 판단 기준을 문서로
2. 시험 세트 분리        → 학습에 절대 쓰지 않을 held-out 세트 확보
3. 소량부터 학습         → 100건으로 시작, 300 → 1,000 순으로 증가하며 기록
4. 오류 분석 & 보강      → 틀린 문제 분류(혼동 패턴) → 대비 예시 추가 재학습
5. 교정(calibration)    → 온도 스케일링 / 히스토그램 binning / abstention threshold fit
```

### 10.4 학습 관련 기술 세부

**RLCD (Reinforcement Learning for Calibrated Decisions)**
- properly scoring rule 보상 + GRPO 스타일 정책 그래디언트
- **실측: 2×T4에서 4 epoch / 약 30,000 question ≈ 4~5시간**
- 제로샷 베이스: typed-decisions 0.36 / 0.35 (random 0.318) → 파인튜닝 후 **0.766**

**Gradient checkpointing**
- 인코더와 decision head 양쪽에 활성화
- 커스텀 루프에서는 `model.head_checkpointing = True` (기본 `False`), 인코더는 별도 활성화
- eval / `torch.no_grad()` 에서는 우회
- 계산량 ↔ 활성화 메모리 트레이드오프

**교정(calibration) 파라미터**
```python
save_calibration(...)                    # 원자적 쓰기
fit_abstention_thresholds(...)           # 선택지 수 버킷별 컷 1개 (temperature_by_options 와 동일 키 구조)
fit_binning_map(...) / apply_binning_map(...)   # 온도 스케일링으로 못잡는 버킷에 히스토그램 binning
records_from_labeled(agent)               # 학습된 agent 에서 레코드 추출
```
- 타입별 온도 1개씩 (`choice`, `score`, `noul`) + `temperature_by_options` 버킷별 온도
- 로드 시 [0.5, 5.0] 으로 clamp, 비정상 값은 1.0 + 경고
- 버킷별 온도가 타입별 온도보다 우선
- **`rl_agent_config.json` 은 체크포인트와 함께 배포되는 파일입니다.** 소스 저장소에서 직접 만들지 마세요. 로컬 모델은 `model.safetensors` 가 있는 디렉터리를 지정합니다.
- 교정 샘플은 학습 데이터에서, 평가는 별도 held-out 데이터에서
- 교정 persisted 확인: `python tests/test_calibration_persistence.py` (CPU 전용, 다운로드/학습 없음)

**autocast 와 임계값**
```python
agent.dtype_for(rows)   # 해당 행 수에 대한 precision 반환
agent.dtype             # MPS autocast 대상
```
autocast dtype(bf16 vs fp16 vs fp32)에 따라 신뢰도 임계값이 이동합니다. 반드시 **배포할 dtype 으로 fit 하고 측정하세요.**

### 10.5 실전 파인튜닝 팁 (영상에서 나온 조언)

1. **라우팅은 본답 생성 전에 반드시 거치는 단계** → 여기서 지연이 생기면 뒤단 LLM 도 그만큼 늦게 시작
2. **적은 데이터로 큰 효과** — 100건/4분으로 32.8% → 68.8%
3. **학습량을 늘려도 개선 폭은 줄어듦** — 300→1,000건에서 83.2% → 86.4% (그 원리를 우선: **양보다 데이터 품질/보강**)
4. **범용 성능은 여전히 Jev 우위** — Jev 는 제로샷 100%, Laya 는 특정 영역 최적화 결과
5. **로컬 실행 = 속도 + 보안** — 외부 전송 불가 데이터(고객 정보, 사내 자료)를 다룬다면 결정적 우위

---

## 11. 성능 및 벤치마크

### 처리량 (Tesla T4)

| 질문 수 / 호출 | `laya` (ms) | `laya-multilingual` (ms) |
|---|---|---|
| 1 | 39.5 | **32.8** |
| 5 | 84.5 | 40.1 |
| 10 | 158.6 (15.9/질문) | **72.3 (7.2/질문)** |
| 50 | 771 | 337 (6.8/질문) |

배치 처리량 약 **103 ~ 332 question/초**.

### 영어 성능

| task | `laya` | `laya-multilingual` | 비고 |
|---|---|---|---|
| AG News | 0.947 | 0.937 | 학습 혼합 |
| BoolQ | 0.830 | 0.787 | 학습 혼합 |
| DAIR Emotion | 0.573 | 0.513 | held out |
| prompt-injections | 0.698 | 0.578 | held out (n=116) |
| SST-5 (순서형) | 0.372 | 0.282 | held out — **가장 약한 유형** |

### 교정 효과 (Calibration)

| 모델 | 평균 ECE (교정 전) | 평균 ECE (교정 후) |
|---|---|---|
| `laya` | 0.466 | **0.081** |
| `laya-multilingual` | 0.314 | **0.106** |

> 다국어 체크포인트는 사전 fitted 온도 없이 배포됩니다.

### 다국어 커버리지 (51개 언어)

| 모델 | 매크로 정확도 | 사용 가능한 언어 (random 대비 ≥3×) |
|---|---|---|
| `laya` | 0.227 | 23 / 51 |
| `laya-multilingual` | 0.401 | 48 / 51 |
| **Router** | – | **48 / 51** |

---

## 12. 한계점 — 반드시 읽을 것

### 선택지 개수
- **약 20개 넘으면 정확도가 급락** — Bank77 (72 vs 77 class) 에서 Jev 0.870 vs Laya 0.425
- HTTP 서버는 100개에서 413
- 해결책: `head_max_len` / `max_len` 상향, 또는 `predict_shortlist` 로 후보 축소

### 컨텍스트 길이
- `laya` 512, 나머지 1024 (다국어는 `max_len=8192` 명시 시 최대 8,192)
- 실측: 4,000 토큰까지 20개 요청 중 16~18개 정확, 그 이상은 8~17개로 편차
- Jev 는 64k 토큰 → **긴 티켓/문서에서는 Jev 우위**

### 유형별 강약
- `noul` 0.857 (가장 강함) / `choice` 0.733 / `score` 0.723 (가장 약함)
- SST-5 (5단계 순서 분류) 0.372 — 순서형 등급은 명백히 어려움
- `laya-multilingual` 의 score 위치 편향 (첫 등급 선호) → 영어 score 는 `english` 로

### 신뢰도 관련
- `confidence` 공식이 Jev 와 다름 → Jev 임계값 이식 불가
- `answer_confidence` 도 선택지 수에 따라 변함 → 선택지 수별 임계값 필요 (`fit_abstention_thresholds`)
- 영어 체크포인트는 기본 `false:`/`true:` 페어에 강하게 끌림 → noul 판별이 필요하면 `criteria`/`labels` 명시

### 기타
- 의미적 부정(semantic negation)이 좁은 케이스에서 실패 가능 → 반드시 자체 데이터로 검증
- `action.act_probability` 사용 불가 (이슈 #185)
- 체크포인트 cold build 에 수 초 → 운영은 `preload=True`
- `max_loaded=1` 은 언어 전환마다 재로드 (CPU 7.4s, T4 10.3s)
- 배치 모양 변경(`sort_by_length`)은 임계값 근처에서 미세한 float 차이를 만들 수 있음

### **Laya 를 선택해야 하는 경우**
- ✅ 선택지 ≤ 20개
- ✅ 데이터가 네트워크 밖으로 나가면 안 됨 (데이터 레지던시)
- ✅ 자사 도메인 라벨 데이터로 파인튜닝할 준비가 되어 있음
- ✅ 조기 분류(triage)에 제3자 분류기(zsro, winnow 등)를 로컬로 대체하고 싶음
- ✅ GPU 운영 역량 없이 소비자 GPU / 노트북에서 돌려야 함
- ✅ 신뢰도 보정(ECE)이 중요

### **Jev 가 나은 경우**
- 라벨 데이터 없음, 파인튜닝 인력도 없음 → 제로샷으로 쓸 만한 범용 성능
- 선택지가 255개까지 필요
- 긴 문서(64k 토큰) 분류
- 관리형 서비스(인프라·추론 모니터링)를 원할 때
- $/100만 토큰 0.042 가 예산 대비 합리적일 때

---

## 13. 활용 예제

### 예제 1. 쇼핑몰 고객 지원 라우팅 (영상의 주제)

```python
from laya import Router

router = Router(preload=True, default="multilingual")   # 한국어 트래픽이 대부분

questions = {
    "category": {
        "type": "choice",
        "instructions": "이 문의의 처리 부서를 분류하세요.",
        "criteria": {
            "shipping":  "배송 조회, 도착 예정일, 배송 지연",
            "cancel":    "주문 취소, 발송 전 취소 요청",
            "exchange":  "교환 및 반품,.product 교환 가능 여부",
            "product":   "상품 정보, 사이즈, 재고 문의",
            "payment":   "결제 오류, 중복 청구, 영수증",
            "human":     "분쟁, 법적 조치, 컴플레인 — 상담원 직접 대응"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "응답 urgency 정도를 판단하세요.",
        "criteria": ["low", "normal", "high", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "고객이 구독을 취소하거나 이탈을 언급했는가?",
    },
    "needs_human": {
        "type": "noul",
        "instructions": "상담원의 직접 개입이 반드시 필요한가?",
    },
}

def handle_ticket(ticket):
    r = router.predict(ticket, questions)
    a = r["answers"]

    # 게이트: 확신이 없으면 사람이 직접 검토
    if a["category"].get("low_confidence") or a["category"]["answer_confidence"] < 0.7:
        return {"route": "human_review", "reason": "low confidence"}

    cat = a["category"]["choice"]

    # 라우팅: 간단한 조회는 코드/DB 로직으로
    if cat == "shipping":
        return lookup_order(ticket["order_id"])                 # LLM 호출 0회
    if cat == "payment":
        return check_payment_ledger(ticket)                     # LLM 호출 0회
    if cat == "cancel":
        return check_returnable(ticket)                         # 정책 판단 로직
    if cat == "human" or a["needs_human"]["noul"] > 0.8:
        return escalate_to_agent(ticket, reason=cat)

    # 나머지만 큰 LLM 으로
    return llm_answer(ticket, urgency=a["urgency"]["score"],
                      churn_risk=a["churn_risk"]["noul"])
```

### 예제 2. 프롬프트 인젝션 / 가드레일

```python
# 내장 프리셋 사용
guard = laya.guard_questions()      # 또는 moderation_questions()

r = router.predict({"body": user_input}, guard)
if r["answers"]["injection"]["noul"] > 0.5:
    return {"blocked": True}

# LangChain 에서
from laya.integrations.langchain import LayaGuardrail
guardrail = LayaGuardrail(action="raise")   # 위반 시 LayaGuardrailError
```

### 예제 3. 사내 보안 데이터 분류 (데이터 레지던시)

```python
# 고객 정보 / 내부 문서를 외부로 보내면 안 되는 환경
#  → 라우팅/분류 단계만 내부 네트워크의 laya-serve 에서 수행

questions = {
    "pii": {
        "type": "choice",
        "instructions": "문서에 포함된 개인정보 유형을 분류하세요.",
        "criteria": {
            "none":     "개인정보 없음",
            "contact":  "이름, 전화번호, 이메일",
            "financial": "계좌번호, 카드번호",
            "id":       "주민등록번호, 여권번호",
            "medical":  "진단명, 처방전, 병력",
        }
    },
    "external_share_ok": {
        "type": "noul",
        "instructions": "이 문서를 외부 AI 서비스에 전송해도 안전한가?",
    },
    "retention_days": {
        "type": "choice",
        "instructions": "법적 보관 기간 분류.",
        "criteria": {"30": "단기", "180": "중기", "2555": "장기 보관(5년)", "permanent": "영구"},
    },
}
```

### 예제 4. 버전 관리 (의사코드 자유 출력의 대체)

LLM 에게 "다음 중 하나로만 답하라"고 지시하면 형식을 안 지킵니다. Laya 는 **샘플을 만들지 않고** 강제합니다.

```python
schema = {
    "type": "object",
    "properties": {
        "intent":  {"type": "string", "enum": ["query", "refund", "cancel", "chitchat"]},
        "retry":   {"type": "integer", "minimum": 0, "maximum": 5},
        "human":   {"type": "boolean"},
    },
    "required": ["intent", "retry", "human"],
}

out = router.decide_batch(listeners, schema=schema, return_details=True, batch_size=64)
```

### 예제 5. 브라우저 에이전트 판단

문서: `docs/finetune_browser_agent.md`, 가중치: `cklxx/laya-browser`

브라우저 에이전트가 매 스텝마다 LLM을 호출하면 느리고 비쌉니다.
페이지 상태를 Laya 로 한 번 분류해서 "클릭 / 입력 / 추출 / 중단" 중 무엇을 할지 O(수 ms)에 결정 후, 필요한 순간에만 LLM 호출.

```python
questions = {
    "action": {
        "type": "choice",
        "instructions": "현재 페이지 상태에서 다음으로 해야 할 행동.",
        "criteria": {
            "click_login": "로그인 버튼이 보인다",
            "fill_form":   "입력 폼이 비어 있다",
            "extract":     "완료된 페이지에서 데이터를 추출해야 한다",
            "ask_human":   "봇 우회에 걸렸거나 CAPTCHA가 있다",
            "done":        "목표 달성이 확인된다",
        }
    },
    "blocked": {"type": "noul", "instructions": "봇 탐지 또는 CAPTCHA에 막혔는가?"},
}
```

### 예제 6. 성능 최적화 — 배치를 제대로 쓰기

```python
import time

# ❌ 나쁜 패턴: 요청마다 checkpoint 로드 반복
for t in tickets:
    result = router.predict(t, questions)

# ✅ 좋은 패턴: 길이 정렬 배치 (Yelp 128건, MPS, batch 8 기준 5.35s → 3.04s)
results = router.predict_batch(
    [{"state": t, "questions": questions} for t in tickets],
    batch_size=8,
    sort_by_length=True,
)
```

### 예제 7. 3단 분류 라우팅 (가장 일반적인 패턴)

```python
questions = {
    "complexity": {
        "type": "choice",
        "instructions": "이 요청을 처리하는 데 어느 수준의 자원이 필요한가?",
        "criteria": {
            "trivial": "단순 조회. 결정적 로직으로 100% 처리 가능",
            "moderate": "판단이 일부 필요. 저비용 모델로 처리",
            "complex": "창의적 사고/다단계 추론 필요. 최고 모델 필요",
        }
    }
}

r = router.predict(state, questions)
c = r["answers"]["complexity"]
if c.get("low_confidence"):
    c = {"choice": "complex"}    # 불확실하면 안전하게 최고 모델로

if c["choice"] == "trivial":
    return rule_engine(state)
elif c["choice"] == "moderate":
    return cheap_model(state)
else:
    return frontier_model(state)
```

---

## 14. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'laya'` | 설치와 실행이 서로 다른 venv 의 Python 을 쓰고 있음. 설치에 쓴 exe 로 실행 |
| 에디터에서 import 안 됨 | 에디터 인터프리터를 해당 venv 로 지정 |
| `Missing rl_agent_config.json` | 이 파일은 체크포인트와 함께 배포됨. 직접 만들 필요 없음. 로컬 모델은 `model.safetensors` 가 있는 디렉터리 지정 |
| 체크포인트 다운로드 실패 | HuggingFace 접근 확인. `HF_HUB_CACHE` 로 캐시 위치 변경 |
| `C.UTF-8` 인 모든 요청이 multilingual 로 감 | 해결됨 — `und`/`zxx`/`mul`/`C`/`POSIX`/`C.UTF-8` 는 모두 abstain 하고 내장 감지로 폴백 |
| venv 에서 `ensurepip is not available` (Debian/Ubuntu) | `sudo apt install python3-venv` |
| Windows + Python 3.14 모델 생성 크래시 | Laya 0.3.7 에서 수정됨 (#195). `-m pip install -U laya` 로 업그레이드 |
| GPU OOM | `Router(max_loaded=1)` 또는 `LAYA_IDLE_UNLOAD_SECONDS=300`. OOM 폴백이 진행 중 요청 중 모델을 이동시키지 않음(#649) |
| 첫 요청이 느림 | `Router(preload=True)` 사용 |
|置信도 임계값이 맞지 않음 | 배포 dtype 으로 재 fit. autocast 설정 확인 |
| 영어인데 라틴 문자라 multilingual 로 감 | `LAYA_DEFAULT_MODEL` 확인, 또는 `lang_guess` LID 연동 |
| 413 응답 | 선택지 100개 초과 |
| 422 응답 | 등급에 description 없음 / 토큰 예산 초과 / HTTP 로 callable 훅 전송 |

### GPU OOM / 장치 강등
CUDA ordinal 이 `device_count` 를 넘거나 `auto` 가 다양한 표기로 주어지면, 나중死在 아니라 로드 시점에 warn 후 CPU 로 폴백합니다.

### `nix` / NixOS
Flake: `nix run .#laya-serve`, `nix develop`
NixOS 모듈 `services.laya-serve`: `enable`, `host`, `openFirewall`, `device`, `models`, `apiKeyFile` (LoadCredential), 하드닝된 DynamicUser, 가중치 캐시 `/var/lib/laya-serve`

---

## 15. 참고 링크

| 리소스 | URL |
|---|---|
| GitHub | https://github.com/NandhaKishorM/laya |
| 문서 사이트 | https://nandhakishorm.github.io/laya/ |
| HuggingFace | https://huggingface.co/convaiinnovations/laya |
| HF Demo (Space) | https://huggingface.co/spaces/convaiinnovations/laya-demo |
| Laya-MLX | https://github.com/mizorewww/laya-mlx |
| Ollaya | https://github.com/ollaya-dev/ollaya |
| PHP 클라이언트 | https://github.com/marcreichel/laya-php |
| 브라우저 에이전트 가중치 | `cklxx/laya-browser` |
| Fine-tuning | `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, `notebooks/laya_finetune_typed_decisions_mps.py` |
| Docker | `docs/docker.md`, `compose.yaml`, `compose.cuda.yaml`, `compose.http.yaml` |
| Hooks | `docs/hooks/lifecycle.md`, `examples/hooks/` |
| 벤치마크 | `BENCHMARKS.md`, `research/scripts/laya_benchmark_colab.ipynb` |
| 웹 플레이그라운드 | `python examples/server.py` |
| 교정 persistence 테스트 | `python tests/test_calibration_persistence.py` |

### 커뮤니티 도구 (README 목록)
`omp-laya-judge`, `laya-adk-toolkit`, `laya-Ascend`, `laya-apple`, `stuntd`

---

## 부록: 관련 비교 모델

| 모델 | 설명 |
|---|---|
| `kev` | Qwen + LoRA로 같은 문제를 재현. 디코더 모델의 프리필 1회로 상태 공유 (Laya 와 반대 접근) |
| `decider` | 다른 결정 모델 |
| `Decision 1.0` | vLLM Semantic Router 팀 |
| `Qwen3Guard` | 가드레일 특화 |
| `winnow` | 결정 모델 |
| GLiClass / NLI 분류기 | 제로샷 자연어 추론 분류기 |
| `TypeLLM` | 기존 자기회귀 모델에 JSON Schema 제약 출력 추가. Jev 에서 영감 |

`Ollaya` 로 위 모델들을 이름으로 내려받아 서빙할 수 있습니다 (Ollama for decision models).

---

*문서 작성일: 2026-10-05 · Laya Apache-2.0 · Jev 는 TypeSafe AI 의 호스팅 제품*
