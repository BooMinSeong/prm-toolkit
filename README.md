# vLLM Process Reward Model (PRM) Inference

vLLM을 사용한 수학 추론 단계 평가를 위한 Process Reward Model (PRM) 추론 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 단계별 추론 평가를 위한 통합 PRM 서버 인터페이스를 제공합니다. vLLM 서버 HTTP 엔드포인트를 통해 모든 모델에 일관된 API로 접근할 수 있습니다.

**주요 특징:**
- 통합된 서버 기반 아키텍처
- 타입 안전 설정 (PrmConfig dataclass)
- 모델별 자동 전처리/후처리
- 단일 `score(prompt, response)` API

## 지원 모델

- **Qwen2.5-Math-PRM-7B**: Qwen의 수학 추론용 PRM
  - 단계 구분자: `\n\n` (이중 줄바꿈)
  - 출력: `[negative_prob, positive_prob]` 쌍에서 positive 확률 추출

- **Skywork-o1-Open-PRM-Qwen-2.5-1.5B**: Skywork의 오픈소스 PRM
  - 단계 구분자: `\n` (단일 줄바꿈)
  - 출력: Sigmoid 정규화된 보상값 [0, 1]
  - vLLM 0.14.1 호환 커스텀 모델 구현 포함

## 빠른 시작

### 1. 환경 설정

```bash
# 가상 환경 활성화
source .venv/bin/activate

# vLLM 플러그인 설치 (Skywork PRM에 필수)
pip install -e .

# 플러그인 로드 확인 (다음 메시지가 표시되어야 함)
# ✓ Registered SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) for Skywork-o1-Open-PRM
```

### 2. PRM 서버 시작

**Qwen PRM:**
```bash
vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code
```

**Skywork PRM:**
```bash
python start_reward_server.py
# 또는: vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B --port 8081 --trust-remote-code
```

### 3. 통합 API 사용

```python
from prm_server import PrmConfig, create_prm_server

# 설정
config = PrmConfig(
    model="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8080"
)

# PRM 인스턴스 생성
prm = create_prm_server(config)

# 응답 평가
prompt = "15 + 27은 얼마입니까?"
response = "첫째, 일의 자리를 더합니다: 5 + 7 = 12\n\n둘째, 십의 자리를 더합니다: 10 + 20 = 30\n\n마지막으로: 30 + 12 = 42"
rewards = prm.score(prompt, response)

# 결과: [0.92, 0.95, 0.98] (각 단계별 보상값)
print(f"단계별 보상: {rewards}")
print(f"평균 보상: {sum(rewards) / len(rewards):.4f}")
```

**예제 실행:**
```bash
# Qwen PRM (서버가 8080 포트에서 실행 중이어야 함)
python example_prm_usage.py --model qwen

# Skywork PRM (서버가 8081 포트에서 실행 중이어야 함)
python example_prm_usage.py --model skywork
```

## Unified PRM Server 아키텍처

### 핵심 구성요소

```
prm_server.py
├── PrmConfig              # 타입 안전 설정 dataclass
├── PrmServer              # 추상 베이스 클래스
│   ├── preprocess_input() # 모델별 포맷팅
│   ├── send_request()     # HTTP 통신
│   ├── post_process_output() # 보상값 추출
│   └── score()            # 메인 진입점
├── QwenPrmServer          # Qwen2.5-Math-PRM 구현
├── SkyworkPrmServer       # Skywork-o1-Open-PRM 구현
└── create_prm_server()    # 팩토리 함수
```

### 설정 (PrmConfig)

```python
from dataclasses import dataclass

@dataclass
class PrmConfig:
    model: str                    # 모델 이름 (예: "Qwen/Qwen2.5-Math-PRM-7B")
    base_url: str                 # 서버 URL (예: "http://localhost:8080")
    timeout: int = 300            # 요청 타임아웃 (초)
    trust_remote_code: bool = True
```

### 통합 인터페이스

모든 PRM 서버는 동일한 인터페이스를 구현합니다:

```python
class PrmServer(ABC):
    def score(self, prompt: str, response: str) -> List[float]:
        """
        단계별 응답을 평가합니다.

        Args:
            prompt: 문제 또는 질문
            response: 단계별 솔루션 (모델별 구분자로 단계 구분)

        Returns:
            정규화된 보상값 리스트 (각 단계당 하나)
        """
```

## 사용 패턴

### 기본 사용법

```python
from prm_server import PrmConfig, create_prm_server

config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")
prm = create_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

### LLM 생성과 함께 사용

```python
from prm_server import PrmConfig, create_prm_server
from vllm import LLM, SamplingParams

# 응답 생성
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(
    prompts=["15 + 27을 계산하세요"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=200)
)
response = outputs[0].outputs[0].text

# PRM으로 평가
prm_config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")
prm = create_prm_server(prm_config)
rewards = prm.score(prompt="15 + 27을 계산하세요", response=response)
```

### 배치 처리

```python
problems = [
    ("문제 1", "해결 과정 1\n\n단계별"),
    ("문제 2", "해결 과정 2\n\n단계별"),
]

results = []
for prompt, response in problems:
    rewards = prm.score(prompt, response)
    avg_reward = sum(rewards) / len(rewards)
    results.append((prompt, avg_reward))

# 최고 솔루션 찾기
best_prompt, best_score = max(results, key=lambda x: x[1])
print(f"최고 점수 ({best_score:.4f}): {best_prompt}")
```

### Best-of-N 랭킹

Process Reward Model은 여러 후보 솔루션 중 최선을 선택하는 Best-of-N 전략에 사용할 수 있습니다:

```python
# 동일 문제에 대해 N개 솔루션 생성
solutions = [
    "솔루션 1 with\n\nsteps",
    "솔루션 2 with\n\nsteps",
    "솔루션 3 with\n\nsteps",
]

# 각 솔루션 평가
avg_rewards = []
for sol in solutions:
    rewards = prm.score(prompt="문제", response=sol)
    avg_rewards.append(sum(rewards) / len(rewards))

# 최고 평균 보상 솔루션 선택
best_idx = avg_rewards.index(max(avg_rewards))
print(f"최고 솔루션: {best_idx + 1} (평균 보상: {avg_rewards[best_idx]:.4f})")
```

## 모델별 상세 정보

### Qwen2.5-Math-PRM-7B

**단계 구분자:** `\n\n` (이중 줄바꿈)

**입력 형식:**
```python
prompt = "2 + 2 = ?"
response = "첫 번째 단계: 2와 2를 더합니다\n\n두 번째 단계: 결과는 4입니다"
```

**내부 처리:**
1. 채팅 템플릿으로 감싸기: `<im_start>system\n...<im_end>\n<im_start>user\n...<im_end>\n<im_start>assistant\n...`
2. 단계를 `<extra_0>` 토큰으로 변환: `첫 번째 단계<extra_0>두 번째 단계<extra_0>`
3. 서버가 `[negative_prob, positive_prob]` 쌍 반환
4. Positive 확률(인덱스 1)을 보상값으로 추출

**출력:** [0, 1] 범위의 확률 (추가 정규화 없음)

**서버 시작 예시:**
```bash
vllm serve Qwen/Qwen2.5-Math-PRM-7B \
    --port 8080 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

### Skywork-o1-Open-PRM-Qwen-2.5-1.5B

**단계 구분자:** `\n` (단일 줄바꿈)

**입력 형식:**
```python
prompt = "2 + 2 = ?"
response = "첫 번째 단계: 2와 2를 더합니다\n두 번째 단계: 결과는 4입니다"
```

**내부 처리:**
1. BOS 토큰 + 문제 + 응답으로 토큰화
2. reward_flags 배열 생성 (단계 끝 위치에 1 표시)
3. 서버가 원시 logits 반환
4. reward_flags로 필터링하고 sigmoid 적용: `1 / (1 + exp(-x))`

**출력:** Sigmoid 정규화된 [0, 1] 범위의 보상값

**서버 시작 예시:**
```bash
# 헬퍼 스크립트 사용
python start_reward_server.py

# 또는 수동 실행
vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --port 8081 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

## Legacy Scripts (참고용)

이전 버전 스크립트는 호환성을 위해 유지됩니다. **새 코드에는 Unified PRM Server를 사용하세요.**

### 기본 Reward Model 테스트

```bash
python reward.py
```
- Qwen2.5-Math-PRM-7B 사용
- 단순 텍스트 프롬프트
- 기본 vLLM pooling/reward 모델 예시

### Qwen PRM - 단계별 평가

```bash
python reward_qwen_prm.py
```
- `<extra_0>` 토큰으로 단계 구분
- `<im_start>`/`<im_end>` 채팅 템플릿
- 단계별 추론 평가

### Skywork-o1-Open-PRM

**옵션 A: 직접 실행**
```bash
python reward_skywork_o1_prm.py
```

**옵션 B: 서버/클라이언트 모드**
```bash
# 터미널 1: 서버 시작
python start_reward_server.py

# 터미널 2: 클라이언트 실행
python reward_skywork_server.py
```

### Legacy에서 Unified로 마이그레이션

**이전 방식 (reward_qwen_prm_server.py):**
```python
# 수동 포맷팅
steps = response.split("\n\n")
formatted = "<extra_0>".join(steps) + "<extra_0>"
prompt = f"<im_start>system\n{system}<im_end>\n<im_start>user\n{query}<im_end>\n<im_start>assistant\n{formatted}<im_end><|endoftext|>"

# 수동 요청
response = requests.post(
    f"{base_url}/pooling",
    json={"input": [prompt]},
    timeout=300
)

# 수동 추출
rewards_raw = response.json()["data"][0]["data"]
rewards = [r[1] for r in rewards_raw]  # positive 확률 추출
```

**새로운 방식 (unified prm_server.py):**
```python
from prm_server import PrmConfig, create_prm_server

config = PrmConfig(model="Qwen/Qwen2.5-Math-PRM-7B", base_url="http://localhost:8080")
prm = create_prm_server(config)
rewards = prm.score(prompt="...", response="...")
```

**장점:**
- 3줄 코드 (기존 ~15줄에서 단축)
- 수동 포맷팅 불필요
- 타입 안전 설정
- 일관된 에러 처리
- 모든 지원 PRM 모델에서 동작

## vLLM 0.14.1 호환성

### Skywork PRM 커스텀 모델 구현

Skywork-o1-Open-PRM은 `Qwen2ForPrmModel` 아키텍처를 사용하며, vLLM의 표준 `Qwen2ForProcessRewardModel`과 다른 `v_head` 파라미터 구조를 가집니다.

**커스텀 모델 (`skywork_prm_model.py`):**
- Skywork의 정확한 아키텍처 구현 (ValueHead with `v_head` 파라미터)
- vLLM 0.14.1 네이티브 API 사용 (Pooler, DispatchPooler)
- STEP pooling 지원 (프로세스 수준 보상)
- vLLM ModelRegistry 자동 등록

### vLLM 플러그인 시스템

`pyproject.toml`의 entry point를 통해 자동 등록:

```toml
[project.entry-points."vllm.general_plugins"]
register_skywork_prm = "skywork_prm_model:register_skywork_prm_model"
```

**설치:**
```bash
pip install -e .
```

플러그인 로드 확인:
```
✓ Registered SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) for Skywork-o1-Open-PRM
```

### vLLM 0.14+ 참고사항

- **비동기 스케줄링**: v0.14.0부터 기본 활성화. 문제 발생 시 `--disable-async-output-proc`로 비활성화
- **PyTorch 요구사항**: v2.5.0 이상 필요
- **플러그인 시스템**: Entry point 등록 방식 변경 없음

## 프로젝트 구조

```
.
├── prm_server.py                 # 통합 PRM 서버 아키텍처 (권장)
├── example_prm_usage.py          # 통합 API 사용 예제
├── start_reward_server.py        # Skywork PRM 서버 시작 헬퍼
│
├── reward.py                     # Legacy: 기본 보상 모델 예제
├── reward_qwen_prm.py            # Legacy: Qwen PRM 단계별 평가
├── reward_skywork_o1_prm.py      # Legacy: Skywork PRM 직접 실행
├── reward_skywork_server.py      # Legacy: Skywork PRM 클라이언트
│
├── skywork_prm_model.py          # Skywork 커스텀 모델 (vLLM 플러그인)
├── pyproject.toml                # 패키지 설정 및 플러그인 entry point
│
├── README.md                     # 본 파일 (통합 가이드)
├── CLAUDE.md                     # Claude Code용 프로젝트 문서
└── IMPLEMENTATION_SUMMARY_V2.md  # 기술 구현 세부사항
```

## 문제 해결

### 서버 미실행

**에러:** `RuntimeError: PRM server request failed: Connection refused`

**해결:**
```bash
# Qwen 서버 시작
vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8080 --trust-remote-code

# Skywork 서버 시작
python start_reward_server.py

# 서버 상태 확인
curl http://localhost:8080/health
```

### 잘못된 포트

**에러:** `RuntimeError: PRM server request failed: Connection refused`

**해결:** `base_url`이 서버 포트와 일치하는지 확인
- Qwen 기본값: `http://localhost:8080`
- Skywork 기본값: `http://localhost:8081`

### 빈 보상값

**에러:** 빈 보상값 리스트 `[]`

**원인:**
1. **Qwen:** 응답에 `\n\n` (이중 줄바꿈) 구분자 없음
2. **Skywork:** 응답에 `\n` (단일 줄바꿈) 구분자 없음

**해결:** 모델에 맞는 단계 구분자 사용
```python
# Qwen: 이중 줄바꿈
response = "단계 1: ...\n\n단계 2: ..."

# Skywork: 단일 줄바꿈
response = "단계 1: ...\n단계 2: ..."
```

### CUDA Out of Memory

```bash
# GPU 메모리 사용률 낮추기
python reward_skywork_o1_prm.py --gpu-memory-utilization 0.7

# 또는 서버 시작 시
vllm serve Qwen/Qwen2.5-Math-PRM-7B \
    --port 8080 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7
```

### 요청 타임아웃

**에러:** `RuntimeError: PRM server request failed: Request timeout`

**해결:** 타임아웃 증가 또는 입력 크기 감소
```python
config = PrmConfig(
    model="...",
    base_url="...",
    timeout=600  # 기본 300초에서 증가
)
```

### 모델 다운로드 실패

```bash
# Hugging Face 토큰 설정 (비공개 모델)
export HF_TOKEN=your_token_here
```

## vLLM 엔진 옵션

모든 스크립트는 vLLM 엔진 인자를 CLI로 전달할 수 있습니다:

```bash
# 예시: 최대 토큰 길이 및 텐서 병렬화 설정
python reward_skywork_o1_prm.py \
    --max-model-len 2048 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95
```

**주요 옵션:**
- `--model`: 모델 경로/이름
- `--max-model-len`: 최대 시퀀스 길이 (기본값: 1024)
- `--tensor-parallel-size`: GPU 병렬화 수 (기본값: 1)
- `--gpu-memory-utilization`: GPU 메모리 사용률 (기본값: 0.9)
- `--trust-remote-code`: 원격 코드 실행 허용 (필수)

## 성능 최적화 팁

1. **PRM 인스턴스 재사용:** 한 번 생성하고 여러 번 `score()` 호출
   ```python
   prm = create_prm_server(config)
   for prompt, response in data:
       rewards = prm.score(prompt, response)
   ```

2. **애플리케이션 레벨 배치:** 여러 (prompt, response) 쌍 순차 처리
   ```python
   results = [prm.score(p, r) for p, r in zip(prompts, responses)]
   ```

3. **서버 스케일링:** 병렬 처리를 위해 다른 포트에 여러 vLLM 서버 실행

4. **토크나이저 캐싱:** Skywork는 초기화 시 한 번만 토크나이저 로드

## API 레퍼런스

### PrmConfig

```python
@dataclass
class PrmConfig:
    model: str                    # 모델 식별자
    base_url: str                 # 서버 HTTP URL
    timeout: int = 300            # 요청 타임아웃 (초)
    trust_remote_code: bool = True
```

### PrmServer

```python
class PrmServer(ABC):
    def score(self, prompt: str, response: str) -> List[float]:
        """단계별 응답 평가"""

    def preprocess_input(self, prompt: str, response: str) -> Dict[str, Any]:
        """모델별 입력 포맷팅"""

    def send_request(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """/pooling 엔드포인트로 HTTP 요청"""

    def post_process_output(self, raw_results: Dict[str, Any]) -> List[float]:
        """보상값 추출 및 정규화"""
```

### 팩토리 함수

```python
def create_prm_server(config: PrmConfig) -> PrmServer:
    """
    모델 이름을 기반으로 적절한 PRM 서버 인스턴스 생성.

    Args:
        config: 모델 식별자와 서버 URL을 포함한 PrmConfig

    Returns:
        QwenPrmServer 또는 SkyworkPrmServer 인스턴스

    Raises:
        ValueError: 모델 타입을 인식할 수 없는 경우
    """
```

## 참고 자료

- [vLLM Pooling Models 문서](https://docs.vllm.ai/en/stable/models/pooling_models/)
- [Skywork-o1-Open-PRM on HuggingFace](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B)
- [Qwen2.5-Math-PRM-7B on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)
- [vLLM 공식 문서](https://docs.vllm.ai/)

## 라이선스

Apache-2.0 License
