# vLLM Process Reward Model (PRM) Inference

vLLM을 사용한 수학 추론 단계 평가를 위한 Process Reward Model (PRM) 추론 테스트 프로젝트입니다.

## 지원 모델

이 프로젝트는 다음 Process Reward Models을 지원합니다:

- **Qwen2.5-Math-PRM-7B**: Qwen의 수학 추론용 PRM
- **Skywork-o1-Open-PRM-Qwen-2.5-1.5B**: Skywork의 오픈소스 PRM (vLLM 0.13.0 호환)

## 환경 설정

```bash
# 가상 환경 활성화
source .venv/bin/activate

# 필요한 경우 의존성 설치
pip install vllm transformers torch

# Skywork-o1-Open-PRM 사용을 위한 vLLM 플러그인 설치 (필수)
pip install -e .
```

**중요:** Skywork-o1-Open-PRM을 사용하려면 반드시 `pip install -e .`로 플러그인을 설치해야 합니다. 이를 통해 vLLM이 시작할 때 `SkyworkQwen2ForPrmModel`을 자동으로 등록합니다.

## 사용법

### 1. 기본 Reward Model 테스트

간단한 텍스트 프롬프트로 보상 모델을 테스트합니다.

```bash
python reward.py
```

**특징:**
- Qwen2.5-Math-PRM-7B 모델 사용
- 단순 텍스트 프롬프트 입력
- 기본적인 vLLM pooling/reward 모델 사용법 예시

### 2. Qwen PRM - 수학 단계별 평가

Qwen2.5-Math-PRM 포맷으로 수학 문제 풀이의 단계별 보상을 계산합니다.

```bash
python reward_qwen_prm.py
```

**특징:**
- `<extra_0>` 토큰으로 단계 구분
- `<im_start>`/`<im_end>` 채팅 템플릿 마커 사용
- 단계별 추론 평가

**출력 예시:**
```
Prompt: '<im_start>system\n...<im_end><|endoftext|>'
Reward: [0.591, 0.529, 0.634, 0.625] (size=4)
```

### 3. Skywork-o1-Open-PRM

Skywork-o1-Open-PRM 모델을 사용한 추론을 두 가지 방식으로 실행할 수 있습니다.

#### 옵션 A: 직접 실행 (권장)

```bash
python reward_skywork_o1_prm.py
```

**특징:**
- 단일 스크립트로 모델 로드 및 추론
- 자동으로 vLLM 0.13.0 호환 모델 등록
- GPU 메모리 직접 사용

**출력 예시:**
```
================================================================================
SKYWORK-O1-OPEN-PRM STEP-WISE REWARDS
================================================================================

Problem 1:
--------------------------------------------------------------------------------
Number of steps: 4

Step-wise rewards:
  Step 1: 0.5910 | To find out how many more pink plastic flamingos...
  Step 2: 0.5298 | On Saturday, they take back one third of the flamingos...
  Step 3: 0.6344 | On Sunday, the neighbors add another 18 pink plastic...
  Step 4: 0.6254 | To find the difference, subtract the number of white...

Average reward (for Best-of-N ranking): 0.5952
```

#### 옵션 B: 서버/클라이언트 모드

여러 클라이언트에서 동일한 모델을 공유하거나, 원격 추론이 필요한 경우 사용합니다.

**터미널 1: 서버 시작**
```bash
python start_reward_server.py
```

서버 옵션 커스터마이징:
```bash
python start_reward_server.py \
    --model Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

**터미널 2: 클라이언트 실행**
```bash
python reward_skywork_server.py
```

클라이언트 옵션:
```bash
python reward_skywork_server.py \
    --model Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --base-url http://localhost:8081/v1 \
    --api-key EMPTY
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
- `--model`: 사용할 모델 경로/이름
- `--max-model-len`: 최대 시퀀스 길이 (기본값: 1024)
- `--tensor-parallel-size`: GPU 병렬화 수 (기본값: 1)
- `--gpu-memory-utilization`: GPU 메모리 사용률 (기본값: 0.9)
- `--trust-remote-code`: 원격 코드 실행 허용 (필수)

## 프로젝트 구조

```
.
├── reward.py                      # 기본 보상 모델 예제
├── reward_qwen_prm.py            # Qwen PRM 단계별 평가
├── reward_skywork_o1_prm.py      # Skywork PRM 직접 실행
├── reward_skywork_server.py      # Skywork PRM 클라이언트
├── start_reward_server.py        # Skywork PRM 서버 시작 스크립트
├── skywork_prm_model.py          # Skywork 커스텀 모델 구현 (vLLM 플러그인)
├── skywork_utils.py              # Skywork 유틸리티 함수
├── pyproject.toml                # 패키지 설정 및 vLLM 플러그인 등록
├── README.md                     # 본 파일 (사용자 가이드)
├── CLAUDE.md                     # Claude Code용 프로젝트 문서
└── IMPLEMENTATION_SUMMARY_V2.md  # 기술 구현 세부사항
```

### 핵심 파일 설명

- **reward.py**: vLLM의 기본 pooling/reward 모델 사용법을 보여주는 기본 예제
- **reward_qwen_prm.py**: Qwen2.5-Math-PRM 포맷을 사용한 수학 단계별 추론 평가
- **reward_skywork_o1_prm.py**: Skywork-o1-Open-PRM의 직접 실행 스크립트
- **start_reward_server.py**: vLLM 서버를 시작하는 헬퍼 스크립트
- **reward_skywork_server.py**: OpenAI 호환 API를 사용하는 서버/클라이언트 모드 클라이언트
- **skywork_prm_model.py**: Skywork-o1-Open-PRM의 vLLM 0.13.0 호환 커스텀 모델 구현 (vLLM 플러그인)
- **skywork_utils.py**: Skywork PRM 입력 준비 및 sigmoid 정규화 유틸리티 함수
- **pyproject.toml**: 패키지 메타데이터 및 vLLM 플러그인 entry point 정의

모든 스크립트는 vLLM의 `LLM.reward()` API를 `runner="pooling"` 설정과 함께 사용합니다.

## vLLM 0.13.0 호환성

Skywork-o1-Open-PRM은 `Qwen2ForPrmModel` 아키텍처를 사용하며, vLLM의 표준 `Qwen2ForProcessRewardModel`과 다른 `v_head` 파라미터 구조를 가지고 있습니다.

### vLLM 플러그인 시스템

이 프로젝트는 vLLM의 공식 플러그인 시스템을 사용하여 `SkyworkQwen2ForPrmModel`을 자동으로 등록합니다.

**설치 방법:**
```bash
pip install -e .
```

이 명령은 `pyproject.toml`에 정의된 entry point를 등록하여, vLLM이 시작할 때 자동으로 Skywork PRM 모델을 인식하게 합니다.

**커스텀 모델 구현 (`skywork_prm_model.py`):**
- Skywork의 정확한 아키텍처 구현 (`ValueHead` with `v_head` 파라미터)
- vLLM 0.13.0의 네이티브 API 사용 (`Pooler`, `DispatchPooler`)
- 프로세스 수준 보상을 위한 STEP pooling 지원
- vLLM의 `ModelRegistry`에 자동 등록 (`Qwen2ForPrmModel` → `SkyworkQwen2ForPrmModel`)

**플러그인 Entry Point:**
```toml
[project.entry-points."vllm.general_plugins"]
register_skywork_prm = "skywork_prm_model:register_skywork_prm_model"
```

플러그인이 정상적으로 로드되면 vLLM 시작 시 다음 메시지가 표시됩니다:
```
✓ Registered SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) for Skywork-o1-Open-PRM
```

기술적 세부사항은 `IMPLEMENTATION_SUMMARY_V2.md`를 참조하세요.

### 플러그인 확인

vLLM이 플러그인을 정상적으로 로드했는지 확인하려면:

```bash
# vLLM 시작 로그에서 다음 메시지 확인
✓ Registered SkyworkQwen2ForPrmModel (Qwen2ForPrmModel) for Skywork-o1-Open-PRM

# 또는 Python에서 직접 확인
python -c "from vllm import ModelRegistry; print('Qwen2ForPrmModel' in ModelRegistry.get_supported_archs())"
```

## Best-of-N 랭킹 활용

Process Reward Model은 여러 후보 솔루션 중 최선을 선택하는 Best-of-N 전략에 사용할 수 있습니다:

1. 동일한 문제에 대해 N개의 솔루션을 생성
2. 각 솔루션의 단계별 보상 점수를 계산
3. 평균 보상이 가장 높은 솔루션을 선택

```python
# 예시: 4개 솔루션의 평균 보상
solutions_avg_rewards = [0.5952, 0.6234, 0.5678, 0.6421]
best_solution_idx = np.argmax(solutions_avg_rewards)  # 3번 솔루션 선택
```

## 문제 해결

### CUDA Out of Memory
```bash
# GPU 메모리 사용률 낮추기
python reward_skywork_o1_prm.py --gpu-memory-utilization 0.7
```

### 서버 연결 실패
```bash
# 서버가 실행 중인지 확인
curl http://localhost:8081/v1/models
```

### 모델 다운로드 실패
```bash
# Hugging Face 토큰 설정 (비공개 모델의 경우)
export HF_TOKEN=your_token_here
```

## 참고 자료

- [vLLM Pooling Models 문서](https://docs.vllm.ai/en/stable/models/pooling_models/)
- [Skywork-o1-Open-PRM on HuggingFace](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B)
- [Qwen2.5-Math-PRM-7B on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)

## 라이선스

Apache-2.0 License
