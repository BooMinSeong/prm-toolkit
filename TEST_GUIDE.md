# Token Validation Testing Guide

이 문서는 PRM Toolkit의 token length validation 기능을 테스트하는 방법을 안내합니다.

## 1. 서버 없이 테스트 (Unit Tests)

validation 로직만 테스트 (서버 연결 불필요):

```bash
# 모든 validation 테스트 실행
uv run python test_validation.py
```

**테스트 항목:**
- ✓ Qwen PRM: truncation 동작, delimiter 보존 (double newline)
- ✓ Skywork PRM: truncation 동작, delimiter 보존 (single newline)
- ✓ Edge cases: invalid max_tokens, 극단적으로 작은 값
- ✓ 다양한 max_tokens 값 (256, 512, 1024, 2048, 4096)

## 2. 실제 서버와 통합 테스트

### 2-1. Qwen PRM 서버 테스트

#### Step 1: Qwen PRM 서버 시작

Terminal 1에서:
```bash
vllm serve Qwen/Qwen2.5-Math-PRM-7B \
    --port 8082 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

#### Step 2: 기본 테스트 (default max_tokens=4096)

Terminal 2에서:
```bash
# 단일 입력 테스트
uv run python example_prm_usage.py --model qwen

# 배치 테스트
uv run python example_prm_usage.py --model qwen --batch
```

**기대 결과:**
- ✓ 정상적으로 score 계산됨
- ✓ Truncation 경고 없음 (입력이 4096 토큰 이하이므로)

#### Step 3: Truncation 테스트 (max_tokens=512)

```bash
# 작은 max_tokens로 truncation 강제 발동
uv run python example_prm_usage.py --model qwen --max-tokens 512

# Truncation 데모 (100 steps → 자동 truncation)
uv run python example_prm_usage.py --model qwen --demo-truncation
```

**기대 결과:**
```
WARNING: Qwen: Truncated 1429 → 512 tokens (99 → 34 steps)
✓ Truncation occurred (as expected)
Original steps: 99
Truncated steps: 34
```

#### Step 4: 다양한 max_tokens 값 테스트

```bash
# 256 토큰
uv run python example_prm_usage.py --model qwen --max-tokens 256

# 1024 토큰
uv run python example_prm_usage.py --model qwen --max-tokens 1024

# 2048 토큰
uv run python example_prm_usage.py --model qwen --max-tokens 2048
```

### 2-2. Skywork PRM 서버 테스트

#### Step 1: Skywork PRM 서버 시작

Terminal 1에서:
```bash
# vLLM plugin 먼저 설치 (필수!)
uv pip install -e .

# 서버 시작
python start_reward_server.py

# 또는 수동으로:
vllm serve Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --port 8081 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

#### Step 2: 기본 테스트

Terminal 2에서:
```bash
# 단일 입력 테스트
uv run python example_prm_usage.py --model skywork

# 배치 테스트
uv run python example_prm_usage.py --model skywork --batch
```

#### Step 3: Truncation 테스트

```bash
# 작은 max_tokens
uv run python example_prm_usage.py --model skywork --max-tokens 512

# Truncation 데모
uv run python example_prm_usage.py --model skywork --demo-truncation
```

**기대 결과:**
```
WARNING: Skywork: Truncated 1384 → 512 tokens (99 → 37 steps)
✓ Truncation occurred (as expected)
```

## 3. 커스텀 테스트 스크립트

직접 Python 코드로 테스트:

```python
from prm_toolkit import PrmConfig, load_prm_server

# 1. 설정 생성
config = PrmConfig(
    prm_path="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8082",
    max_tokens=512  # 원하는 값으로 조정
)

# 2. PRM 서버 인스턴스 생성
prm = load_prm_server(config)

# 3. 테스트 입력 준비
prompt = "Your math problem here"
response = "Step 1: ...\n\nStep 2: ...\n\nStep 3: ..."

# 4. Score 계산 (자동으로 validation 수행)
rewards = prm.score(prompt, response)

# 5. 결과 확인
print(f"Number of steps: {len(rewards)}")
for i, reward in enumerate(rewards, 1):
    print(f"Step {i}: {reward:.6f}")
```

## 4. 검증 체크리스트

### Validation Logic (서버 불필요)
- [ ] `test_validation.py` 모든 테스트 통과
- [ ] Qwen: double newline delimiter 보존 확인
- [ ] Skywork: single newline delimiter 보존 확인
- [ ] Edge cases: invalid max_tokens 처리 확인
- [ ] 다양한 max_tokens 값에서 올바른 truncation

### Integration (서버 필요)
- [ ] Default max_tokens (4096)로 정상 동작
- [ ] Truncation 발생 시 warning 로그 출력
- [ ] Truncated input도 정상적으로 score 계산
- [ ] Batch processing에서도 truncation 동작
- [ ] Prompt는 보존되고 response만 truncation됨

### End-to-End
- [ ] `example_prm_usage.py` 기본 실행
- [ ] `--max-tokens` 옵션으로 다양한 값 테스트
- [ ] `--demo-truncation` 데모 실행
- [ ] `--batch`로 배치 처리 테스트

## 5. 트러블슈팅

### 문제: "max_tokens must be positive" 에러
**해결:** max_tokens를 1 이상의 값으로 설정

### 문제: "Extreme truncation - prompt exceeded max_tokens" 에러
**해결:** max_tokens 값을 늘리거나 prompt를 짧게 수정

### 문제: Truncation이 발생하지 않음
**확인:**
1. 입력이 실제로 max_tokens보다 긴지 확인
2. Logging level이 WARNING 이상인지 확인
3. 올바른 delimiter 사용 (Qwen: `\n\n`, Skywork: `\n`)

### 문제: 서버 연결 실패
**확인:**
1. vLLM 서버가 실행 중인지 확인
2. `base_url`이 올바른지 확인
3. 포트 번호 확인 (Qwen: 8082, Skywork: 8081)

## 6. 성능 테스트

Truncation이 성능에 미치는 영향 측정:

```python
import time
from prm_toolkit import PrmConfig, load_prm_server

# Without truncation (large max_tokens)
config_large = PrmConfig(
    prm_path="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8082",
    max_tokens=8192
)
prm_large = load_prm_server(config_large)

# With truncation (small max_tokens)
config_small = PrmConfig(
    prm_path="Qwen/Qwen2.5-Math-PRM-7B",
    base_url="http://localhost:8082",
    max_tokens=512
)
prm_small = load_prm_server(config_small)

# Create long input
long_response = '\n\n'.join([f'Step {i}: ...' for i in range(100)])

# Measure with large limit
start = time.time()
rewards_large = prm_large.score("Test", long_response)
time_large = time.time() - start

# Measure with truncation
start = time.time()
rewards_small = prm_small.score("Test", long_response)
time_small = time.time() - start

print(f"Large limit: {len(rewards_large)} steps, {time_large:.3f}s")
print(f"Small limit: {len(rewards_small)} steps, {time_small:.3f}s")
print(f"Speedup: {time_large/time_small:.2f}x")
```

## 요약

**서버 없이 빠른 테스트:**
```bash
uv run python test_validation.py
```

**서버와 함께 전체 테스트:**
```bash
# Terminal 1: 서버 시작
vllm serve Qwen/Qwen2.5-Math-PRM-7B --port 8082 --trust-remote-code

# Terminal 2: 테스트 실행
uv run python example_prm_usage.py --model qwen
uv run python example_prm_usage.py --model qwen --demo-truncation
uv run python example_prm_usage.py --model qwen --batch
```

모든 테스트가 통과하면 token validation 구현이 성공적으로 완료된 것입니다! 🎉
