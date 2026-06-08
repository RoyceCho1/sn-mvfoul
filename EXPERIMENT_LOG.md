# SoccerNet MVFoul — Experiment Log

> 기준일: 2026-06-06  
> 목표: SoccerNet MVFoul 데이터셋에서 파울 종류(action_class) 및 심각도(offence_severity) 분류

---

## 1. 태스크 정의

| 태스크 | 분류 대상 | 클래스 수 |
|---|---|---|
| Task 1 | action_class | 8 (Tackling, Standing tackling, High leg, Holding, Pushing, Elbowing, Challenge, Dive) |
| Task 2 | offence_severity | 4 (No offence, Offence + No card, Offence + Yellow card, Offence + Red card) |

**평가 지표**: Balanced Accuracy (seen classes)  
**VARS 논문 베이스라인**: Task1 = **47.0%**, Task2 = **43.0%**, Avg = **45.0%**

---

## 2. 데이터셋

```
data/SoccerNet/mvfouls/
  Train/  Valid/  Test/
    action_{id}/
      annotations.json   ← 파울 속성 (Contact, Bodypart, Severity 등)
      clip_0.mp4         ← 주 카메라 (항상 존재)
      clip_1.mp4 ...     ← 추가 카메라 앵글
```

| Split | Actions | clip_0 samples (official_target 필터 후) |
|---|---|---|
| Train | 2,916 | ~6,621 |
| Valid | 411 | 321 |
| Test | 301 | 247 |

**Annotation 필터 (`official_target`)**: action_class="Dont know", Offence="Between", Severity="2.0"/"4.0" 제외  
**클래스 불균형**: Standing tackling ~45%, Dive ~1.2%

---

## 3. 모델 구조

### 베이스 모델
- `nvidia/Cosmos-Reason2-8B` (Qwen3VLForConditionalGeneration)
- Vision Encoder (SigLIP) + LLM (Qwen3 8B)

### QLoRA 설정
```
4-bit NF4 quantization (bitsandbytes)
Vision Encoder: 완전 동결
LoRA 적용 대상: LLM attention + MLP 레이어
  r=16 (SV) / r=32 (MV)
  alpha=32 (SV) / alpha=64 (MV)
  dropout=0.05
실제 학습 파라미터: ~20M (전체의 ~0.25%)
```

---

## 4. 학습 방식 (Reasoning)

### 핵심 아이디어: think + answer 동시 학습
모델이 reasoning trace를 생성한 뒤 JSON 답을 내도록 학습.

**학습 타겟 형식**:
```
<think>
Analyzing the video clip:
Contact: There is clear physical contact between the players.
Body part: The challenge involves the lower body (legs/feet).
Ball interaction: The player attempts to play the ball but does not make contact with it.
A lower body challenge from a standing position... [action별 고정 reasoning]
The card decision depends on the force of impact... [severity 기준]
</think>
<answer>{"action_class": "Standing tackling", "offence_severity": "Offence + Yellow card"}</answer>
```

**Reasoning trace 합성 (`synthesize_think`)**: 실제 사람이 쓴 reasoning 없음 → annotation 속성(Contact, Bodypart, Try to play, Touch ball)에서 deterministic하게 생성

**Loss 마스킹**: assistant turn 전체(`<think>` + `<answer>`) 포함, user 프롬프트는 -100

### 비디오 입력
- **Foul-anchored sampling**: VARS 스펙상 t=3.0s(5초 클립의 60% 지점)에 파울 발생 → 해당 시점 밀집 샘플링
- **Frame hint**: 프롬프트에 "Frame N of M이 충돌 순간" 명시
  - SV 16프레임: Frame 10 of 16
  - MV 32프레임: Frame 20 of 32

### 균형 샘플링
- `WeightedRandomSampler` + sqrt-inverse-frequency 가중치
- Standing tackling 45% → ~27%로 완화

### 반복 방지
- `repetition_penalty=1.5` in `model.generate()`

---

## 5. 실험 결과

### 베이스라인

| 모델 | 방식 | Valid Task1 | Valid Task2 | Avg |
|---|---|---|---|---|
| VARS 논문 | — | **47.0%** | **43.0%** | **45.0%** |
| Cosmos-8B zero-shot (JSON) | 직접 분류 | 11.8% | 25.0% | 18.4% |
| Cosmos-8B zero-shot (think+answer) | Reasoning 프롬프트 | 11.1% | 19.5% | 15.3% |

### 완료된 Fine-tuning 실험

| 실험 | 모델 | 설정 | Valid Task1 | Valid Task2 | Avg | 비고 |
|---|---|---|---|---|---|---|
| SV Reason v1 | 8B | LR=2e-4, 3ep, 16frames | **21.5%** | **25.2%** | **23.3%** | 현재 최고 |
| SV Reason v1 (Test) | 8B | — | 17.9% | 25.2% | 21.6% | |
| SV Reason Answer | 8B | answer-only masking | 5.8% | 15.1% | 10.4% | 폐기 |
| 2B MV Reason | 2B | r=32, 32frames, 2views | 4.8% | 1.9% | 3.4% | 실패 |
| 8B MV Reason v1 | 8B | LR=2e-4, 3ep, balanced 없음 | 10.6% | 9.3% | 10.0% | mode collapse |
| 8B MV Reason v1 (Test) | 8B | — | 8.8% | 9.6% | 9.2% | |

### 현재 실행 중 (2026-06-06 야간)

| 실험 | GPU | 진행 상황 | ETA |
|---|---|---|---|
| **SV Reason v2** | GPU 0 | step 390/11595 (3.3%) | ~7시간 |
| **MV Reason v2** | GPU 1 | step 160/11595 (1.4%) | ~17시간 |

**v2 개선사항**:
- Balanced sampling 추가 (sqrt-inverse-frequency)
- LR: 2e-4 → **1e-4** (epoch 2→3 불안정 방지)
- Epochs: 3 → **5** (안정 수렴)
- Frame hint 프롬프트 추가 (충돌 시점 프레임 번호 명시)
- repetition_penalty=1.5

---

## 6. 실패 원인 분석

### SV Reason Answer (answer-only masking) — 폐기
- `<think>` 블록을 loss에서 제외 → 모델이 think 종료 방법을 학습 못 함
- 추론 시 think 블록이 무한 반복 → `<answer>` 태그 도달 실패 (38.9%)
- 해결: think+answer 전체 loss 포함으로 전환

### 2B MV Reason — 실패
- 2B 모델 용량 부족: 멀티뷰 reasoning + JSON 동시 처리 불가
- CJK drift (중국어 생성), think 블록 붕괴 (78.8% 동일 boilerplate)
- JSON 스키마 오류 95.3%

### 8B MV Reason v1 — Mode Collapse
- balanced sampling 없음 → 95% 예측이 "Standing tackling"
- LR=2e-4에서 epoch 2→3 정확도 하락 (불안정)
- 해결: balanced sampling + LR 절반 + epochs 연장

---

## 7. 학습 설정 파일

| 스크립트 | 역할 |
|---|---|
| `scripts/sh/run_reason.sh` | SV Reason v2 train + eval |
| `scripts/sh/run_multiview_reason_8b.sh` | MV Reason v2 train + eval |
| `scripts/train/train_reason.py` | SV QLoRA 학습 |
| `scripts/train/train_multiview_reason.py` | MV QLoRA 학습 |
| `scripts/eval/eval_finetuned_reason.py` | SV 평가 (JSONL 스트리밍) |
| `scripts/eval/eval_finetuned_multiview_reason.py` | MV 평가 |
| `scripts/train/frame_utils.py` | Foul-anchored 프레임 추출, foul_frame_index() |

### SV Reason v2 하이퍼파라미터
```
MODEL_ID  = nvidia/Cosmos-Reason2-8B
NUM_FRAMES = 16
NUM_EPOCHS = 5
LR         = 1e-4
LORA_R     = 16, LORA_ALPHA = 32
GRAD_ACCUM = 8
MAX_NEW_TOKENS = 512
MAX_CLASS_WEIGHT = 3.0
--balanced-sampling
```

### MV Reason v2 하이퍼파라미터
```
MODEL_ID   = nvidia/Cosmos-Reason2-8B
NUM_FRAMES = 32 (per view, 2 views)
MAX_PIXELS = 10,000,000
NUM_EPOCHS = 5
LR         = 1e-4
LORA_R     = 32, LORA_ALPHA = 64
GRAD_ACCUM = 8
MAX_NEW_TOKENS = 1024
MAX_CLASS_WEIGHT = 3.0
--balanced-sampling
VRAM: ~23.2GB reserved
```

---

## 8. Skeleton 데이터 탐색

### 데이터 구조
```
data/SoccerNet/PDF1_skeleton_share/X-VARS/outputs/skeleton_yolo11/
  {train|valid|test}/action_{id}/clip_{idx}.npz
  pose_qc_{split}.csv   ← QC 상태 (ok / too_few_people / failed)
```

```
NPZ 구조:
  keypoints      (12, 10, 17, 3)  — 픽셀 좌표 + confidence
  keypoints_norm (12, 10, 17, 3)  — 정규화
  bboxes         (12, 10, 4)
  person_scores  (12, 10)         — YOLO 검출 신뢰도
  frame_indices  (12,)
  video_width, video_height, fps
```

### QC 필터 후 매칭 수 (status='ok')

| Split | 전체 | ok | too_few_people | failed |
|---|---|---|---|---|
| Train | 6,621 | 4,933 | 1,400 | 288 |
| Valid | 970 | 741 | 197 | 32 |
| Test | 706 | 527 | 145 | 34 |

### 시도 및 결론

| 접근 | 결과 | 결론 |
|---|---|---|
| 별도 skeleton 분류기 (MLP) | Valid action_acc=**8.4%** (랜덤=12.5%보다 낮음) | **폐기** |
| Late fusion (VLM + skeleton) | base 분류기가 나쁘면 VLM을 오히려 망침 | **폐기** |
| VLM 학습에 skeleton overlay | 파울 당사자 2명 식별 불가, 노이즈 리스크 큼 | **폐기** |
| Eval 시 skeleton text hint | 재학습 불필요, A/B 테스트 가능 | **보류** (VLM 결과 확인 후 판단) |

**핵심 문제**: 축구 경기 wide-angle에서 10명 검출 시 상위 2명 ≠ 파울 당사자. 관절 좌표만으로 foul 종류 구분은 인간도 어려운 과제.

---

## 9. 출력 디렉토리

```
outputs/
  qlora_cosmos8b_reason_2/        ← SV Reason v2 체크포인트 (학습 중)
  qlora_cosmos8b_multiview_reason/ ← MV Reason v2 체크포인트 (학습 중)
  reason_eval_2/                  ← SV v2 eval 결과
  multiview_reason_8b_eval/       ← MV eval 결과
  skeleton_model/                 ← skeleton MLP (사용 안 함)
  zero_shot_8b/                   ← zero-shot 베이스라인
  logs/
    sv_reason/   mv_reason/   skeleton/
```

---

## 10. 다음 단계 (v2 결과 확인 후)

1. **SV Reason v2** 결과가 v1 대비 얼마나 개선됐는지 확인 (balanced sampling 효과)
2. **MV Reason v2** 결과 확인 (mode collapse 해소 여부)
3. MV가 SV보다 좋다면 → 두 뷰 정보를 활용한 추가 실험 설계
4. skeleton text hint를 eval에 추가해서 비교 (선택적)
