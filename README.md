# Sorting Grokking v1 구현 명세

## 1. 목표

`[0, n)`에서 중복 없이 선택한 `m`개의 categorical token을 무작위 순서로 입력하고, task 규칙에 맞게 재배열된 `m`개의 token을 출력하도록 decoder-only Transformer를 학습한다.

확인할 항목:

* train set memorization
* test generalization
* delayed generalization 형태의 grokking 발생 여부
* task별 grokking 차이
* random split과 relation-complete split 차이

지원 task:

```text
ascending
mod
alternating
alt_mod
shift_mod
shift_alt_mod
```

각 task는 독립 run으로 학습한다.

---

# 2. 프로젝트 구조

```text
sorting-grokking/
├─ src/
│  ├─ tasks.py
│  ├─ data.py
│  ├─ model.py
│  └─ metrics.py
│
├─ generate_data.py
├─ train.py
├─ plot.py
│
├─ data/
├─ runs/
│
├─ requirements.txt
└─ README.md
```

---

# 3. Task 정의

`src/tasks.py`

```python
TASK_REGISTRY = {
    "ascending": ascending,
    "mod": mod_sort,
    "alternating": alternating,
    "alt_mod": alt_mod,
    "shift_mod": shift_mod,
    "shift_alt_mod": shift_alt_mod,
}
```

공통 인터페이스:

```python
target = TASK_REGISTRY[task](
    input_values,
    modulus=modulus,
)
```

모든 task 함수는 입력을 수정하지 않는 순수 함수로 작성한다.

## ascending

```python
sorted(xs)
```

## mod

```python
sorted(xs, key=lambda x: (x % modulus, x))
```

## alternating

정렬된 값이:

```text
a0 < a1 < ... < a(m-1)
```

이면:

```text
a0, a(m-1), a1, a(m-2), ...
```

순서로 출력한다.

## alt_mod

먼저 다음 key로 전체 순위를 계산한다.

```python
sorted(xs, key=lambda x: (x % modulus, x))
```

그 순위의 최솟값과 최댓값을 번갈아 출력한다. 예를 들어
`xs = [1, 2, 3, 4, 5]`, `modulus = 3`이면 순위는
`[3, 1, 4, 2, 5]`, target은 `[3, 5, 1, 2, 4]`다.

## shift_mod

입력의 첫 token을 sample별 offset으로 사용한다.

```python
offset = xs[0]
zs = [(x + offset) % modulus for x in xs]
target = sorted(zs)
```

원래 token이 아니라 변환된 token을 출력하며, modulo 충돌에 따른 중복 token을
허용한다. 첫 token도 변환과 출력에 포함한다. 따라서 이 task는 입력 순서에
의존한다.

## shift_alt_mod

`shift_mod`와 동일하게 `zs`를 만든 뒤, 정렬된 `zs`의 최솟값과 최댓값을
번갈아 출력한다.

`shift_mod`, `shift_alt_mod`는 출력 token이 sorting vocabulary 안에 있도록
다음을 요구한다.

```text
0 < modulus <= n
```

---

# 4. 데이터 형식

`[0, n)`에서 중복 없이 `m`개를 선택한 combination을 사용한다.

각 combination은 한 번 무작위 shuffle하여 입력으로 저장한다.

target은 파일에 저장하지 않고 학습 시 `tasks.py`에서 계산한다.

예:

```text
12 13 23 14 29
3 27 8 11 19
```

파일 구조:

```text
data/
└─ n30_m5_tr1000_te4096_random_s42/
   ├─ train.txt
   ├─ test.txt
   └─ meta.json
```

relation-complete 예:

```text
data/
└─ n30_m5_tr1000_te4096_relation-complete_s42/
```

---

# 5. 데이터 생성 CLI

예:

```bash
python generate_data.py \
  --n 30 \
  --m 5 \
  --train-count 1000 \
  --n-test 4096 \
  --split-strategy random \
  --seed 42
```

지원 옵션:

```text
--n
--m
--train-count
--n-test
--split-strategy random|relation-complete
--seed
--out
```

검증:

```text
n > 0
m > 0
m <= n
train_count > 0
n_test >= 0
train_count + n_test <= C(n,m)
train_count + n_test <= 5,000,000
```

train/test는 combination 기준으로 disjoint해야 한다.

동일 seed에서는 동일 split과 동일 input shuffle을 생성한다.

---

# 6. Random split

`--split-strategy random`

전체 `C(n,m)` combination 공간에서 seed 기반으로 중복 없이 필요한 수만 선택한다.

```text
C(n,m) <= 5,000,000  -> 기존처럼 전체 조합을 열거한 뒤 선택
C(n,m) > 5,000,000   -> 조합 rank를 직접 샘플링하고 해당 조합만 복원
```

따라서 random split에서는 `C(n,m)` 자체에 상한이 없고, 실제 생성하는
`train_count + n_test` 행에만 5,000,000 상한을 적용한다. 작은 조합 공간은
기존 생성 결과와 seed 호환성을 유지한다.

그 순서에서:

```text
앞 train_count개 → train
다음 n_test개    → test
```

로 사용한다.

train/test는 combination 기준으로 disjoint하다.

---

# 7. Relation-complete split

`--split-strategy relation-complete`

unordered pair:

```text
{a, b}
```

를 상대적 순서 관계로 본다.

전체 관계 수:

```text
C(n,2)
```

각 training combination은:

```text
C(m,2)
```

개의 pair를 포함한다.

relation-complete split은 train set이 모든 pair를 최소 한 번 포함하도록 구성한다.
전체 combination을 대상으로 gain을 계산하므로 이 전략에 한해
`C(n,m) <= 5,000,000`이어야 한다.

## 구성

1. 전체 pair 집합 생성
2. 아직 보지 못한 pair를 가장 많이 포함하는 combination을 greedy하게 선택
3. 모든 pair가 coverage될 때까지 반복
4. 남은 train budget은 seed 기반 random combination으로 채움
5. train에 포함되지 않은 combination에서 test를 선택

lower bound:

```text
ceil(C(n,2) / C(m,2))
```

`train_count`가 이보다 작으면 즉시 error.

greedy basis를 실제 구성한 결과:

```text
basis_size > train_count
```

이면 필요한 basis size를 표시하고 error.

---

# 8. `meta.json`

예:

```json
{
  "n": 30,
  "m": 5,
  "train_size": 1000,
  "test_size": 4096,
  "split_strategy": "random",
  "seed": 42
}
```

relation-complete에서는 추가:

```json
{
  "relation_basis_size": 52
}
```

`task`, `modulus`는 dataset metadata에 저장하지 않는다.

---

# 9. 모델

`src/model.py`

GPT-2 style decoder-only Transformer를 직접 구현한다.

구조:

```text
token embedding
learned positional embedding

N × Transformer block

final LayerNorm
LM head
```

Transformer block:

```text
Pre-LN
Causal Self-Attention
Residual

Pre-LN
GELU MLP
Residual
```

기본값:

```text
n_embd = 128
n_head = 4
n_layer = 2
MLP hidden = 4 * n_embd
dropout = 0
```

조건:

```text
n_embd % n_head == 0
```

token embedding과 LM head weight를 공유한다.

---

# 10. Vocabulary

sorting vocabulary:

```text
0 ... n-1
BOS
SEP
```

special token:

```text
BOS = n
SEP = n + 1
```

총 vocabulary 크기:

```text
n + 2
```

---

# 11. Sequence 형식

```text
BOS x1 x2 ... xm SEP y1 y2 ... ym
```

총 길이:

```text
2m + 2
```

EOS는 사용하지 않는다.

---

# 12. Loss

autoregressive cross entropy를 사용한다.

다음 prediction에만 loss 적용:

```text
SEP      -> y1
y1       -> y2
...
y(m-1)   -> ym
```

입력 구간 prediction은 loss에서 제외한다.

출력 `m`개 token의 CE 평균을 사용한다.

---

# 13. 초기화

명시적으로 고정한다.

```text
Embedding:
Normal(0, 0.02)

Linear weight:
Normal(0, 0.02)

Linear bias:
0

LayerNorm weight:
1

LayerNorm bias:
0
```

동일 seed에서 동일 initialization을 얻어야 한다.

---

# 14. Optimizer

AdamW.

기본:

```text
lr = 1e-3
betas = (0.9, 0.98)
weight_decay = 1.0
```

모든 trainable parameter에 동일한 weight decay를 적용한다.

parameter grouping은 사용하지 않는다.

---

# 15. Learning rate

기본:

```text
warmup = 10
schedule = constant
```

앞 10 step은 linear warmup:

```text
0 -> configured lr
```

이후 constant lr 유지.

다른 scheduler는 v1에서 지원하지 않는다.

---

# 16. Training CLI

예:

```bash
python train.py \
  --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task ascending \
  --steps 100000
```

mod:

```bash
python train.py \
  --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task mod \
  --modulus 5 \
  --steps 100000
```

추가 task:

```bash
python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task alt_mod --modulus 5 --steps 100000

python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task shift_mod --modulus 5 --steps 100000

python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task shift_alt_mod --modulus 5 --steps 100000
```

Grokfast-EMA:

```bash
python train.py \
  --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task mod \
  --grokfast --grokfast-alpha 0.98 --grokfast-lamb 2.0
```

기본 옵션:

```text
--task ascending
--modulus 5

--n-embd 128
--n-head 4
--n-layer 2

--batch-size 512

--lr 1e-3
--weight-decay 1.0
--warmup 10

--precision auto
--compile
--fused-adamw

--grokfast disabled
--grokfast-alpha 0.98
--grokfast-lamb 2.0
--grokfast-start-step 0

--steps 100000

--eval-every 250
--checkpoint-every 2500
--n-eval 4096

--seed 42
--runs-dir runs
```

CUDA에서는 BF16 지원 시 자동으로 BF16 autocast를 사용하고, compiled model과 fused
AdamW를 기본으로 사용한다. 각각 `--precision fp32`, `--no-compile`,
`--no-fused-adamw`로 비활성화할 수 있다. CPU는 FP32 eager mode를 사용한다.

GPU가 없으면 CPU 실행 가능.

multi-GPU는 지원하지 않는다.

---

# 17. Batch sampling

매 training step마다 train set에서 무작위 batch를 샘플링한다.

replacement sampling을 허용한다.

batch size가 train set 크기보다 커도 허용한다.

동일 seed에서는 동일 batch sequence를 재현할 수 있어야 한다.

---

# 18. Evaluation subset

`--n-eval K`개를 train/test 각각 평가한다.

split 크기가 K 이하이면 전수 평가한다.

evaluation sample은 run 시작 시 한 번 선택하고 모든 evaluation에서 동일하게 사용한다.

resume 후에도 동일 evaluation indices를 복원한다.

---

# 19. Teacher-forced metrics

다음 metric을 계산한다.

```text
train_loss
test_loss

train_token_acc
test_token_acc

train_exact_acc
test_exact_acc
```

## token accuracy

output `m`개 위치 중 올바르게 예측한 token의 비율.

## exact accuracy

output `m`개 token을 모두 맞힌 sample의 비율.

---

# 20. Greedy generation

prefix:

```text
BOS x1 ... xm SEP
```

만 모델에 입력한다.

각 step:

```text
argmax(next_token_logits)
```

으로 다음 token을 선택한다.

정확히 `m`개의 output token을 생성한다.

teacher forcing은 사용하지 않는다.

---

# 21. Generation metrics

다음 metric을 계산한다.

```text
train_gen_token_acc
test_gen_token_acc

train_gen_exact_acc
test_gen_exact_acc

train_gen_valid_acc
test_gen_valid_acc
```

## gen token accuracy

생성 output과 target의 position별 token accuracy.

## gen exact accuracy

생성된 `m`개 token 전체가 target과 일치한 sample 비율.

## gen valid accuracy

생성된 `m`개 token의 multiset이 target token의 multiset과 정확히 같으면 1.

판정:

```python
sorted(pred) == sorted(target)
```

기존 세 task와 `alt_mod`에서는 입력 permutation 판정과 같다. Shift task는
변환된 target에 중복 token이 생길 수 있으므로 target multiset을 기준으로 한다.

---

# 22. Norm metrics

evaluation마다 기록:

```text
param_norm
embd_norm
```

## param norm

전체 trainable parameter의 L2 norm.

## embd norm

token embedding weight의 L2 norm.

---

# 23. Metrics CSV

매 `eval_every` step마다 한 줄 append.

schema:

```text
step
wall_time
lr

train_loss
test_loss

train_token_acc
test_token_acc

train_exact_acc
test_exact_acc

train_gen_token_acc
test_gen_token_acc

train_gen_exact_acc
test_gen_exact_acc

train_gen_valid_acc
test_gen_valid_acc

param_norm
embd_norm
```

---

# 24. Run 출력

형식:

```text
runs/
└─ {task}_{split}_n{n}m{m}_tr{train_size}_s{seed}/
   ├─ config.json
   ├─ metrics.csv
   └─ ckpt_last.pt
```

예:

```text
runs/
├─ ascending_random_n30m5_tr1000_s42/
├─ ascending_relation-complete_n30m5_tr1000_s42/
├─ mod_random_n30m5_k5_tr1000_s42/
└─ shift_alt_mod_random_n30m5_k5_tr1000_s42/
```

기존 directory가 존재하면 error.

---

# 25. `config.json`

최소 다음 정보 저장:

```json
{
  "task": "ascending",
  "modulus": null,

  "n": 30,
  "m": 5,

  "train_size": 1000,
  "test_size": 4096,

  "split_strategy": "random",

  "seed": 42,

  "n_embd": 128,
  "n_head": 4,
  "n_layer": 2,

  "batch_size": 512,

  "lr": 0.001,
  "weight_decay": 1.0,
  "betas": [0.9, 0.98],

  "warmup": 10,
  "grokfast": false,
  "grokfast_alpha": 0.98,
  "grokfast_lamb": 2.0,
  "grokfast_start_step": 0,
  "steps": 100000,

  "eval_every": 250,
  "n_eval": 4096
}
```

---

# 26. Checkpoint

항상 최신 저장 시점의:

```text
ckpt_last.pt
```

하나만 유지한다.

기본적으로 평가는 250 step마다 실행하고 checkpoint는 2500 step마다 저장한다.
마지막 step에서는 주기와 관계없이 평가와 저장을 모두 실행한다.

checkpoint 저장 내용:

```text
model state
optimizer state
scheduler state
Grokfast EMA state (enabled runs)
global step

Python RNG state
NumPy RNG state
torch RNG state
CUDA RNG state

train sampling RNG state
eval indices
```

---

# 27. Resume

사용:

```bash
python train.py \
  --resume runs/<run>/ckpt_last.pt \
  --steps 500000
```

checkpoint에서 training state를 복원하고 이어서 학습한다.

`--steps`는 기존 run의 목표 step보다 크게 연장할 수 있다.

---

# 28. Plot

사용:

```bash
python plot.py \
  "runs/*/metrics.csv" \
  --out runs/grokking.png
```

3개 panel을 생성한다.

## Accuracy

```text
train_gen_exact_acc
test_gen_exact_acc
test_gen_valid_acc
```

## Loss

```text
train_loss
test_loss
```

y축 log.

## Norm

```text
param_norm
embd_norm
```

x축은 기본 log scale.

---

# 29. 기본 실험 설정

```text
n = 30
m = 5

n_embd = 128
n_head = 4
n_layer = 2
dropout = 0

batch_size = 512

AdamW
lr = 1e-3
betas = (0.9, 0.98)
weight_decay = 1.0

warmup = 10
constant lr

steps = 100000

eval_every = 250
n_eval = 4096

seed = 42
```

---

# 30. 기본 실험 순서

먼저 random split에서:

```text
ascending
mod
alternating
alt_mod
shift_mod
shift_alt_mod
```

을 실행한다.

train size 예:

```text
100
500
1000
5000
```

이후 동일 조건에서:

```text
random
relation-complete
```

를 비교한다.

그 다음 weight decay:

```text
0.3
1.0
3.0
```

를 비교한다.

100k step에서:

```text
train_gen_exact_acc 높음
test_gen_exact_acc 낮음
```

상태인 run은 resume하여:

```text
200k
500k
```

까지 연장한다.

---

# 31. v1 포함 기능

```text
ascending
mod
alternating
alt_mod
shift_mod
shift_alt_mod

random split
relation-complete split

GPT-2 decoder-only
categorical embeddings
causal LM training

AdamW
warmup + constant lr

teacher-forced:
  loss
  token accuracy
  exact accuracy

free-running:
  token accuracy
  exact accuracy
  permutation-valid accuracy

parameter norm
embedding norm

fixed evaluation subset

checkpoint
resume

CSV logging
basic plotting
```

---

# 32. v1 제외 기능

```text
multi-GPU / DDP

pair coverage reporting

shuffle robustness

embedding별 weight decay
optimizer parameter grouping

cosine/linear scheduler

automatic grokking onset detection

automatic hyperparameter sweep

multiple historical checkpoints

positive-control integration

config hash
dataset hash
git metadata

W&B
TensorBoard

mixed precision

variable sequence length
EOS generation
beam search
```
