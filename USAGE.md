# Sorting Grokking 사용법

명세는 [README.md](README.md). 여기는 실행 방법만.

```text
generate_data.py → data/<dataset>/{train,test}.txt + meta.json
train.py         → runs/<run>/{config.json, metrics.csv, ckpt_last.pt}
plot.py          → runs/grokking.png
```

target은 저장하지 않는다. dataset 하나로 3개 task를 모두 학습할 수 있고, task는 `train.py --task`가 정한다.

---

## 1. 시작 전 테스트

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python generate_data.py --n 10 --m 3 --train-count 60 --n-test 40 --seed 1 --out data/_smoke

python train.py --data-dir data/_smoke --task ascending --steps 40 \
  --eval-every 20 --n-eval 32 --batch-size 32 --n-embd 32 --n-head 2 --seed 7 --runs-dir runs_smoke

python train.py --resume runs_smoke/ascending_random_n10m3_tr60_s7/ckpt_last.pt --steps 60
python plot.py "runs_smoke/*/metrics.csv" --out runs_smoke/smoke.png

rm -rf runs_smoke data/_smoke        # PowerShell: Remove-Item -Recurse -Force runs_smoke, data/_smoke
```

기대 출력 (seed 같으면 숫자까지 일치):

```text
step       0 | loss 2.5250/2.5223 | gen_exact 0.000/0.000 | ... | pnorm 13.04
step      20 | loss 2.1235/2.1865 | gen_exact 0.031/0.000 | ... | pnorm 12.95
step      40 | loss 1.8038/1.8530 | gen_exact 0.156/0.125 | ... | pnorm 12.89
step      60 | loss 1.5372/1.6177 | gen_exact 0.281/0.156 | ... | pnorm 12.86   ← resume
```

resume 결과가 끊김 없이 돌린 것과 일치하면 통과.

---

## 2. 명령어

```bash
# 데이터 (--out 생략 시 data/n30_m5_tr1000_te4096_random_s42 자동)
python generate_data.py --n 30 --m 5 --train-count 1000 --n-test 4096 \
  --split-strategy random --seed 42

# 학습 (아래 값이 전부 기본값)
python train.py \
  --data-dir data/n30_m5_tr1000_te4096_random_s42 \
  --task ascending --modulus 5 \
  --n-embd 128 --n-head 4 --n-layer 2 \
  --batch-size 512 --lr 1e-3 --weight-decay 1.0 --warmup 10 \
  --steps 100000 --eval-every 250 --n-eval 4096 \
  --seed 42 --runs-dir runs --device auto

# 기본 설정이면 이것만
python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 --task ascending
python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 --task mod --modulus 5
python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 --task alternating

# 이어서 학습 (--steps만 바꿀 수 있음, 나머지는 config.json에서 복원)
python train.py --resume runs/ascending_random_n30m5_tr1000_s42/ckpt_last.pt --steps 500000

# 플롯 (glob은 반드시 따옴표)
python plot.py "runs/*/metrics.csv" --out runs/grokking.png \
  --x-scale log --norm-y-scale linear --dpi 150
```

---

## 3. 파라미터

### generate_data.py

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--n` | 필수 | 값 범위 `[0,n)`. vocab = `n+2` (BOS=n, SEP=n+1) |
| `--m` | 필수 | 샘플당 토큰 수. 시퀀스 길이 `2m+2` |
| `--train-count` | 필수 | train 개수. **난이도 핵심 knob** — 작을수록 grokking이 잘 보임 |
| `--n-test` | 필수 | test 개수. `0` 허용(test 컬럼은 빈 칸) |
| `--split-strategy` | random | `random` \| `relation-complete` |
| `--seed` | 42 | split 순서 + 입력 shuffle |
| `--out` | 자동 | `data/n{n}_m{m}_tr{tr}_te{te}_{split}_s{seed}` |

제약: `0 < m <= n`, `train_count + n_test <= C(n,m) <= 5,000,000`.

`relation-complete`는 train이 모든 pair `{a,b}`를 최소 한 번 포함하도록 greedy로 채운다. 관계는 다 봤지만 그 조합은 못 본 상태를 만들어 일반화와 암기를 분리한다. `train_count`가 `ceil(C(n,2)/C(m,2))`(n30m5 → 44) 미만이거나 greedy basis(seed42 → 54)보다 작으면 error.

### train.py

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--data-dir` | — | dataset 경로. `--resume` 없으면 필수 |
| `--task` | ascending | `ascending` / `mod`(`x%k`, 동률은 값 순) / `alternating`(`a0,a(m-1),a1,...`) |
| `--modulus` | 5 | mod 전용. run 이름에 `k5`로 들어감 |
| `--n-embd` / `--n-head` / `--n-layer` | 128 / 4 / 2 | `n_embd % n_head == 0` 필요. MLP hidden = `4*n_embd` |
| `--batch-size` | 512 | 매 step 복원추출. train set보다 커도 됨 |
| `--lr` | 1e-3 | AdamW |
| `--weight-decay` | 1.0 | 전 파라미터 동일 적용. **grokking 타이밍 knob** |
| `--warmup` | 10 | linear warmup step. 이후 constant |
| `--steps` | 100000 | 총 step |
| `--eval-every` | 250 | 평가 + CSV 1줄 + checkpoint 주기 |
| `--n-eval` | 4096 | 평가 샘플 수. run 시작 시 고정, resume 후에도 동일. **CPU면 이걸 줄이는 게 제일 빠름** |
| `--seed` | 42 | 초기화 + batch 순서 + eval subset |
| `--runs-dir` | runs | run 부모 디렉토리 |
| `--device` | auto | `auto` / `cpu` / `cuda` |

run 이름은 `{task}_{split}_n{n}m{m}[_k{mod}]_tr{train}_s{seed}`. **이미 있으면 error** — 지우거나 `--runs-dir`를 바꾼다. dropout 0, betas (0.9, 0.98), fp32 단일 GPU 고정.

### plot.py

`pattern`(glob 여러 개 가능), `--out`, `--x-scale`(기본 log, step 0 행 생략), `--norm-y-scale`(두 norm 스케일 차가 크면 log), `--dpi`. run은 색, metric은 선 스타일로 구분.

---

## 4. 지표

| 컬럼 | 의미 |
|---|---|
| `step` `wall_time` `lr` | step, 누적 초(resume 포함), 현재 lr |
| `*_loss` | 출력 m개 토큰 CE 평균 (teacher forcing) |
| `*_token_acc` / `*_exact_acc` | teacher forcing 위치별 정답률 / m개 전부 맞힌 비율 |
| `*_gen_token_acc` / `*_gen_exact_acc` | greedy 생성 기준 같은 지표 |
| `*_gen_valid_acc` | 생성 결과가 입력의 순열이기만 하면 1 |
| `param_norm` `embd_norm` | 전체 파라미터 / token embedding L2 norm |

**주 지표는 `test_gen_exact_acc`.** teacher-forced는 정답 prefix를 보여준 측정이라 낙관적이다.

읽는 순서:

1. `train_gen_exact_acc`가 1.0에 붙는 step = 암기 완료 시점
2. 그 뒤 `test_gen_exact_acc`가 급상승하는 step = grokking. **둘의 간격이 delayed generalization** (log-x에서 잘 보임)
3. `train_loss`는 0인데 `test_loss`가 정체/상승 = 순수 암기 구간. grokking 오면 뒤늦게 급락
4. `test_gen_valid_acc`가 `test_gen_exact_acc`보다 먼저 1.0 = "재배열해야 한다"는 형식은 배웠고 "어떤 순서인지"는 아직
5. `param_norm`/`embd_norm`이 꺾이는 구간과 test 상승 구간이 겹치는지 = 압축과 일반화의 동행 여부
6. `test_exact_acc`는 높은데 `test_gen_exact_acc`가 낮다 = 앞 토큰을 틀린 뒤 복구 못 함(오류 누적)

판정:

- 100k에서 train만 높으면 아직 grokking 전일 수 있다 → 200k/500k로 resume
- test가 train과 같이 처음부터 오르면 너무 쉬운 것 → `--train-count` 축소
- 셋 다 0에서 안 움직이면 학습 자체 실패 → lr, warmup 확인

---

## 5. 실험 순서 (bash)

```bash
# 1) random split × task 3종 × train size 4종
for TR in 100 500 1000 5000; do
  python generate_data.py --n 30 --m 5 --train-count $TR --n-test 4096 --seed 42
  for T in ascending mod alternating; do
    python train.py --data-dir data/n30_m5_tr${TR}_te4096_random_s42 --task $T --steps 100000
  done
done

# 2) random vs relation-complete
python generate_data.py --n 30 --m 5 --train-count 1000 --n-test 4096 --split-strategy relation-complete --seed 42
python train.py --data-dir data/n30_m5_tr1000_te4096_relation-complete_s42 --task ascending

# 3) weight decay 비교 (run 이름에 wd가 없으므로 --runs-dir로 분리)
for WD in 0.3 1.0 3.0; do
  python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 --task ascending \
    --weight-decay $WD --runs-dir runs_wd$WD
done

# 4) grokking 전인 run 연장
python train.py --resume runs/ascending_random_n30m5_tr1000_s42/ckpt_last.pt --steps 500000
```

n30m5 tr1000 ascending은 250 step만에 test 0.99라 grokking이 안 보인다. tr=100 / 500을 봐야 한다.

---

## 6. 자주 나는 error

| 메시지 | 조치 |
|---|---|
| `run directory already exists` | 지우거나 `--runs-dir` 변경 (덮어쓰기는 일부러 막음) |
| `--lr cannot be changed on resume` | resume은 `--steps`만 변경 가능. 나머지는 새 run |
| `--steps N < already completed step M` | `--steps`를 M보다 크게 |
| `train_count < lower bound` / `basis size > train_count` | 메시지가 알려주는 최소값 이상으로 `--train-count` |
| `exceeds MAX_COMBINATIONS` | `C(n,m)` 500만 초과. n/m 축소 |

**재현성**: 같은 seed면 dataset, 초기화, batch 순서, eval subset이 모두 같다. checkpoint에 RNG state 전부와 eval index가 들어가 resume해도 갈라지지 않는다. 단 GPU가 다르면 완전 일치는 보장하지 않는다.
