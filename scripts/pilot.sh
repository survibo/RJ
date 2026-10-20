#!/usr/bin/env bash
# 파일럿 (RESEARCH_PLAN.md 진행 순서 2). GPU 1장에서 run 여러 개를 동시에 돌린다.
#
#   bash scripts/pilot.sh data       # n=50 m=5 데이터셋 4개 생성 (fraction 0.1% 0.3% 1% 3%)
#   bash scripts/pilot.sh model      # 모델 크기 비교: 규칙 3개 x 크기 3개, fraction 1%
#   MODEL="2 128 4 1e-3" bash scripts/pilot.sh fraction
#                                    # 고른 크기로 규칙 5개 x fraction 4개 x seed 2개
#   python summarize.py runs_pilot --out runs_pilot/summary.csv
#
# 환경 변수: JOBS (동시 run 수, 기본 4), STEPS (기본 50000), EVAL (기본 250)
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-4}
STEPS=${STEPS:-50000}
EVAL=${EVAL:-250}
N=50
M=5
N_TEST=10000
SEED_DATA=42
# C(50,5) = 2,118,760 의 0.1% 0.3% 1% 3%
TRAIN_COUNTS=(2119 6356 21188 63563)
BASE_COUNT=21188

data_dir() { echo "data/n${N}_m${M}_tr$1_te${N_TEST}_random_s${SEED_DATA}"; }

# 한 줄 = 한 run. "출력디렉토리 로그이름 train.py인자..."
run_all() {
  mkdir -p runs_pilot/logs
  xargs -P "$JOBS" -L 1 bash -c '
    runs="$0"; log="$1"; shift
    python train.py "$@" --runs-dir "$runs" --steps '"$STEPS"' --eval-every '"$EVAL"' \
      --n-eval 2048 --prefix-lm > "runs_pilot/logs/$log.log" 2>&1 \
      && echo "done  $log" || echo "FAIL  $log (runs_pilot/logs/$log.log)"
  '
}

case "${1:-}" in
  data)
    for tr in "${TRAIN_COUNTS[@]}"; do
      python generate_data.py --n $N --m $M --train-count "$tr" --n-test $N_TEST \
        --split-strategy random --seed $SEED_DATA
    done
    ;;
  model)
    # "층 차원 head lr"
    MODELS=("2 128 4 1e-3" "4 256 8 5e-4" "4 512 8 3e-4")
    for spec in "${MODELS[@]}"; do
      read -r L D H LR <<< "$spec"
      for w in sortnbr first nbr; do
        echo "runs_pilot/model_L${L}d${D} model_L${L}d${D}_${w} --data-dir $(data_dir $BASE_COUNT)" \
             "--wiring $w --n-layer $L --n-embd $D --n-head $H --lr $LR --seed 0"
      done
    done | run_all
    ;;
  fraction)
    read -r L D H LR <<< "${MODEL:?set MODEL, e.g. MODEL=\"2 128 4 1e-3\"}"
    for tr in "${TRAIN_COUNTS[@]}"; do
      for w in sortnbr sum first nbr nbrpos; do
        for s in 0 1; do
          echo "runs_pilot/fraction_L${L}d${D} frac_tr${tr}_${w}_s${s} --data-dir $(data_dir "$tr")" \
               "--wiring $w --n-layer $L --n-embd $D --n-head $H --lr $LR --seed $s"
        done
      done
    done | run_all
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 1
    ;;
esac
