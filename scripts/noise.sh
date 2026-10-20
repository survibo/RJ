#!/usr/bin/env bash
# 오답 섞기 실험 (Plan.md). GPU 1장에서 run 여러 개를 동시에 돌린다.
#
#   bash scripts/noise.sh data          # n=70 m=7 데이터셋 생성 (train 크기 = TRAIN)
#   JOBS=8 bash scripts/noise.sh speed  # 동시 실행 수 측정: 같은 run 을 JOBS 개 동시에 짧게
#   bash scripts/noise.sh pilot         # 후보 과제 5개를 0% 오답으로 1번씩
#   TASKS="asc add addsort" bash scripts/noise.sh main
#                                       # 본 실험: 과제 x 오답 5단계 + 삭제 2단계, seed 0 먼저 전부 -> seed 1
#   python summarize.py runs_noise --out runs_noise/summary.csv
#
# 환경 변수: JOBS (동시 run 수, 기본 8), STEPS (기본 30000), EVAL (기본 100),
#            TRAIN (train 크기, 기본 4096), MODEL ("층 차원 head lr", 기본 "2 128 4 1e-3"),
#            SEEDS (기본 "0 1")
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-8}
STEPS=${STEPS:-30000}
EVAL=${EVAL:-100}
TRAIN=${TRAIN:-4096}
MODEL=${MODEL:-"2 128 4 1e-3"}
SEEDS=${SEEDS:-"0 1"}
N=70
M=7
N_TEST=20000
SEED_DATA=42
NOISE_SEED=0
NOISE=(0 0.05 0.1 0.2 0.4)
DROP=(0.1 0.4)
DATA="data/n${N}_m${M}_tr${TRAIN}_te${N_TEST}_random_s${SEED_DATA}"
read -r L D H LR <<< "$MODEL"

# 과제 이름 -> 규칙 조합 (Plan.md 4절 난이도 사다리)
rules_of() {
  case "$1" in
    asc) echo "A:asc" ;;
    add) echo "Q:first > R:add" ;;
    addsort) echo "Q:first > R:add > A:asc" ;;
    addalt) echo "Q:first > R:add > A:high" ;;
    above) echo "AB:above,fn=add,fc=5" ;;
    *) echo "unknown task $1" >&2; exit 1 ;;
  esac
}

# 한 줄 = 한 run. "출력디렉토리 로그이름 train.py인자..." (규칙은 큰따옴표로 감싼다)
run_all() {
  xargs -P "$JOBS" -L 1 bash -c '
    runs="$0"; log="$1"; shift
    mkdir -p "$runs/logs"
    python train.py "$@" --runs-dir "$runs" --steps '"$STEPS"' --eval-every '"$EVAL"' \
      --n-eval 2048 > "$runs/logs/$log.log" 2>&1 \
      && echo "done  $log" || echo "FAIL  $log ($runs/logs/$log.log)"
  '
}

common() {  # 과제 seed
  echo "--data-dir $DATA --rules \"$(rules_of "$1")\" --n-layer $L --n-embd $D --n-head $H" \
       "--lr $LR --seed $2 --noise-seed $NOISE_SEED --no-sym-metrics"
}

case "${1:-}" in
  data)
    python generate_data.py --n $N --m $M --train-count "$TRAIN" --n-test $N_TEST \
      --split-strategy random --seed $SEED_DATA
    ;;
  speed)
    # 2000 step 만 돌려서 log 마지막 wall time 을 비교한다.
    STEPS=2000
    out="runs_speed/j${JOBS}"
    for s in $(seq 0 $((JOBS - 1))); do
      echo "$out speed_s${s} $(common add "$s")"
    done | run_all
    grep -h "^step *2000 " "$out"/logs/*.log | awk '{print $NF}'
    echo "JOBS=$JOBS: 위 시간이 run 하나가 2000 step 에 걸린 시간"
    ;;
  pilot)
    for t in asc add addsort addalt above; do
      echo "runs_noise/pilot pilot_${t} $(common "$t" 0)"
    done | run_all
    ;;
  main)
    for s in $SEEDS; do
      for t in ${TASKS:?set TASKS, e.g. TASKS=\"asc add addsort\"}; do
        for p in "${NOISE[@]}"; do
          echo "runs_noise/main main_${t}_ln${p}_s${s} $(common "$t" "$s") --label-noise $p"
        done
        for p in "${DROP[@]}"; do
          echo "runs_noise/main main_${t}_drop${p}_s${s} $(common "$t" "$s") --drop-frac $p"
        done
      done
    done | run_all
    ;;
  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac
