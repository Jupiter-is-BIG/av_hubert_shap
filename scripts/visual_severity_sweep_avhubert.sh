#!/usr/bin/env bash
# Visual-severity sweep for AV-HuBERT, LRS3 test set, Permutation SHAP.
#
# Audio is left clean; only the video stream is degraded via
# override.visual_noise_type/level (see avhubert/distortions.py).
#
# Runs all severity levels for ONE distortion type in parallel,
# waits for that type to finish, then moves to the next type.

set -uo pipefail

GPUS=(0 1 2 3 4)
TYPES=(CC BW GNC GB JPEG)

ROOT=/ucappell                      # dir with test.tsv, test.wrd, dict.wrd.txt
AVH_CKPT=/aa4825/models/av_hubert/large_noise_pt_noise_ft_433h.pt
OUT_DIR=output/shap_visual_sweep
WANDB_PROJECT=Dr-SHAP2
NUM_SAMPLES_SHAP=2000
MAX_SAMPLES=50                                     # e.g. 50 for a quick smoke test; empty = full test set

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$SCRIPT_DIR/../logs/visual_sweep_avhubert"
OUT_DIR="$SCRIPT_DIR/../$OUT_DIR"
mkdir -p "$LOGDIR" "$OUT_DIR"

# cd into the avhubert/ folder: infer_s2s_shap.py's --config-dir is relative.
cd "$SCRIPT_DIR/../avhubert"

COMMON_ARGS=(
  --wandb-project "$WANDB_PROJECT"
  --shap-alg permutation
  --num-samples-shap "$NUM_SAMPLES_SHAP"
  --output-path "$OUT_DIR"
  --config-dir ./conf/
  --config-name s2s_decode
  dataset.gen_subset=test
  +override.data="$ROOT"
  +override.label_dir="$ROOT"
  common_eval.path="$AVH_CKPT"
  override.modalities=['audio','video']
  common.user_dir=`pwd`
  generation.beam=1
)

if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# Severity levels to run for every distortion type.
JOBS=(
  "lvl1:1"
  "lvl2:2"
  "lvl3:3"
  "lvl4:4"
  "lvl5:5"
)

MAX_PARALLEL=${#GPUS[@]}

for type in "${TYPES[@]}"; do

  echo ""
  echo "============================================================"
  echo "Starting distortion type: $type"
  echo "============================================================"
  echo ""

  gpu_idx=0

  for job in "${JOBS[@]}"; do
    suffix="${job%%:*}"
    level="${job#*:}"

    gpu="${GPUS[$((gpu_idx % MAX_PARALLEL))]}"
    gpu_idx=$((gpu_idx + 1))

    exp_name="LRS3_AVHuBERT_shap_permutation_viddist-${type}-${suffix}"
    results_path="$OUT_DIR/decode/${exp_name}"
    mkdir -p "$results_path"

    echo "Launching $exp_name on GPU $gpu"
    echo "  Log: $LOGDIR/${exp_name}.log"

    CUDA_VISIBLE_DEVICES=$gpu python -B infer_s2s_shap.py \
      "${COMMON_ARGS[@]}" \
      --exp-name "$exp_name" \
      common_eval.results_path="$results_path" \
      override.visual_noise_type="$type" \
      override.visual_noise_prob=1 \
      override.visual_noise_level="$level" \
      > "$LOGDIR/${exp_name}.log" 2>&1 &

  done

  echo ""
  echo "Waiting for all $type runs to finish..."
  wait

  echo ""
  echo "============================================================"
  echo "Finished distortion type: $type"
  echo "============================================================"
  echo ""

done

echo "============================================================"
echo "ALL visual severity sweeps finished."
echo "Logs in $LOGDIR/"
echo "Results (.npz per run) in $OUT_DIR/"
echo "============================================================"
