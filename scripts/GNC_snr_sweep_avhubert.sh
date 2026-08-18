#!/usr/bin/env bash
# GNC visual-severity x audio-SNR sweep for AV-HuBERT, LRS3 test set,
# Permutation SHAP.
# For each audio SNR target, launches all 5 GNC severity levels in parallel.

set -uo pipefail

GPUS=(0 1 2 3 4)
TYPE=GNC
SNR_LEVELS=(-5 -2.5 0 2.5 10 -10)   # -10 runs last, after the others finish (only GPUs 0-4 available)

ROOT=/ucappell                      # dir with test.tsv, test.wrd, dict.wrd.txt
AVH_CKPT=/aa4825/models/av_hubert/large_noise_pt_noise_ft_433h.pt
NOISE_WAV=/path/to/noise            # dir with {valid,test}.tsv noise manifest, see avhubert/preparation/README.md
OUT_DIR=output/shap_GNC_snr_sweep
WANDB_PROJECT=dr-shap-av-visual
NUM_SAMPLES_SHAP=2000
MAX_SAMPLES=50                      # e.g. 50 for a quick smoke test; empty = full test set

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$SCRIPT_DIR/../logs/GNC_snr_sweep_avhubert"
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
  +override.noise_wav="$NOISE_WAV"
  common_eval.path="$AVH_CKPT"
  override.modalities=['audio','video']
  common.user_dir=`pwd`
  generation.beam=1
)

if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# Severity levels to run for every SNR target.
JOBS=(
  "lvl1:1"
  "lvl2:2"
  "lvl3:3"
  "lvl4:4"
  "lvl5:5"
)

MAX_PARALLEL=${#GPUS[@]}

for snr in "${SNR_LEVELS[@]}"; do
  echo "=============================================="
  echo "Starting $TYPE sweep: snr=$snr"
  echo "=============================================="

  gpu_idx=0

  for job in "${JOBS[@]}"; do
    suffix="${job%%:*}"
    level="${job#*:}"

    gpu="${GPUS[$((gpu_idx % MAX_PARALLEL))]}"
    gpu_idx=$((gpu_idx + 1))

    # Make a filesystem-safe SNR name
    snr_name="${snr//-/neg_}"
    snr_name="${snr_name//./p}"

    exp_name="LRS3_AVHuBERT_shap_permutation_viddist-${TYPE}-snr${snr_name}-${suffix}"
    results_path="$OUT_DIR/decode/${exp_name}"
    mkdir -p "$results_path"

    echo "Launching $exp_name on GPU $gpu"
    echo "  snr=$snr"
    echo "  visual-noise-type=$TYPE level=$level"
    echo "  log=$LOGDIR/${exp_name}.log"

    CUDA_VISIBLE_DEVICES=$gpu python -B infer_s2s_shap.py \
      "${COMMON_ARGS[@]}" \
      --exp-name "$exp_name" \
      common_eval.results_path="$results_path" \
      override.noise_prob=1 \
      override.noise_snr="$snr" \
      override.visual_noise_type="$TYPE" \
      override.visual_noise_prob=1 \
      override.visual_noise_level="$level" \
      > "$LOGDIR/${exp_name}.log" 2>&1 &

  done

  echo "Waiting for all jobs at snr=$snr to finish..."
  wait
  echo "Finished snr=$snr"
  echo
done

echo "All AV-HuBERT $TYPE SNR-sweep runs finished."
echo "Logs in $LOGDIR/"
echo "Results (.npz per run) in $OUT_DIR/"
