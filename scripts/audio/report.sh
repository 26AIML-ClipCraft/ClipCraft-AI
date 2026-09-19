#!/bin/bash
# Ablation report: per-row metrics (mean ± std over seeds), paired bootstrap R2−R1, H2 breakdown.
# Usage: bash scripts/audio/report.sh   (after eval_row.sh for R0..R7; seeds as EVAL_ROOT/<row>_seedN)
source "$(dirname "$0")/env.sh"
E=${EVAL_ROOT:-$CKPT/eval}
python revisionllm/eval/ablation_report.py \
  --run R0="$E/R0" \
  --run R1="$E/R1_seed0,$E/R1_seed1,$E/R1_seed2" \
  --run R2="$E/R2_seed0,$E/R2_seed1,$E/R2_seed2" \
  --run R3="$E/R3" --run R4="$E/R4" --run R5="$E/R5" --run R6="$E/R6" --run R7="$E/R7" \
  --compare R1 R2 --labels "$QUERY_TYPES" --n_boot 1000 --out "$E/ablation_report.json"
