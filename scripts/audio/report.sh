#!/bin/bash
# Ablation report: per-row metrics (mean +- std over seeds), paired bootstrap R2-R1, H2 breakdown.
# Metrics come from the dense predictions of every row ($E/<row>/dense). Rows/seeds that were not evaluated are skipped,
# so the first comparison (R0, R1_seed0, R2_seed0, R4) works as is.
# Usage: bash scripts/audio/report.sh      (after eval_row.sh; seed dirs are EVAL_ROOT/<row>_seedN)
source "$(dirname "$0")/env.sh"
E=${EVAL_ROOT:-$CKPT/eval}
args=()
add() {   # add NAME dir...   -> --run NAME=dir/dense[,dir/dense...] for the dirs that exist
  local name=$1; shift; local list=()
  for d in "$@"; do if [[ -d "$d/dense" ]]; then list+=("$d/dense"); fi; done
  if [[ ${#list[@]} -gt 0 ]]; then args+=(--run "$name=$(IFS=,; echo "${list[*]}")"); fi
}
add R0 "$E/R0"
add R1 "$E/R1_seed0" "$E/R1_seed1" "$E/R1_seed2"
add R2 "$E/R2_seed0" "$E/R2_seed1" "$E/R2_seed2"
for r in R3 R4 R5 R6 R7; do add $r "$E/$r"; done
if [[ -f "$QUERY_TYPES" ]]; then args+=(--labels "$QUERY_TYPES"); fi
python revisionllm/eval/ablation_report.py "${args[@]}" --compare R1 R2 --n_boot 1000 --out "$E/ablation_report.json"
