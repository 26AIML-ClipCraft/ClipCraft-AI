#!/bin/bash
# Stage C — hierarchical (coarse-to-fine) re-alignment, LoRA only.
#   C-33  : 33 sub-videos, 3,000 x 32, LR 5e-5  (~12 h, one job)
#   C-100 : 100 sub-videos, 3,000 x 32, LR 3e-5 (~1 day => 2 jobs, auto-resume)
#   frozen: ClipEncoder adapter, audio_fusion (weights still saved into non_lora_trainables.bin)
#   NOTE  : 'alternate' hierarchy training needs dataloader_num_workers 0 — the dataset reads the
#           collator's iteration_step parity, which only exists in the main process.
# Usage: [SEED=0] bash scripts/audio/stage_c.sh R1|R2 33|100
source "$(dirname "$0")/env.sh"
ROW=${1:?usage: stage_c.sh R1|R2 33|100}
NV=${2:?usage: stage_c.sh R1|R2 33|100}
OUT=$CKPT/stageC${NV}_${ROW}_seed$SEED
mkdir -p "$OUT"

if [[ "$NV" == 33 ]]; then
  INIT=${INIT:-$CKPT/stageB_${ROW}_seed$SEED}; LR=5e-5
else
  INIT=${INIT:-$CKPT/stageC33_${ROW}_seed$SEED}; LR=3e-5
fi
EXTRA=()
if [[ $(row_is_audio "$ROW") == 1 ]]; then
  EXTRA=("${AUDIO_ARGS[@]}" --freeze_audio_fusion True)
fi

python revisionllm/train/train.py \
  "${COMMON_TRAIN_ARGS[@]}" "${EXTRA[@]}" \
  --training_stage 4 --stage2_path "$INIT" \
  --freeze_mm_mlp_adapter True \
  --hierarchy True --hierarchy_num_videos "$NV" --fix_hierarchy_zoom 5 --sparse_length ${SPARSE_LENGTH:-2500} \
  --output_dir "$OUT" \
  --max_steps ${MAX_STEPS:-3000} --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --learning_rate $LR \
  --dataloader_num_workers 0 --visual_cache_size 128 \
  2>&1 | tee -a "$OUT/train.log"
