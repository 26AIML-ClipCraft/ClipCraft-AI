#!/bin/bash
# Stage B — short-window grounding with audio (or visual-only baseline).
#   R2 : starts from Stage A output; trains audio_fusion + ClipEncoder adapter + LoRA; modality dropout 15/15
#   R1 : starts from the public checkpoint; same data / steps / seed; NO audio input (fair baseline)
#   steps : 6,000 x (2 x 32 = 64)   LR 5e-5 (audio 5e-4)   ~1-1.5 days => 2 jobs of <=20 h, auto-resume
# Usage: [SEED=0] bash scripts/audio/stage_b.sh R1|R2      (re-run the same command to resume)
source "$(dirname "$0")/env.sh"
ROW=${1:?usage: stage_b.sh R1|R2}
OUT=$CKPT/stageB_${ROW}_seed$SEED
mkdir -p "$OUT"

EXTRA=()
if [[ $(row_is_audio "$ROW") == 1 ]]; then
  INIT=${INIT:-$CKPT/stageA_R2_seed$SEED}
  EXTRA=("${AUDIO_ARGS[@]}" --tune_audio_fusion True --audio_lr_multiplier 10
         --modality_dropout_visual ${MD_VISUAL:-0.15} --modality_dropout_audio ${MD_AUDIO:-0.15})
else
  INIT=${INIT:-$PUBLIC_STAGE2}
fi

python revisionllm/train/train.py \
  "${COMMON_TRAIN_ARGS[@]}" "${EXTRA[@]}" \
  --training_stage 4 --stage2_path "$INIT" \
  --tune_clip_adapter True \
  --output_dir "$OUT" \
  --max_steps ${MAX_STEPS:-6000} --per_device_train_batch_size 2 --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --dataloader_num_workers ${WORKERS:-4} \
  2>&1 | tee -a "$OUT/train.log"
