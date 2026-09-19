#!/bin/bash
# Stage A — audio branch warm-up (audio-visual row only).
#   trains : audio_fusion.* (alpha gate + MLP)      frozen : LoRA, ClipEncoder adapter, LLM
#   data   : hierarchy off, 500 s window, 250 frames, neg_window on
#   steps  : 2,000 x (2 x 16 = 32)   LR 1e-3 on audio (1e-4 x 10)   ~4 h on a 4090, one job
# Usage: [SEED=0] bash scripts/audio/stage_a.sh
source "$(dirname "$0")/env.sh"
OUT=$CKPT/stageA_R2_seed$SEED
mkdir -p "$OUT"

python revisionllm/train/train.py \
  "${COMMON_TRAIN_ARGS[@]}" "${AUDIO_ARGS[@]}" \
  --training_stage 4 --stage2_path "$PUBLIC_STAGE2" \
  --tune_audio_fusion True --freeze_lora True --freeze_mm_mlp_adapter True \
  --audio_gate_init 0.0 --audio_lr_multiplier 10 \
  --modality_dropout_visual 0.0 --modality_dropout_audio 0.0 \
  --output_dir "$OUT" \
  --max_steps ${MAX_STEPS:-2000} --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --dataloader_num_workers ${WORKERS:-4} \
  2>&1 | tee -a "$OUT/train.log"
