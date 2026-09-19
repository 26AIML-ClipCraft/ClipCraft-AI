#!/bin/bash
# Shared paths / hyper-parameters for the ClipCraft audio-visual experiment.
# Source this from every scripts/audio/*.sh. Override anything via the environment, e.g.
#   SEED=1 bash scripts/audio/stage_b.sh R2
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# ---- base weights ---------------------------------------------------------------
export VICUNA=${VICUNA:-checkpoints/vicuna-7b-v1.5}                                   # safetensors only
export VTIMELLM_STAGE1_PROJ=${VTIMELLM_STAGE1_PROJ:-checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin}
export PUBLIC_STAGE1_SPARSE=${PUBLIC_STAGE1_SPARSE:-checkpoints/public/chapters_stage1_sparse/non_lora_trainables.bin}
export PUBLIC_STAGE2=${PUBLIC_STAGE2:-checkpoints/public/chapters_stage2_long_100}     # LoRA dir (+ non_lora_trainables.bin)
export CLIP_PT=${CLIP_PT:-checkpoints/clip/ViT-L-14.pt}

# ---- data (all LMDB) --------------------------------------------------------------
export DATA=${DATA:-data/chapters_audio}
export CLIP_LMDB=${CLIP_LMDB:-$DATA/clip_l14_2fps_lmdb}
export CLAP_LMDB=${CLAP_LMDB:-$DATA/clap_1s_lmdb}
export QFEAT_TRAIN=${QFEAT_TRAIN:-$DATA/clip_L14_text_features_train}
export QFEAT_TEST=${QFEAT_TEST:-$DATA/clip_L14_text_features_test}
export SUBSETS=${SUBSETS:-$DATA/subsets}
export TRAIN_ANNO=${TRAIN_ANNO:-$SUBSETS/train_sub_anno.json}     # activitynet-format list (chapters_to_activitynet.py, filtered)
export TEST_ANNO=${TEST_ANNO:-$SUBSETS/test_sub_anno.json}        # chapters_test.json style, filtered
export TEST_IDS=${TEST_IDS:-$SUBSETS/test_sub.json}
export QUERY_TYPES=${QUERY_TYPES:-$SUBSETS/test_query_types.json}

# ---- fixed settings (do not change after feature extraction) ----------------------
export FEATURE_FPS=2
export AUDIO_DIM=512
export AUDIO_HOP=1.0
export FRAMES=250
export WINDOW=500

# ---- run bookkeeping -----------------------------------------------------------------
export SEED=${SEED:-0}
export CKPT=${CKPT:-checkpoints/audio}
export MAX_TRAIN_HOURS=${MAX_TRAIN_HOURS:-20}     # 1-day job limit: stop + checkpoint at 20 h, re-run to resume
export ATTN=${ATTN:-sdpa}

# Common HF-Trainer flags shared by every stage (single GPU, no DeepSpeed).
COMMON_TRAIN_ARGS=(
  --lora_enable True --lora_r 64 --lora_alpha 128
  --model_name_or_path "$VICUNA" --version v1
  --pretrain_mm_mlp_adapter "$VTIMELLM_STAGE1_PROJ"
  --pretrain_clip_adapter "$PUBLIC_STAGE1_SPARSE"
  --cross_attn True --clip_adapter_text True --clip_adapter_feature alternate --adapter_input_dim 768
  --attn_implementation "$ATTN"
  --data_path "$TRAIN_ANNO" --feat_folder "$CLIP_LMDB" --q_feat_dir "$QFEAT_TRAIN" --vis_feat_storage lmdb
  --dataset mad --feature_fps $FEATURE_FPS --num_frames $FRAMES --debug_window $WINDOW
  --neg_window True --neg_samples 1 --neg_factor 1
  --bf16 True --tf32 True --gradient_checkpointing True
  --lr_scheduler_type cosine --warmup_ratio 0.03 --weight_decay 0.
  --evaluation_strategy no --save_strategy steps --save_steps 500 --save_total_limit 2
  --logging_steps 10 --report_to tensorboard --model_max_length 2048 --lazy_preprocess True
  --seed $SEED --max_train_hours $MAX_TRAIN_HOURS
)

# Audio flags (only for the audio-visual rows: R2 and friends).
AUDIO_ARGS=(
  --audio_fusion True --audio_dim $AUDIO_DIM --audio_hop_sec $AUDIO_HOP --audio_feat_folder "$CLAP_LMDB"
)

# ROW = R1 (visual-only continued training) | R2 (audio-visual). Usage: row_is_audio R2 -> 0/1
row_is_audio() { [[ "$1" == R1 ]] && echo 0 || echo 1; }
