#!/bin/bash
# Evaluate one ablation row on Test-sub (same list, same settings for every row).
#   1. dense sliding-window pass   (eval_nlq_negative.py, fine-tuned model on its temporal path); videos <= 500 s are
#                                  included (one window = whole video)
#   2. dense_metrics.py            mIoU / R1,R5@t over ALL queries (default; $LOG/result_dense.json)
#   Optional (STAGE2=1): hierarchical window retrieval (eval_nlq_retrieval_e2e2.py) + the original merge. That stage only
#   works for videos with >= ~100 windows (~167 min) and silently drops every other query, so it is off by default.
# Usage: bash scripts/audio/eval_row.sh <run_dir> <name> [AUDIO=1] [DROP_VISUAL=0] [DROP_AUDIO=0]
#   R0 : bash scripts/audio/eval_row.sh $PUBLIC_STAGE2 R0 AUDIO=0
#   R1 : bash scripts/audio/eval_row.sh $CKPT/stageC100_R1_seed0 R1 AUDIO=0
#   R2 : bash scripts/audio/eval_row.sh $CKPT/stageC100_R2_seed0 R2
#   R3 : DROP_VISUAL=1 bash scripts/audio/eval_row.sh $CKPT/stageC100_R2_seed0 R3
#   R4 : DROP_AUDIO=1  bash scripts/audio/eval_row.sh $CKPT/stageC100_R2_seed0 R4
source "$(dirname "$0")/env.sh"
RUN=${1:?run dir}; NAME=${2:?row name}
AUDIO=${AUDIO:-1}; DROP_VISUAL=${DROP_VISUAL:-0}; DROP_AUDIO=${DROP_AUDIO:-0}
LOG=${EVAL_ROOT:-$CKPT/eval}/$NAME
mkdir -p "$LOG/dense" "$LOG/e2e2"

AUDIO_EVAL=()
if [[ "$AUDIO" == 1 ]]; then
  AUDIO_EVAL=(--audio_fusion True --audio_feat_folder "$CLAP_LMDB" --audio_dim $AUDIO_DIM --audio_hop_sec $AUDIO_HOP
              --drop_visual "$DROP_VISUAL" --drop_audio "$DROP_AUDIO")
fi
MODEL_ARGS=(--clip_path "$CLIP_PT" --model_base "$VICUNA" --stage2 "$RUN"
            --pretrain_mm_mlp_adapter "$VTIMELLM_STAGE1_PROJ" --pretrain_clip_adapter "$PUBLIC_STAGE1_SPARSE"
            --attn_implementation "$ATTN" --adapter_input_dim 768 --feature_fps $FEATURE_FPS
            --data_path "$TEST_ANNO" --feat_folder "$CLIP_LMDB" --q_feat_dir "$QFEAT_TEST" --vis_feat_storage lmdb
            --video_ids "$TEST_IDS" --mad_prompt mad_grounding --debug False)

# 1) dense windows (temporal path of the 'alternate' model => iteration_step 1)
python revisionllm/eval/eval_nlq_negative.py "${MODEL_ARGS[@]}" "${AUDIO_EVAL[@]}" \
  --cross_attn True --clip_adapter_text True --clip_adapter_feature alternate --dense_iteration_step 1 \
  --debug_window $WINDOW --num_frames $FRAMES --batch ${DENSE_BATCH:-8} \
  --split 0 --total_split 1 --log_path "$LOG/dense" --score cosine_sim --topk_pool True --skip_small_videos False

# 2) metrics over all queries from the dense predictions
python revisionllm/eval/dense_metrics.py --grounding_path "$LOG/dense" --anno "$TEST_ANNO" --video_ids "$TEST_IDS" \
  --out "$LOG/result_dense.json"

if [[ "${STAGE2:-0}" == 1 ]]; then
  # 3) hierarchical retrieval over the dense predictions (only long videos survive) + original merge
  python revisionllm/eval/eval_nlq_retrieval_e2e2.py "${MODEL_ARGS[@]}" "${AUDIO_EVAL[@]}" \
    --cross_attn True --clip_adapter_text True --clip_adapter_feature cls --hierarchy True \
    --hierarchy_num_videos 100 --batch 100 --debug_window $WINDOW --num_frames $FRAMES \
    --split 0 --total_split 1 --log_path "$LOG/e2e2" --grounding_path "$LOG/dense" --distributed_retrieval 1
  python revisionllm/eval/metric_retrieval_forward_chapters.py \
    --grounding_path "$LOG/dense" --retrieval_path "$LOG/e2e2" \
    --distributed_grounding 1 --distributed_retrieval 1 --dump_path "$LOG/predictions_merged.txt"
fi
echo "row $NAME done -> $LOG"
