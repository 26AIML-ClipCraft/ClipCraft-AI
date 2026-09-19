#!/bin/bash
# Data pipeline: official splits -> candidates -> rolling download/extract -> frozen subsets -> labels.
# Every step is resumable; re-run the same command after a job limit hit.
# Usage: bash scripts/audio/data_pipeline.sh <step> [args]
#   subsets                        sample Train/Val/Test-sub candidates (seed 42)
#   extract test|val|train [i N]   download + CLIP 2fps + CLAP 1s -> LMDB (split i of N for the 1-day limit)
#   finalize                       keep ids with both features, freeze lists, filter annotation JSONs
#   labels                         keyword + Claude labelling of Test-sub queries (H2)
source "$(dirname "$0")/env.sh"
STEP=${1:?step}
RAW=${RAW:-data/chapters}        # chapters_vmr_{train,val,test}.jsonl + activitynet-format annotations
TMP=${TMP:-$DATA/tmp_media}      # rolling media buffer (<= 100 GB)
mkdir -p "$SUBSETS" "$TMP"

case "$STEP" in
  subsets)
    python revisionllm/data/audio_pipeline/subsets.py \
      --train_jsonl "$RAW/chapters_vmr_train.jsonl" --val_jsonl "$RAW/chapters_vmr_val.jsonl" \
      --test_jsonl "$RAW/chapters_vmr_test.jsonl" --out_dir "$SUBSETS" \
      --train_hours ${TRAIN_HOURS:-10000} --val_videos 300 --test_videos 1000 --seed 42 ;;
  extract)
    WHICH=${2:?test|val|train}; I=${3:-0}; N=${4:-1}
    KEEP=none; [[ "$WHICH" == test ]] && KEEP=audio      # Test-sub audio backup (~20 GB)
    python revisionllm/data/audio_pipeline/download_and_extract.py \
      --ids "$SUBSETS/${WHICH}_sub_candidates.json" --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" \
      --tmp_dir "$TMP" --log "$SUBSETS/extract_${WHICH}.jsonl" --clip_path "$CLIP_PT" \
      --fps $FEATURE_FPS --hop_sec $AUDIO_HOP --download_workers ${DL_WORKERS:-4} --prefetch 8 \
      --split "$I" --total_split "$N" --keep_media $KEEP --max_hours $MAX_TRAIN_HOURS ;;
  finalize)
    python revisionllm/data/audio_pipeline/finalize_subsets.py \
      --cand_dir "$SUBSETS" --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" --out_dir "$SUBSETS" \
      --train_anno "$RAW/chapters_train.json" --test_anno "$RAW/chapters_test.json" --val_anno "$RAW/chapters_val.json" ;;
  labels)
    python revisionllm/data/audio_pipeline/label_query_types.py \
      --test_anno "$TEST_ANNO" --out "$QUERY_TYPES" ${NO_LLM:+--no_llm} ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
