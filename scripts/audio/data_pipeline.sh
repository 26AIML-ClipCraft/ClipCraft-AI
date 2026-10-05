#!/bin/bash
# Data pipeline: official splits -> candidates -> rolling download/extract -> frozen subsets -> labels.
# Every step is resumable; re-run the same command after a job limit hit.
# Usage: bash scripts/audio/data_pipeline.sh <step> [args]
#   subsets                        sample Train/Val/Test-sub candidates (seed 42)
#   extract test|val|train [i N]   download + CLIP 2fps + CLAP 1s -> LMDB (split i of N for the 1-day limit)
#   extract_par test|val|train     same, but K parallel workers (GPUS="0 1", PROCS_PER_GPU=4 -> 8 workers), then merge
#   extract_local NAME IDS.json MEDIA_DIR   same parallel extraction for videos already on disk (no download; files are kept)
#   merge test|val|train           merge the workers' shard LMDBs into the main LMDBs (extract_par does this itself)
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
      --train_hours ${TRAIN_HOURS:-2500} --val_videos 300 --test_videos 1000 --seed 42 ;;
  extract)
    WHICH=${2:?test|val|train}; I=${3:-0}; N=${4:-1}
    KEEP=none; [[ "$WHICH" == test ]] && KEEP=audio      # Test-sub audio backup (~20 GB)
    TARGET=0     # val/test: stop at n_final successes (candidates beyond that are only spares for failed downloads)
    [[ "$WHICH" != train ]] && TARGET=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_final'])" "$SUBSETS/${WHICH}_sub_candidates.json")
    python revisionllm/data/audio_pipeline/download_and_extract.py \
      --ids "$SUBSETS/${WHICH}_sub_candidates.json" --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" \
      --tmp_dir "$TMP" --log "$SUBSETS/extract_${WHICH}.jsonl" --clip_path "$CLIP_PT" \
      --fps $FEATURE_FPS --hop_sec $AUDIO_HOP --download_workers ${DL_WORKERS:-4} --prefetch 8 \
      --split "$I" --total_split "$N" --keep_media $KEEP --target $TARGET --max_hours $MAX_TRAIN_HOURS ;;
  extract_par)
    WHICH=${2:?test|val|train}
    read -ra GPU_LIST <<< "${GPUS:-0 1}"; PPG=${PROCS_PER_GPU:-4}; K=$(( ${#GPU_LIST[@]} * PPG ))
    KEEP=none; [[ "$WHICH" == test ]] && KEEP=audio
    LIMIT=0   # val/test: first n_final + 10% spares for failed downloads (finalize keeps the first n_final that succeeded)
    if [[ "$WHICH" != train ]]; then
      NF=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_final'])" "$SUBSETS/${WHICH}_sub_candidates.json")
      LIMIT=$(( NF + NF / 10 + 1 ))
    fi
    INTERLEAVE_FLAG=""; [[ "$WHICH" == train ]] && INTERLEAVE_FLAG="--interleave"   # train list is sorted by length -> balance workers
    mkdir -p "$DATA/shards" "$DATA/test_audio_keep"
    echo "extract_par $WHICH: $K workers on GPUs ${GPU_LIST[*]}, limit_ids=$LIMIT"
    pids=()
    for ((i=0; i<K; i++)); do
      g=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=${OMP_THREADS:-2} MKL_NUM_THREADS=${OMP_THREADS:-2} \
      python revisionllm/data/audio_pipeline/download_and_extract.py \
        --ids "$SUBSETS/${WHICH}_sub_candidates.json" --limit_ids $LIMIT \
        --clip_lmdb "$DATA/shards/clip_${WHICH}_$i" --clap_lmdb "$DATA/shards/clap_${WHICH}_$i" \
        --also_done_clip "$CLIP_LMDB" --also_done_clap "$CLAP_LMDB" \
        --tmp_dir "$TMP/w$i" --keep_dir "$DATA/test_audio_keep" --log "$SUBSETS/extract_${WHICH}_w$i.jsonl" \
        --clip_path "$CLIP_PT" --fps $FEATURE_FPS --hop_sec $AUDIO_HOP \
        --download_workers ${DL_WORKERS:-2} --prefetch 4 --clip_batch ${CLIP_BATCH:-128} \
        --split $i --total_split $K ${INTERLEAVE_FLAG} --keep_media $KEEP --max_hours $MAX_TRAIN_HOURS \
        > "$SUBSETS/extract_${WHICH}_w$i.out" 2>&1 &
      pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [[ $rc != 0 ]] && echo "WARNING: some workers exited with an error; see $SUBSETS/extract_${WHICH}_w*.out"
    bash "$0" merge "$WHICH" ;;
  extract_local)   # NAME IDS.json MEDIA_DIR  (e.g. charades data/charades/all_ids.json data/charades/Charades_v1_480)
    NAME=${2:?name}; IDS=${3:?ids.json}; MEDIA=${4:?media dir}
    read -ra GPU_LIST <<< "${GPUS:-0 1}"; PPG=${PROCS_PER_GPU:-4}; K=$(( ${#GPU_LIST[@]} * PPG ))
    mkdir -p "$DATA/shards"
    echo "extract_local $NAME: $K workers on GPUs ${GPU_LIST[*]}, media=$MEDIA"
    pids=()
    for ((i=0; i<K; i++)); do
      g=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=${OMP_THREADS:-2} MKL_NUM_THREADS=${OMP_THREADS:-2} \
      python revisionllm/data/audio_pipeline/download_and_extract.py \
        --ids "$IDS" --media_dir "$MEDIA" --interleave \
        --clip_lmdb "$DATA/shards/clip_${NAME}_$i" --clap_lmdb "$DATA/shards/clap_${NAME}_$i" \
        --also_done_clip "$CLIP_LMDB" --also_done_clap "$CLAP_LMDB" \
        --tmp_dir "$TMP/${NAME}_w$i" --log "$SUBSETS/extract_${NAME}_w$i.jsonl" \
        --clip_path "$CLIP_PT" --fps $FEATURE_FPS --hop_sec $AUDIO_HOP --download_workers 2 --prefetch 4 \
        --clip_batch ${CLIP_BATCH:-128} --split $i --total_split $K --max_hours $MAX_TRAIN_HOURS \
        > "$SUBSETS/extract_${NAME}_w$i.out" 2>&1 &
      pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [[ $rc != 0 ]] && echo "WARNING: some workers exited with an error; see $SUBSETS/extract_${NAME}_w*.out"
    python revisionllm/data/audio_pipeline/merge_lmdb.py --dst "$CLIP_LMDB" --src_glob "$DATA/shards/clip_${NAME}_*"
    python revisionllm/data/audio_pipeline/merge_lmdb.py --dst "$CLAP_LMDB" --src_glob "$DATA/shards/clap_${NAME}_*" ;;
  merge)
    WHICH=${2:?test|val|train}
    python revisionllm/data/audio_pipeline/merge_lmdb.py --dst "$CLIP_LMDB" --src_glob "$DATA/shards/clip_${WHICH}_*"
    python revisionllm/data/audio_pipeline/merge_lmdb.py --dst "$CLAP_LMDB" --src_glob "$DATA/shards/clap_${WHICH}_*" ;;
  finalize)
    python revisionllm/data/audio_pipeline/finalize_subsets.py \
      --cand_dir "$SUBSETS" --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" --out_dir "$SUBSETS" \
      --train_anno "$RAW/chapters_train.json" --test_anno "$RAW/chapters_test.json" --val_anno "$RAW/chapters_val.json" ;;
  labels)
    python revisionllm/data/audio_pipeline/label_query_types.py \
      --test_anno "$TEST_ANNO" --out "$QUERY_TYPES" ${NO_LLM:+--no_llm} ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
