#!/bin/bash
# Charades-STA version of the first comparison (R0, R1, R2, R4), with VidChapters REPLAY mixed into the training data.
# Videos come straight from Allen AI (S3) - nothing is downloaded from YouTube. Research / non-commercial licence
# (Charades): keep the result in the research scope.
#
#   bash scripts/audio/charades.sh fetch     # unzip the needed videos (the zip must be downloaded: data/charades/Charades_v1_480.zip)
#   bash scripts/audio/charades.sh extract   # CLIP 2 fps + CLAP 1 s features for the Charades videos (GPU 0,1, minutes)
#   bash scripts/audio/charades.sh anno      # Charades-STA annotations, query features, VidChapters replay, training mix
#   bash scripts/audio/charades.sh run       # Stage A -> Stage B (R2) | Stage B (R1) -> evaluation R1, R2, R4, R0
#   bash scripts/audio/charades.sh status | cases
# Training mix: all Charades-STA train queries + VidChapters replay (REPLAY_SHARE=0.2 of the mix; the stored pool of long VidChapters videos is small, so its queries are repeated) to keep the long-video behaviour.
# Short training: STEPS_A=500, STEPS_B=400 (~1-1.5 passes over the mix).
export CKPT=${CKPT:-checkpoints/audio_charades}
export STEPS_A=${STEPS_A:-500} STEPS_B=${STEPS_B:-400}
source "$(dirname "$0")/env.sh"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
CH=data/charades
export TRAIN_ANNO=$CH/charades_mix_train_anno.json
export TEST_ANNO=$CH/charades_test_anno.json
export TEST_IDS=$CH/charades_test_ids.json
REPLAY_SHARE=${REPLAY_SHARE:-0.2}

case "${1:?fetch | extract | anno | run | status | cases}" in
  fetch)
    python - <<PY
import json
ids=set()
for sp in ("train","test"):
    for l in open("$CH/Charades_sta_%s.txt" % sp):
        if "##" in l: ids.add(l.split()[0])
json.dump(sorted(ids), open("$CH/all_ids.json","w")); print(len(ids), "videos needed")
open("$CH/needed_files.txt","w").write("\n".join("Charades_v1_480/%s.mp4" % i for i in sorted(ids)))
PY
    ( cd $CH && unzip -q -o Charades_v1_480.zip $(cat needed_files.txt | tr '\n' ' ') && ls Charades_v1_480 | wc -l ) ;;
  extract)
    GPUS=${GPUS:-"0 1"} PROCS_PER_GPU=${PROCS_PER_GPU:-4} TMP=${TMP:-/tmp/clipcraft_media} \
      bash $(dirname "$0")/data_pipeline.sh extract_local charades $CH/all_ids.json $CH/Charades_v1_480 ;;
  anno)
    set -x
    python revisionllm/data/audio_pipeline/charades_sta.py anno --sta_dir $CH --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" --out_dir $CH
    # VidChapters replay: stored Train videos that are long enough (>= 1000 s) for the loader's normal window branch
    python revisionllm/data/audio_pipeline/charades_sta.py replay_ids --candidates "$SUBSETS/train_sub_candidates_2500h.json" \
      --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" --out $CH/replay_ids.json
    python revisionllm/data/vidchap7m/chapters_to_activitynet.py --chapters_data_path data/chapters/chapters_vmr_train.jsonl \
      --chapters_out_path $CH/chapters_replay_anno.json --feat_lmdb "$CLIP_LMDB" --ids $CH/replay_ids.json
    ( cd revisionllm/data/feature_extraction && CUDA_VISIBLE_DEVICES=${GPU_A:-0} python chapters_clip_text_extractor.py \
        --chapters_dir ../../../data/chapters --splits train --key_style vid_idx --clip_path ../../../$CLIP_PT \
        --feature_output_path "$OLDPWD/$QFEAT_TRAIN" --ids "$OLDPWD/$CH/replay_ids.json" )
    CUDA_VISIBLE_DEVICES=${GPU_A:-0} python revisionllm/data/audio_pipeline/charades_sta.py text --out_dir $CH --clip_path "$CLIP_PT" \
      --qfeat_train "$QFEAT_TRAIN" --qfeat_test "$QFEAT_TEST"
    python revisionllm/data/audio_pipeline/charades_sta.py mix --out_dir $CH --replay_anno $CH/chapters_replay_anno.json \
      --replay_share $REPLAY_SHARE --out $TRAIN_ANNO
    echo "anno done" ;;
  smoke)   # 8 optimizer steps of Stage A on GPU 0 into a throw-away dir: does the training code run at all?
    CKPT=checkpoints/audio_charades_smoke MAX_STEPS=${SMOKE_STEPS:-8} CUDA_VISIBLE_DEVICES=${GPU_A:-0} bash $(dirname "$0")/stage_a.sh ;;
  run|status|cases)
    bash $(dirname "$0")/first_comparison.sh "$1" ;;
  *) echo "usage: $0 fetch | extract | anno | smoke | run | status | cases"; exit 1 ;;
esac
