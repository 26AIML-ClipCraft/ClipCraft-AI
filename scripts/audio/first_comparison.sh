#!/bin/bash
# First comparison: R0 (public checkpoint), R1 (visual-only), R2 (audio-visual), R4 (R2 with audio zeroed); seed 0,
# Stage A (R2 only) + Stage B, evaluated on Test-sub. Every long job is detached (setsid nohup), so it keeps running
# when the terminal / ssh session / app is closed, and it can be re-run to resume.
#
#   bash scripts/audio/first_comparison.sh prep     # after `extract_par train` finished: Train annotations + text features + finalize
#   bash scripts/audio/first_comparison.sh run      # GPU0: Stage A -> Stage B R2 -> eval R2, R4 | GPU1: Stage B R1 -> eval R1, R0
#   bash scripts/audio/first_comparison.sh status   # progress of the above
#   bash scripts/audio/report.sh                    # tables + paired bootstrap once the evals are done
#   bash scripts/audio/first_comparison.sh cases    # qualitative cases with the R4 (audio off) control -> $CKPT/eval/cases.md
#
# SHORT run by default (STEPS_A=1000, STEPS_B=1200) into checkpoints/audio_short, so it never collides with a later full run
# (full run: CKPT=checkpoints/audio STEPS_A=2000 STEPS_B=6000 bash scripts/audio/first_comparison.sh run).
# Needs `python` = the revaudio env on PATH. Override GPUs with GPU_A / GPU_B, the seed with SEED.
export CKPT=${CKPT:-checkpoints/audio_short}
export STEPS_A=${STEPS_A:-1000} STEPS_B=${STEPS_B:-1200}
source "$(dirname "$0")/env.sh"
HERE=scripts/audio
LOGS=$CKPT/logs; mkdir -p "$LOGS"
export PYTHONPATH=$PWD:${PYTHONPATH:-}

detach() { local name=$1; shift; setsid nohup "$@" > "$LOGS/$name.log" 2>&1 < /dev/null & echo "started $name -> $LOGS/$name.log"; }
finished() { [[ -f "$1/non_lora_trainables.bin" ]] || { echo "NOT finished: $1 (re-run the same command to resume)"; return 1; }; }

case "${1:?prep | run | status}" in
  prep) detach prep bash "$0" prep_job ;;
  prep_job)
    set -x
    # 1) ids that have both features, then Train annotations restricted to them
    python revisionllm/data/audio_pipeline/finalize_subsets.py --cand_dir "$SUBSETS" --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" --out_dir "$SUBSETS"
    python revisionllm/data/vidchap7m/chapters_to_activitynet.py --chapters_data_path data/chapters/chapters_vmr_train.jsonl \
      --chapters_out_path data/chapters/chapters_train.json --feat_lmdb "$CLIP_LMDB" --ids "$SUBSETS/train_sub.json"
    # 2) CLIP text features for the Train-sub queries only
    ( cd revisionllm/data/feature_extraction && CUDA_VISIBLE_DEVICES=${GPU_A:-0} python chapters_clip_text_extractor.py \
        --chapters_dir ../../../data/chapters --splits train --key_style vid_idx --clip_path ../../../$CLIP_PT \
        --feature_output_path "$OLDPWD/$QFEAT_TRAIN" --ids "$OLDPWD/$SUBSETS/train_sub.json" )
    # 3) final annotation JSONs (train_sub_anno.json / test_sub_anno.json)
    python revisionllm/data/audio_pipeline/finalize_subsets.py --cand_dir "$SUBSETS" --clip_lmdb "$CLIP_LMDB" --clap_lmdb "$CLAP_LMDB" \
      --out_dir "$SUBSETS" --train_anno data/chapters/chapters_train.json --test_anno data/chapters/chapters_test.json
    echo "prep done" ;;
  run)
    detach chain0 env CUDA_VISIBLE_DEVICES=${GPU_A:-0} bash "$0" chain0
    detach chain1 env CUDA_VISIBLE_DEVICES=${GPU_B:-1} bash "$0" chain1 ;;
  chain0)   # GPU A: Stage A -> Stage B (R2, resumes once) -> eval R2 -> eval R4
    MAX_STEPS=$STEPS_A bash $HERE/stage_a.sh && finished "$CKPT/stageA_R2_seed$SEED" || exit 1
    MAX_STEPS=$STEPS_B bash $HERE/stage_b.sh R2; MAX_STEPS=$STEPS_B bash $HERE/stage_b.sh R2
    finished "$CKPT/stageB_R2_seed$SEED" || exit 1
    bash $HERE/eval_row.sh "$CKPT/stageB_R2_seed$SEED" "R2_seed$SEED"
    DROP_AUDIO=1 bash $HERE/eval_row.sh "$CKPT/stageB_R2_seed$SEED" R4 ;;
  chain1)   # GPU B: Stage B R1 (+ extra R1 seeds as a noise baseline) -> eval R1 (+ seeds) -> eval R0 (unless SKIP_R0=1)
    MAX_STEPS=$STEPS_B bash $HERE/stage_b.sh R1; MAX_STEPS=$STEPS_B bash $HERE/stage_b.sh R1
    finished "$CKPT/stageB_R1_seed$SEED" || exit 1
    for s in ${EXTRA_R1_SEEDS:-}; do
      SEED=$s MAX_STEPS=$STEPS_B bash $HERE/stage_b.sh R1; SEED=$s MAX_STEPS=$STEPS_B bash $HERE/stage_b.sh R1
      finished "$CKPT/stageB_R1_seed$s" || exit 1
    done
    AUDIO=0 bash $HERE/eval_row.sh "$CKPT/stageB_R1_seed$SEED" "R1_seed$SEED"
    for s in ${EXTRA_R1_SEEDS:-}; do AUDIO=0 bash $HERE/eval_row.sh "$CKPT/stageB_R1_seed$s" "R1_seed$s"; done
    if [[ "${SKIP_R0:-0}" != 1 ]]; then AUDIO=0 bash $HERE/eval_row.sh "$PUBLIC_STAGE2" R0; fi ;;
  cases)    # qualitative cases: R2 vs R1 with the audio-off control R4 (+ R1_seed1 as noise floor when it exists)
    E=${EVAL_ROOT:-$CKPT/eval}
    B=(); [[ -d "$E/R1_seed1/dense" ]] && B=(--r1b "$E/R1_seed1/dense")
    python revisionllm/eval/qualitative_cases.py --anno "$TEST_ANNO" --r1 "$E/R1_seed$SEED/dense" --r2 "$E/R2_seed$SEED/dense" \
      --r4 "$E/R4/dense" "${B[@]}" --clap_lmdb "$CLAP_LMDB" --out "$E/cases.json" --md "$E/cases.md" ;;
  status)
    for f in "$LOGS"/*.log; do [[ -f $f ]] || continue; echo "== $f"; tr '\r' '\n' < "$f" | grep -a -v -i warn | tail -2 | cut -c1-200; done
    for d in stageA_R2 stageB_R2 stageB_R1; do
      [[ -f "$CKPT/${d}_seed$SEED/non_lora_trainables.bin" ]] && echo "done: $d" || echo "not finished: $d"
    done
    ls "${EVAL_ROOT:-$CKPT/eval}" 2>/dev/null ;;
  *) echo "usage: $0 prep | run | status | cases"; exit 1 ;;
esac
