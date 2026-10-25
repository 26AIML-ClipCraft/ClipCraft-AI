#!/bin/bash
# pilot on gpu4 GPU3: 3 local Charades videos, query_batch 4, VRAM sampled every 2 s
cd /home/minyoy/ClipCraft-AI
OUT=results/multiquery_eval/gpu3_pilot
P=/home/minyoy/pgenv/bin
$P/pg_ctl -D /home/minyoy/pgdata -o "-p 5433 -k /tmp" -l /tmp/pg_gpu4.log start >/dev/null; sleep 4
( while true; do nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits >> $OUT/vram.log; sleep 2; done ) &
SAMPLER=$!
export CLIPCRAFT_DB_DSN=postgresql://localhost:5433/clipcraft PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=3 /home/minyoy/miniconda3/envs/revaudio/bin/python revisionllm/multiquery/evaluate.py all \
  --query_batch 4 --scaling_ns 1 5 10 20 --scaling_videos 1 --repeats 1 \
  --out $OUT --eval_cache_dir /tmp/multiquery_eval_cache_gpu3 > $OUT/eval.log 2>&1
echo "exit $?" >> $OUT/eval.log
kill $SAMPLER; $P/pg_ctl -D /home/minyoy/pgdata stop >/dev/null
echo done > $OUT/DONE
