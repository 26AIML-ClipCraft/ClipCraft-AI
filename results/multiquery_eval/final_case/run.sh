#!/bin/bash
# re-run only the Semantic-Cache case (with frames) on gpu4 GPU3
cd /home/minyoy/ClipCraft-AI
OUT=results/multiquery_eval/final_case
P=/home/minyoy/pgenv/bin
$P/pg_ctl -D /home/minyoy/pgdata -o "-p 5433 -k /tmp" -l /tmp/pg_gpu4.log start >/dev/null; sleep 4
export CLIPCRAFT_DB_DSN=postgresql://localhost:5433/clipcraft PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=3 /home/minyoy/miniconda3/envs/revaudio/bin/python revisionllm/multiquery/evaluate.py case --query_batch 4 --max_videos 300 \
  --out $OUT --eval_cache_dir /tmp/multiquery_eval_cache_gpu3 > $OUT/eval.log 2>&1
echo "exit $?" >> $OUT/eval.log
$P/pg_ctl -D /home/minyoy/pgdata stop >/dev/null
echo done > $OUT/DONE
