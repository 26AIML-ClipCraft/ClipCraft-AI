#!/bin/bash
# recommended evaluation on gpu4 GPU3: 300 random Charades-STA test videos (seed 0), query_batch 4, VRAM sampled
cd /home/minyoy/ClipCraft-AI
OUT=results/multiquery_eval/final
CH=data/charades
PY=/home/minyoy/miniconda3/envs/revaudio/bin/python
$PY - <<PY2
import json, random
ids = json.load(open("$CH/charades_test_ids.json")); ids = sorted(ids); random.Random(0).shuffle(ids)
sel = ids[:300]; json.dump(sel, open("$CH/eval_ids_300.json", "w"))
open("$CH/eval_files_300.txt", "w").write("\n".join("Charades_v1_480/%s.mp4" % i for i in sel))
PY2
( cd $CH && unzip -q -o Charades_v1_480.zip $(cat eval_files_300.txt | tr '\n' ' ') ) || { echo "unzip failed" > $OUT/FAILED; exit 1; }
echo "videos: $(ls $CH/Charades_v1_480 | wc -l)" > $OUT/setup.log
P=/home/minyoy/pgenv/bin
$P/pg_ctl -D /home/minyoy/pgdata -o "-p 5433 -k /tmp" -l /tmp/pg_gpu4.log start >/dev/null; sleep 4
( while true; do nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits >> $OUT/vram.log; sleep 2; done ) &
SAMPLER=$!
export CLIPCRAFT_DB_DSN=postgresql://localhost:5433/clipcraft PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=3 $PY revisionllm/multiquery/evaluate.py all --query_batch 4 --max_videos 300 \
  --out $OUT --eval_cache_dir /tmp/multiquery_eval_cache_gpu3 > $OUT/eval.log 2>&1
echo "exit $?" >> $OUT/eval.log
kill $SAMPLER; $P/pg_ctl -D /home/minyoy/pgdata stop >/dev/null
echo done > $OUT/DONE
