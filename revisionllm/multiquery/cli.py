"""Multi-query grounding for one video with the R2 (audio-visual) checkpoint.

  python revisionllm/multiquery/cli.py --video data/charades/Charades_v1_480/3MSZA.mp4 \
      --query "person turn a light on" --query "person opens a door"
  python revisionllm/multiquery/cli.py --video X.mp4 --queries_file queries.txt     # one query per line
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.multiquery.config import CacheConfig
from revisionllm.multiquery.grounder import Grounder, model_args
from revisionllm.multiquery.service import MultiQueryGrounder, mmss, model_version_of


def add_model_args(ap):
    ap.add_argument("--stage2", default="checkpoints/audio_charades_long/stageB_R2_seed0", help="R2 LoRA dir")
    ap.add_argument("--vicuna", default="checkpoints/vicuna-7b-v1.5")
    ap.add_argument("--clip_pt", default="checkpoints/clip/ViT-L-14.pt")
    ap.add_argument("--stage1_proj", default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    ap.add_argument("--clip_adapter", default="checkpoints/public/chapters_stage1_sparse/non_lora_trainables.bin")
    ap.add_argument("--dsn", default=None, help="PostgreSQL DSN (default: $CLIPCRAFT_DB_DSN)")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="semantic cache cosine threshold")


def make_cfg(a):
    cfg = CacheConfig()
    if a.dsn: cfg.dsn = a.dsn
    if a.cache_dir: cfg.cache_dir = a.cache_dir
    if a.threshold is not None: cfg.similarity_threshold = a.threshold
    return cfg


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--queries_file")
    ap.add_argument("--query_batch", type=int, default=8, help="max (query x window) rows per forward pass")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    queries = list(a.query)
    if a.queries_file:
        queries += [l.strip() for l in open(a.queries_file) if l.strip()]
    if not queries:
        ap.error("give --query and/or --queries_file")
    cfg = make_cfg(a)
    g = Grounder(model_args(a.clip_pt, a.vicuna, a.stage2, a.stage1_proj, a.clip_adapter, cfg), cfg)
    svc = MultiQueryGrounder(cfg, a.clip_pt, g, model_version_of(a.stage2), query_batch=a.query_batch)
    res = svc.run(a.video, queries)
    if a.json:
        print(json.dumps([r.__dict__ for r in res], ensure_ascii=False, indent=1))
    for r in res:
        extra = f" (sim={r.similarity:.3f} to '{r.matched_query}')" if r.source == "semantic_cache" else ""
        print(f"{r.query!r}: {mmss(r.start_time)} ~ {mmss(r.end_time)}  [{r.source}]{extra}")


if __name__ == "__main__":
    main()
