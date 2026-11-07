"""One uploaded video, several editing requests, one feature extraction.

  export OPENAI_API_KEY=...   CLIPCRAFT_DB_DSN=postgresql://localhost:5432/clipcraft
  python revisionllm/editing/cli.py recommend --video X.mp4 [--goal "1분짜리 여행 브이로그"]
  python revisionllm/editing/cli.py highlight --video X.mp4 --out X_highlight.mp4 [--seconds 30] [--title "..."]
  python revisionllm/editing/cli.py ground    --video X.mp4 --query "a person laughs"     # loads the 7B model
  python revisionllm/editing/cli.py all       --video X.mp4 --query "a person laughs" --out X_highlight.mp4
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.editing.config import EditConfig
from revisionllm.editing.service import EditingService
from revisionllm.multiquery.cli import add_model_args, make_cfg
from revisionllm.multiquery.service import mmss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ground", "recommend", "highlight", "all"])
    add_model_args(ap)
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--goal", help="optional editing intent given to the LLM")
    ap.add_argument("--seconds", type=float, help="target length of the scenario / highlight")
    ap.add_argument("--out", help="highlight video path (omit to only print the plan)")
    ap.add_argument("--head", default=EditConfig.head_ckpt, help="trained head checkpoint; missing -> training-free saliency")
    ap.add_argument("--scorer", choices=["head", "univtg"], default="head",
                    help="head: the AV head trained here (default); univtg: released UniVTG on the cached CLIP-L/14 features through the "
                         "aligned Adapter (checkpoints/univtg/); needs a title or the Qwen description as its text query")
    ap.add_argument("--model", help="OpenAI model (default $CLIPCRAFT_OPENAI_MODEL or gpt-4o-mini)")
    ap.add_argument("--query_batch", type=int, default=8)
    ap.add_argument("--title", help="video title: the text query of the head. Without it a local Qwen VLM describes the video")
    ap.add_argument("--vlm_model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vlm_python", help="python of an env with transformers >= 4.57 for Qwen3-VL (default $CLIPCRAFT_VLM_PYTHON)")
    a = ap.parse_args()

    cache_cfg, cfg = make_cfg(a), EditConfig(head_ckpt=a.head)
    if a.model:
        cfg.openai_model = a.model
    scorer = None
    if a.scorer == "univtg":
        from revisionllm.editing.univtg.scorer import UniVTGScorer
        scorer = UniVTGScorer()
    elif a.head and os.path.exists(a.head):
        from revisionllm.editing.univtg_head import HeadScorer
        scorer = HeadScorer(a.head)
    else:
        print(f"[editing] no head checkpoint at {a.head}: using training-free saliency (representativeness)")

    from revisionllm.editing.db import EditDB
    db = EditDB(cache_cfg.dsn, cache_cfg.embedding_dim)               # one connection for caches and edit results
    grounder = None
    if a.cmd in ("ground", "all"):
        if not a.query:
            ap.error("ground needs --query")
        from revisionllm.multiquery.grounder import Grounder, model_args
        from revisionllm.multiquery.service import MultiQueryGrounder, model_version_of
        g = Grounder(model_args(a.clip_pt, a.vicuna, a.stage2, a.stage1_proj, a.clip_adapter, cache_cfg), cache_cfg)
        grounder = MultiQueryGrounder(cache_cfg, a.clip_pt, g, model_version_of(a.stage2), db=db, query_batch=a.query_batch)
    from revisionllm.editing.query_gen import QwenQuery
    svc = EditingService(cache_cfg, cfg, a.clip_pt, db=db, scorer=scorer, grounder=grounder,
                         query_gen=QwenQuery(a.vlm_model, a.vlm_python))

    if a.cmd in ("ground", "all"):
        print("== Scenario Grounding ==")
        for r in svc.ground(a.video, a.query):
            print(f"{r.query!r}: {mmss(r.start_time)} ~ {mmss(r.end_time)}  [{r.source}]")
    if a.cmd in ("recommend", "all"):
        print("== Scenario Recommendation ==")
        print(json.dumps(svc.recommend(a.video, a.goal, a.seconds, a.title), ensure_ascii=False, indent=1))
    if a.cmd in ("highlight", "all"):
        print("== Highlight Extraction ==")
        plan = svc.highlight(a.video, a.out, a.seconds, a.title)
        for c in plan["clips"]:
            print(f"{mmss(c['start'])} ~ {mmss(c['end'])}  score={c['score']:.2f}  {c['dominant_audio']}")
        if a.out:
            print(f"saved {a.out}")
    print(f"[editing] encodings this run: {svc.features.n_extractions}")


if __name__ == "__main__":
    main()
