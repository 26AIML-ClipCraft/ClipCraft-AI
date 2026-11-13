"""One uploaded video, several editing requests, one feature extraction.

  export OPENAI_API_KEY=...   CLIPCRAFT_DB_DSN=postgresql://localhost:5432/clipcraft
  python revisionllm/editing/cli.py summary   --video X.mp4 [--title "..."]                # representative segments, no LLM
  python revisionllm/editing/cli.py scenario  --video X.mp4 [--goal "1분짜리 여행 브이로그"]   # summary -> captions -> scenario (GPT)
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
    ap.add_argument("cmd", choices=["ground", "summary", "scenario", "recommend", "highlight", "all"])
    add_model_args(ap)
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--goal", help="optional editing intent given to the LLM")
    ap.add_argument("--seconds", type=float, help="target length of the scenario / highlight")
    ap.add_argument("--out", help="highlight video path (omit to only print the plan)")
    ap.add_argument("--model", help="OpenAI model (default $CLIPCRAFT_OPENAI_MODEL or gpt-4o-mini)")
    ap.add_argument("--query_batch", type=int, default=8)
    ap.add_argument("--title", help="video title: the text query of UniVTG. Without it a local Qwen VLM describes the video")
    ap.add_argument("--vlm_model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vlm_python", help="python of an env with transformers >= 4.57 for Qwen3-VL (default $CLIPCRAFT_VLM_PYTHON)")
    a = ap.parse_args()

    cache_cfg, cfg = make_cfg(a), EditConfig()
    if a.model:
        cfg.openai_model = a.model
    from revisionllm.editing.scorers import make_scorer
    scorer = make_scorer()
    print(f"[editing] scorer: {type(scorer).__name__}")

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
    if a.cmd == "summary":
        print("== Video Summary ==")
        for sg in svc.summary(a.video, a.title):
            print(f"{mmss(sg.start)} ~ {mmss(sg.end)}  importance={sg.importance:.2f}  tags={sg.tags}")
    if a.cmd in ("scenario", "recommend", "all"):         # recommend: the old name of scenario
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
