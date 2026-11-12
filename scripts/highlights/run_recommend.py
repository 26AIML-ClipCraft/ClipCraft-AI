"""Existing pipeline, unchanged: summary -> per-segment caption (GPT) -> scenario (GPT).  Writes results/highlights/rec{n}.json.
The OpenAI key is read from ~/.openai_key (never printed)."""
import glob, json, os, re, sys, unicodedata
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
key = os.path.expanduser("~/.openai_key")
if "OPENAI_API_KEY" not in os.environ and os.path.exists(key):
    os.environ["OPENAI_API_KEY"] = open(key).read().strip()
from revisionllm.editing.config import EditConfig
from revisionllm.editing.service import EditingService
from revisionllm.editing.scorers import make_scorer
from revisionllm.multiquery.config import CacheConfig
from revisionllm.editing.db import EditDB
from run_all import title_of   # same title rule as the highlight run

cc, cfg = CacheConfig(), EditConfig()
db = EditDB(cc.dsn, cc.embedding_dim)
svc = EditingService(cc, cfg, "checkpoints/clip/ViT-L-14.pt", db=db, scorer=make_scorer(), grounder=None, query_gen=None)
for i, v in enumerate(sorted(glob.glob("*.mp4")), 1):
    out = svc.recommend(v, goal=None, target_sec=None, title=title_of(v))
    json.dump(out, open(f"results/highlights/rec{i}.json", "w"), ensure_ascii=False, indent=1)
    print(i, out["title"], len(out["scenes"]), "scenes; llm calls so far:", svc.llm.n_calls, flush=True)
