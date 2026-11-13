"""Run highlight extraction on every *.mp4 in the repo root; write results/highlights/<n>.json (+ highlight mp4)."""
import glob, json, os, re, subprocess, sys, unicodedata
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
os.makedirs("results/highlights", exist_ok=True)
from revisionllm.editing.config import EditConfig
from revisionllm.editing.service import EditingService
from revisionllm.editing.scorers import make_scorer
from revisionllm.multiquery.config import CacheConfig
from revisionllm.editing.db import EditDB

def title_of(p):
    t = unicodedata.normalize("NFC", os.path.splitext(os.path.basename(p))[0])
    t = re.sub(r"^YTDown\.com_YouTube_Media_[\w-]+_", "", t)
    return re.sub(r"[…@＂\"\[\]]", " ", t).strip()

if __name__ == '__main__':
    videos = sorted(glob.glob("*.mp4"))
    cc, cfg = CacheConfig(), EditConfig()
    db = EditDB(cc.dsn, cc.embedding_dim)
    svc = EditingService(cc, cfg, "checkpoints/clip/ViT-L-14.pt", db=db, scorer=make_scorer(), grounder=None, query_gen=None)
    index = []
    for i, v in enumerate(videos, 1):
        title = title_of(v)
        dur = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",v]))
        out = f"results/highlights/video{i}_highlight.mp4"
        plan = svc.highlight(v, out, None, title)
        rec = {"id": i, "file": unicodedata.normalize("NFC", v), "title": title, "duration": dur, "plan": plan}
        json.dump(rec, open(f"results/highlights/video{i}.json", "w"), ensure_ascii=False, indent=1)
        index.append(rec); print(i, title, dur, len(plan["clips"]), plan["saliency_source"], flush=True)
    json.dump(index, open("results/highlights/index.json", "w"), ensure_ascii=False, indent=1)
