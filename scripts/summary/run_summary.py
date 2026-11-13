"""Video summary (no LLM, no OpenAI key) on every *.mp4 in the repo root: the representative segments with time, importance,
audio tags and key-frame times. Writes results/summary/<n>.json."""
import json, os, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import make_service, title_of, videos

if __name__ == "__main__":
    os.makedirs("results/summary", exist_ok=True)
    svc = make_service()
    for i, v in enumerate(videos(), 1):
        title = title_of(v)
        segs = svc.summary(v, title)
        rec = {"id": i, "file": unicodedata.normalize("NFC", v), "title": title, "segments": [s.to_dict() for s in segs]}
        json.dump(rec, open(f"results/summary/{i}.json", "w"), ensure_ascii=False, indent=1)
        print(i, title, len(segs), "segments", flush=True)
