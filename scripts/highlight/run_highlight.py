"""Highlight extraction on every *.mp4 in the repo root. Writes results/highlight/<n>.json and <n>_highlight.mp4."""
import json, os, subprocess, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import make_service, title_of, videos

if __name__ == "__main__":
    os.makedirs("results/highlight", exist_ok=True)
    svc = make_service()
    for i, v in enumerate(videos(), 1):
        title = title_of(v)
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v]))
        plan = svc.highlight(v, f"results/highlight/{i}_highlight.mp4", None, title)
        rec = {"id": i, "file": unicodedata.normalize("NFC", v), "title": title, "duration": dur, "plan": plan}
        json.dump(rec, open(f"results/highlight/{i}.json", "w"), ensure_ascii=False, indent=1)
        print(i, title, round(dur), "s;", len(plan["clips"]), "clips;", plan["saliency_source"], flush=True)
