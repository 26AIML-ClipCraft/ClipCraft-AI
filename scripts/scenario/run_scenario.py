"""Scenario recommendation on every *.mp4 in the repo root: summary segments (computed or cached) -> per-segment captions (GPT)
-> scenario (GPT). Needs OPENAI_API_KEY (or ~/.openai_key, never printed). Writes results/scenario/<n>.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import load_openai_key, make_service, title_of, videos

if __name__ == "__main__":
    os.makedirs("results/scenario", exist_ok=True)
    load_openai_key()
    svc = make_service()
    for i, v in enumerate(videos(), 1):
        out = svc.scenario(v, svc.summary(v, title_of(v)), goal=None, target_sec=None)
        json.dump(out, open(f"results/scenario/{i}.json", "w"), ensure_ascii=False, indent=1)
        print(i, out["title"], len(out["scenes"]), "scenes; llm calls so far:", svc.llm.n_calls, flush=True)
