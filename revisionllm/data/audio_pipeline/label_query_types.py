"""
Label Test-sub queries as "audio cue present" / "no audio cue" for the H2
breakdown (experiment design doc, 평가와 ablation section).

Two independent labellers; a query enters the breakdown only when both agree:
  1. keyword rule (music, song, sing, talk, interview, speech, laugh, applause,
     clap, cheer, crowd, sound, noise, scream, shout, bell, engine, ...)
  2. Claude (claude-opus-5) yes/no classification with a short rubric.
     Skipped with --no_llm (then the keyword label alone is written and the
     'agree' field is null).

Output: <out>.json  {query_id: {"query": str, "kw": bool, "llm": bool|null, "agree": bool|null}}
"""
import argparse
import json
import os
import re
import sys
import time

AUDIO_KEYWORDS = [
    "music", "song", "sing", "singing", "chorus", "guitar", "piano", "drum", "beat", "melody", "concert",
    "talk", "talking", "speech", "speak", "interview", "conversation", "dialogue", "announce", "narrat",
    "laugh", "laughing", "joke", "applause", "clap", "cheer", "cheering", "crowd", "audience", "chant",
    "sound", "noise", "loud", "quiet", "silence", "scream", "shout", "yell", "cry", "whistle", "bell",
    "engine", "horn", "siren", "bark", "explosion", "gunshot", "thunder", "rain", "fireworks", "podcast",
    "asmr", "voice", "vocal", "rap", "dj", "karaoke", "intro music", "outro", "q&a",
]
KW_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in AUDIO_KEYWORDS) + r")", re.IGNORECASE)

RUBRIC = (
    "You label video-chapter titles for a temporal grounding benchmark. Answer with exactly one word, yes or no.\n"
    "Answer yes if locating this chapter inside the video would plausibly benefit from AUDIO cues "
    "(speech or a specific conversation topic, music/singing, laughter, applause, cheering, crowd noise, "
    "engine/animal/environmental sounds, sound effects). Answer no if the chapter is identifiable from "
    "visuals alone (objects, places, on-screen text, actions without a characteristic sound).\n"
    "Chapter title: "
)


def kw_label(q: str) -> bool:
    return KW_RE.search(q or "") is not None


def llm_label(client, q: str, model: str) -> bool:
    import anthropic
    for attempt in range(5):
        try:
            resp = client.beta.messages.create(
                model=model, max_tokens=256,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": RUBRIC + q.strip()}],
            )
            if resp.stop_reason == "refusal":
                return False
            text = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
            return text.startswith("y")
        except anthropic.RateLimitError as e:
            time.sleep(int(e.response.headers.get("retry-after", "10")))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(2 ** attempt)
            else:
                raise
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_anno", required=True, help="test_sub_anno.json from finalize_subsets.py ({qid: {..., 'sentence'}})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--model", default="claude-opus-5")
    a = ap.parse_args()

    data = json.load(open(a.test_anno))
    items = list(data.items()) if isinstance(data, dict) else [(d.get("query_id", str(i)), d) for i, d in enumerate(data)]
    labels = {}
    if os.path.exists(a.out):
        labels = json.load(open(a.out))
    client = None
    if not a.no_llm:
        import anthropic
        client = anthropic.Anthropic()
    for qid, d in items:
        if qid in labels and (a.no_llm or labels[qid].get("llm") is not None):
            continue
        q = d.get("sentence") or d.get("query") or ""
        if isinstance(q, list):
            q = q[0]
        kw = kw_label(q)
        llm = None if client is None else llm_label(client, q, a.model)
        labels[qid] = {"query": q, "kw": kw, "llm": llm, "agree": (None if llm is None else (kw == llm))}
        if len(labels) % 50 == 0:
            json.dump(labels, open(a.out, "w"))
    json.dump(labels, open(a.out, "w"), indent=1)
    n = len(labels)
    n_kw = sum(v["kw"] for v in labels.values())
    n_agree_audio = sum(1 for v in labels.values() if v["agree"] and v["kw"])
    n_agree_visual = sum(1 for v in labels.values() if v["agree"] and not v["kw"])
    print(f"{n} queries: keyword-audio={n_kw}, agreed audio={n_agree_audio}, agreed visual={n_agree_visual}")
    if min(n_agree_audio, n_agree_visual) < 100 and not a.no_llm:
        print("WARNING: a group has < 100 queries; enlarge Test-sub (design rule)")


if __name__ == "__main__":
    main()
