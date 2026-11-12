"""results/highlights/index.json -> results/highlights/data.js (window.CC_DATA), read by 'ClipCraft - Highlights.html'."""
import glob, json, os, subprocess, sys, unicodedata
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
AUDIO_KO = {"speech": "대사 강조", "music": "음악·효과음", "laughter": "웃음 반응", "applause": "환호·박수",
            "ambient": "배경음 변화", "silence": "정적 뒤 장면"}

# 하이라이트 이름: 각 클립의 가운데 프레임을 직접 보고 붙인 이름. 키 = (video id, 클립 시작 초). 없는 클립은 번호 이름으로 대체.
NAMES = {
    (1, 8): "양념 재료 그릇에 넣기", (1, 22): "손으로 양념 비비기", (1, 126): "젓가락으로 양념 버무리기",
    (2, 100): "크림 소스에 담긴 튀김 클로즈업", (2, 244): "주방에서 튀김 튀기기", (2, 258): "접시 위 튀김 한 점 플레이팅",
    (3, 39): "이청아, 술 한 모금 머금는 순간", (3, 79): "숲속에서 시선을 내리는 이청아",
    (4, 128): "여학생에게 말을 거는 남학생", (4, 190): "남학생 앞에서 놀란 여학생의 표정",
}

def label_of(vid, c):
    """Hand-written name for this exact clip (video id, start second); falls back to a numbered label."""
    return NAMES.get((vid, int(round(c["start"]))), f"하이라이트 구간 {int(round(c['start']))}초")

def wave(path, dur, n=240):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-"], capture_output=True).stdout
    x = np.abs(np.frombuffer(raw, np.int16).astype(np.float32)) if raw else np.zeros(n)
    if len(x) < n: return [4.0] * n
    b = np.array([np.sqrt((c ** 2).mean()) for c in np.array_split(x, n)]); b = (b / (np.percentile(b, 99) + 1e-9)).clip(0, 1) ** 0.7
    return [round(3 + float(v) * 62, 1) for v in b]

DISK = {unicodedata.normalize('NFC', f): f for f in glob.glob('*.mp4')}   # names on disk may be NFD
out = []
for r in json.load(open("results/highlights/index.json")):
    clips = r["plan"]["clips"]
    top = max([c["score"] for c in clips] + [1e-9]); lo = min([c["score"] for c in clips] + [0])
    hl = []
    for k, c in enumerate(clips, 1):
        sc = int(round(70 + 29 * (c["score"] - lo) / (top - lo + 1e-9)))     # clip scores re-scaled to 70-99 for display
        th = f"results/highlights/thumbs/v{r['id']}_{k}.jpg"                    # middle frame of the clip
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{(c['start'] + c['end']) / 2:.2f}", "-i", DISK[r["file"]],
                        "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "4", th], check=True)
        hl.append({"id": k, "s": int(c["start"]), "e": int(round(c["end"])), "score": sc, "raw": c["score"],
                   "label": label_of(r["id"], c), "thumb": th, "why": AUDIO_KO.get(c["dominant_audio"], c["dominant_audio"])})
    scenario = None
    rp = f"results/highlights/rec{r['id']}.json"
    if os.path.exists(rp):                       # scenario recommendation (run_recommend.py): scene thumbnails = middle frame
        rec = json.load(open(rp)); scenes = []
        for sc in rec["scenes"]:
            th = f"results/highlights/scene_thumbs/v{r['id']}_{sc['order']}.jpg"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{(sc['start'] + sc['end']) / 2:.2f}", "-i", DISK[r["file"]],
                            "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "4", th], check=True)
            scenes.append({"order": sc["order"], "s": round(sc["start"]), "e": round(sc["end"]), "description": sc["description"],
                           "reason": sc["reason"], "thumb": th})
        scenario = {"title": rec["title"], "summary": rec["summary"], "scenes": scenes}
    out.append({"id": r["id"], "scenario": scenario, "title": r["title"], "duration": round(r["duration"]), "video": DISK[r["file"]],
                "summary": f"results/highlights/video{r['id']}_highlight.mp4", "query": r["plan"].get("query"),
                "saliency_source": r["plan"]["saliency_source"], "highlights": hl, "wave": wave(DISK[r["file"]], r["duration"])})
open("results/highlights/data.js", "w").write("window.CC_DATA = " + json.dumps(out, ensure_ascii=False) + ";\n")
print("videos:", len(out), "clips:", [len(o["highlights"]) for o in out])
