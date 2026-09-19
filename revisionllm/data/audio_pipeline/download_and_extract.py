"""
Rolling download -> feature extraction -> delete pipeline (disk budget 900 GB).

For every video id in --ids (a *_candidates.json or a plain JSON list):
  1. skip if the id is already in BOTH the CLIP and the CLAP LMDB   (resumable)
  2. yt-dlp: 360p video + best audio into --tmp_dir            (thread pool)
  3. CLIP ViT-L/14 @ 2 fps  -> clip LMDB                          (GPU, main thread)
  4. CLAP 1 s hop           -> clap LMDB                          (GPU, main thread)
  5. delete the media file (unless --keep_media, used for Test-sub audio backup
     where only the .m4a is kept)
Progress is appended to --log (one JSON per video: ok / download_failed /
extract_failed) so a killed job can be re-submitted and continues.

Split into N jobs with --split i --total_split N (each job takes a disjoint
slice of the id list) to respect the 1-day job limit.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(ROOT)

from revisionllm.data.audio_pipeline.lmdb_utils import LmdbWriter, lmdb_keys


def load_ids(path):
    obj = json.load(open(path))
    return obj["ids"] if isinstance(obj, dict) else obj


def ytdlp(vid, tmp_dir, audio_only=False, retries=2):
    """Download one YouTube id. Returns the local file path or None."""
    url = f"https://www.youtube.com/watch?v={vid}"
    out_tmpl = os.path.join(tmp_dir, f"{vid}.%(ext)s")
    if audio_only:
        fmt = ["-f", "bestaudio[ext=m4a]/bestaudio", "-x", "--audio-format", "m4a"]
    else:
        # 360p is enough for 224-px CLIP; keep the best audio stream for CLAP.
        fmt = ["-f", "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]/best",
               "--merge-output-format", "mp4"]
    cmd = ["yt-dlp", "--no-playlist", "--quiet", "--no-warnings", "--retries", "3",
           "--socket-timeout", "30", "-o", out_tmpl] + fmt + [url]
    for _ in range(retries + 1):
        try:
            subprocess.run(cmd, check=True, timeout=1800, capture_output=True)
            for f in os.listdir(tmp_dir):
                if f.startswith(vid + "."):
                    return os.path.join(tmp_dir, f)
        except Exception:
            time.sleep(5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--clip_lmdb", required=True)
    ap.add_argument("--clap_lmdb", required=True)
    ap.add_argument("--tmp_dir", required=True, help="rolling media buffer (<=100 GB)")
    ap.add_argument("--log", required=True, help="jsonl progress log (append)")
    ap.add_argument("--clip_path", default="checkpoints/clip/ViT-L-14.pt")
    ap.add_argument("--clap_model", default="laion/clap-htsat-fused")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)
    ap.add_argument("--download_workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=8, help="max downloaded-but-unprocessed videos in tmp_dir")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--total_split", type=int, default=1)
    ap.add_argument("--keep_media", choices=["none", "audio"], default="none",
                    help="'audio' keeps <tmp_dir>/keep/<id>.m4a (Test-sub audio backup)")
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop cleanly after this many hours (job limit)")
    a = ap.parse_args()

    os.makedirs(a.tmp_dir, exist_ok=True)
    keep_dir = os.path.join(a.tmp_dir, "keep")
    if a.keep_media == "audio":
        os.makedirs(keep_dir, exist_ok=True)

    ids = load_ids(a.ids)
    n = len(ids)
    lo, hi = a.split * n // a.total_split, (a.split + 1) * n // a.total_split
    ids = ids[lo:hi]
    done_clip, done_clap = lmdb_keys(a.clip_lmdb), lmdb_keys(a.clap_lmdb)
    failed = set()
    if os.path.exists(a.log):
        for line in open(a.log):
            try:
                rec = json.loads(line)
                if rec.get("status") == "download_failed":
                    failed.add(rec["vid"])
            except Exception:
                pass
    todo = [v for v in ids if not (v in done_clip and v in done_clap) and v not in failed]
    print(f"split {a.split}/{a.total_split}: {len(ids)} ids, {len(todo)} to do "
          f"({len(done_clip)} clip / {len(done_clap)} clap already stored, {len(failed)} known download failures)")

    from revisionllm.data.audio_pipeline.extract_clip import ClipFrameExtractor
    from revisionllm.data.audio_pipeline.extract_clap import ClapAudioExtractor
    clip_ex = ClipFrameExtractor(a.clip_path, fps=a.fps)
    clap_ex = ClapAudioExtractor(a.clap_model, window_sec=a.hop_sec, hop_sec=a.hop_sec)
    clip_db, clap_db = LmdbWriter(a.clip_lmdb), LmdbWriter(a.clap_lmdb)
    deadline = time.time() + a.max_hours * 3600 if a.max_hours > 0 else None

    def log(vid, status, **extra):
        with open(a.log, "a") as f:
            f.write(json.dumps({"vid": vid, "status": status, "t": time.time(), **extra}) + "\n")

    with ThreadPoolExecutor(max_workers=a.download_workers) as pool:
        futures = {}
        it = iter(todo)

        def submit_next():
            try:
                v = next(it)
            except StopIteration:
                return False
            futures[pool.submit(ytdlp, v, a.tmp_dir)] = v
            return True

        for _ in range(a.prefetch):
            if not submit_next():
                break
        while futures:
            if deadline and time.time() > deadline:
                print("[download_and_extract] time limit reached; re-run to resume.")
                for fut in futures:
                    fut.cancel()
                break
            done_fut = next(iter([f for f in list(futures) if f.done()]), None)
            if done_fut is None:
                time.sleep(1.0)
                continue
            vid = futures.pop(done_fut)
            path = done_fut.result()
            submit_next()
            if path is None:
                log(vid, "download_failed")
                continue
            try:
                if vid not in done_clip:
                    clip_db.put(vid, clip_ex(path))
                if vid not in done_clap:
                    clap_db.put(vid, clap_ex(path))
                log(vid, "ok", media=os.path.basename(path))
            except Exception as e:
                log(vid, "extract_failed", error=str(e)[:300])
            finally:
                if a.keep_media == "audio":
                    try:
                        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-vn",
                                        "-acodec", "copy", os.path.join(keep_dir, f"{vid}.m4a")],
                                       check=True, timeout=600)
                    except Exception:
                        pass
                if os.path.exists(path):
                    os.remove(path)
    clip_db.close()
    clap_db.close()


if __name__ == "__main__":
    main()
