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
import math
import os
import shutil
import subprocess
import sys
import time
import numpy as np
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
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", "ignore")
            if "not a bot" in err or "Sign in to confirm" in err:
                return BLOCKED     # the IP is blocked: this says nothing about the video, do not record a failure
            time.sleep(5)
        except Exception:
            time.sleep(5)
    return None


BLOCKED = "__YOUTUBE_BLOCKED__"


def local_media(vid, media_dir):
    """--media_dir mode: the videos are already on disk (<media_dir>/<id>.mp4|mkv|webm), nothing is downloaded."""
    for ext in ("mp4", "mkv", "webm", "avi"):
        p = os.path.join(media_dir, f"{vid}.{ext}")
        if os.path.exists(p):
            return p
    return None


def has_audio_stream(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def media_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


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
    ap.add_argument("--target", type=int, default=0,
                    help="stop starting new downloads once this many ids of --ids are stored (Val/Test-sub n_final); "
                         "ids are taken in list order, so failed downloads are replaced by the next candidates. 0 = all")
    ap.add_argument("--limit_ids", type=int, default=0,
                    help="use only the first N ids of --ids (applied before --split); val/test: n_final + spares")
    ap.add_argument("--also_done_clip", default=None, help="extra (read-only) CLIP LMDB whose ids are skipped")
    ap.add_argument("--also_done_clap", default=None, help="extra (read-only) CLAP LMDB whose ids are skipped")
    ap.add_argument("--keep_dir", default=None, help="where --keep_media audio goes (default <tmp_dir>/keep)")
    ap.add_argument("--clip_batch", type=int, default=128, help="CLIP frames per GPU batch")
    ap.add_argument("--ffmpeg_threads", type=int, default=int(os.environ.get("FFMPEG_THREADS", "3")),
                    help="ffmpeg decoder threads per worker (several workers share the CPU)")
    ap.add_argument("--interleave", action="store_true",
                    help="shuffle the id list (fixed seed) and give worker i every total_split-th id, instead of a contiguous "
                         "slice. Needed for Train-sub: its list is sorted by video length, so contiguous slices leave one worker "
                         "with all the multi-hour videos. Also makes any partial run a uniform sample of the list.")
    ap.add_argument("--media_dir", default=None,
                    help="videos already on disk (<id>.mp4): no download, and the files are NOT deleted afterwards")
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop cleanly after this many hours (job limit)")
    a = ap.parse_args()

    os.makedirs(a.tmp_dir, exist_ok=True)
    keep_dir = a.keep_dir or os.path.join(a.tmp_dir, "keep")
    if a.keep_media == "audio":
        os.makedirs(keep_dir, exist_ok=True)

    ids = load_ids(a.ids)
    if a.limit_ids:
        ids = ids[:a.limit_ids]
    n = len(ids)
    if a.interleave:
        import random
        ids = list(ids)
        random.Random(0).shuffle(ids)
        ids = ids[a.split::a.total_split]
    else:
        lo, hi = a.split * n // a.total_split, (a.split + 1) * n // a.total_split
        ids = ids[lo:hi]
    done_clip, done_clap = lmdb_keys(a.clip_lmdb), lmdb_keys(a.clap_lmdb)
    if a.also_done_clip:
        done_clip |= lmdb_keys(a.also_done_clip)
    if a.also_done_clap:
        done_clap |= lmdb_keys(a.also_done_clap)
    ok_ids = {v for v in ids if v in done_clip and v in done_clap}
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
    if a.target:
        print(f"target {a.target}: {len(ok_ids)} already stored")
    print(f"split {a.split}/{a.total_split}: {len(ids)} ids, {len(todo)} to do "
          f"({len(done_clip)} clip / {len(done_clap)} clap already stored, {len(failed)} known download failures)")

    from revisionllm.data.audio_pipeline.extract_clip import ClipFrameExtractor
    from revisionllm.data.audio_pipeline.extract_clap import ClapAudioExtractor
    clip_ex = ClipFrameExtractor(a.clip_path, fps=a.fps, batch_size=a.clip_batch, ffmpeg_threads=a.ffmpeg_threads)
    clap_ex = ClapAudioExtractor(a.clap_model, window_sec=a.hop_sec, hop_sec=a.hop_sec)
    clip_db, clap_db = LmdbWriter(a.clip_lmdb), LmdbWriter(a.clap_lmdb)
    deadline = time.time() + a.max_hours * 3600 if a.max_hours > 0 else None
    blocked = False

    def log(vid, status, **extra):
        with open(a.log, "a") as f:
            f.write(json.dumps({"vid": vid, "status": status, "t": time.time(), **extra}) + "\n")

    with ThreadPoolExecutor(max_workers=a.download_workers) as pool:
        futures = {}
        it = iter(todo)

        def submit_next():
            if a.target and len(ok_ids) + len(futures) >= a.target:
                return False
            try:
                v = next(it)
            except StopIteration:
                return False
            futures[pool.submit(local_media, v, a.media_dir) if a.media_dir else pool.submit(ytdlp, v, a.tmp_dir)] = v
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
            if path == BLOCKED:
                print("[download_and_extract] YouTube is blocking this IP ('Sign in to confirm you're not a bot'). "
                      "Stopping WITHOUT recording failures; retry later or pass cookies.", flush=True)
                for fut in futures:
                    fut.cancel()
                blocked = True
                break
            submit_next()
            if path is None:
                log(vid, "download_failed")
                continue
            try:
                if vid not in done_clip:
                    clip_db.put(vid, clip_ex(path))
                no_audio = False
                if vid not in done_clap:
                    try:
                        clap_db.put(vid, clap_ex(path))
                    except Exception:
                        if a.media_dir and not has_audio_stream(path):
                            # silent video: store zero features of the right length so that the audio branch sees "no sound"
                            n_sec = max(1, int(math.ceil(media_duration(path) / a.hop_sec)))
                            clap_db.put(vid, np.zeros((n_sec, clap_ex.dim), dtype=np.float16))
                            no_audio = True
                        else:
                            raise
                ok_ids.add(vid)
                log(vid, "ok", media=os.path.basename(path), **({"no_audio": True} if no_audio else {}))
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
                if os.path.exists(path) and not a.media_dir:
                    os.remove(path)
    clip_db.close()
    clap_db.close()
    if blocked:
        sys.exit(3)


if __name__ == "__main__":
    main()
