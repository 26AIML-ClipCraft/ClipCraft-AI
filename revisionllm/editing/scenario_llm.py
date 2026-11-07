"""GPT side of scenario recommendation: caption the chosen segments from key frames, then propose an edit order.

The LLM only sees segments that exist in the video (id, time, audio tags, caption); anything it returns that does not refer to
one of them is dropped, so the proposal is always made of real scenes.
"""
import base64
import json
import subprocess
from typing import List, Optional

CAPTION_PROMPT = ("These are frames from one segment of a video ({start} to {end}). Describe in one or two sentences what "
                  "happens in the segment (people, actions, setting). Audio cues detected: {tags}. Write in {language}.")

SCENARIO_SYSTEM = (
    "You are a video editor's assistant. You receive the segments of ONE uploaded video, each with an id, start and end time in "
    "seconds, a caption and detected audio events. Propose an edit: choose which segments to use and in which order, and write the "
    "scenario for the result. Use ONLY the given segment ids and never invent scenes that are not in the list. You may reorder "
    "segments and trim a segment inside its own time range. Reply with JSON only: "
    '{{"title": str, "summary": str, "scenes": [{{"order": int, "segment_id": int, "start": float, "end": float, '
    '"description": str, "reason": str}}]}}. Write text fields in {language}.')


def grab_frame(video_path: str, t: float, max_side: int = 512) -> bytes:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1",
           "-vf", f"scale='min({max_side},iw)':-2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _image_part(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                                               "detail": "low"}}


def make_client():
    from openai import OpenAI            # reads OPENAI_API_KEY
    return OpenAI()


class ScenarioLLM:
    def __init__(self, cfg, client=None):
        self.cfg, self._client = cfg, client
        self.n_calls = 0

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def _chat(self, messages, json_mode=False) -> str:
        self.n_calls += 1
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(model=self.cfg.openai_model, messages=messages, temperature=0.3, **kw)
        return r.choices[0].message.content

    def caption(self, video_path: str, seg) -> str:
        tags = ", ".join(seg.tags) or "none"
        text = CAPTION_PROMPT.format(start=f"{seg.start:.0f}s", end=f"{seg.end:.0f}s", tags=tags, language=self.cfg.language)
        content = [{"type": "text", "text": text}] + [_image_part(grab_frame(video_path, t)) for t in seg.key_times]
        return self._chat([{"role": "user", "content": content}]).strip()

    def recommend(self, segments, duration: float, goal: Optional[str] = None, target_sec: Optional[float] = None) -> dict:
        payload = {"video_duration_sec": round(duration, 1),
                   "segments": [{"id": s.id, "start": s.start, "end": s.end, "caption": s.caption, "audio": s.tags}
                                for s in segments]}
        if goal:
            payload["user_goal"] = goal
        if target_sec:
            payload["target_length_sec"] = target_sec
        msgs = [{"role": "system", "content": SCENARIO_SYSTEM.format(language=self.cfg.language)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for attempt in range(2):
            try:
                return validate_scenario(json.loads(self._chat(msgs, json_mode=True)), segments)
            except (json.JSONDecodeError, ValueError, TypeError):
                if attempt:
                    raise
        raise RuntimeError("unreachable")


def validate_scenario(raw: dict, segments) -> dict:
    """Keep scenes that point at a real segment, clamp their trim to that segment, renumber in the proposed order."""
    by_id = {s.id: s for s in segments}
    scenes = []
    for sc in sorted(raw.get("scenes", []), key=lambda x: x.get("order", 0)):
        seg = by_id.get(sc.get("segment_id"))
        if seg is None:
            continue
        s = min(max(float(sc.get("start", seg.start)), seg.start), seg.end)
        e = min(max(float(sc.get("end", seg.end)), seg.start), seg.end)
        if e - s < 1.0:
            s, e = seg.start, seg.end
        scenes.append({"segment_id": seg.id, "start": s, "end": e, "description": str(sc.get("description", "")),
                       "reason": str(sc.get("reason", ""))})
    if not scenes:
        raise ValueError("the model proposed no scene that exists in the video")
    for n, sc in enumerate(scenes, 1):
        sc["order"] = n
    return {"title": str(raw.get("title", "")), "summary": str(raw.get("summary", "")), "scenes": scenes}
