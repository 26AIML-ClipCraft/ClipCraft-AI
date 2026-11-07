"""The text query the head needs when the user gives no video title: a one-sentence description written by a local VLM."""
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

import numpy as np

from revisionllm.editing.scenario_llm import grab_frame

HERE = os.path.dirname(os.path.abspath(__file__))


class QwenQuery:
    """Qwen3-VL-8B (default; Qwen2.5-VL also works) from the local HuggingFace cache, run as a subprocess of `python` (an environment with transformers >= 4.57:
    set CLIPCRAFT_VLM_PYTHON or pass python=). The 8B model is loaded per call and freed again, so it does not sit on the
    GPU next to the head; the sentence is cached per video by the service."""

    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", python: Optional[str] = None, n_frames: int = 8,
                 timeout: int = 900):
        self.model, self.n_frames, self.timeout = model, n_frames, timeout
        self.python = python or os.environ.get("CLIPCRAFT_VLM_PYTHON") or sys.executable

    def frame_times(self, duration: float) -> List[float]:
        return [round(float(t), 2) for t in np.linspace(0, max(duration - 1.0, 0.0), self.n_frames + 2)[1:-1]]

    def __call__(self, video_path: str, duration: float) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, t in enumerate(self.frame_times(duration)):
                p = os.path.join(tmp, f"f{i:02d}.jpg")
                with open(p, "wb") as f:
                    f.write(grab_frame(video_path, t, max_side=640))
                paths.append(p)
            cmd = [self.python, os.path.join(HERE, "qwen_describe.py"), "--model", self.model, "--frames", *paths]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"VLM query generation failed ({self.python}):\n{r.stderr[-1500:]}")
        try:
            query = json.loads(r.stdout.strip().splitlines()[-1])["query"].strip()
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"unreadable VLM output: {r.stdout[-500:]!r}") from e
        if not query:
            raise RuntimeError("the VLM returned an empty description")
        return query
