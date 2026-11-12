import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EditConfig:
    """Settings of the editing features (scenario recommendation, highlight extraction). Everything that changes a stored
    result is folded into the result key (see service.params_key), so a changed setting never reuses an old result."""
    head_ckpt: Optional[str] = "checkpoints/editing_head/head.pt"   # None / missing file -> training-free fallback saliency
    openai_model: str = field(default_factory=lambda: os.environ.get("CLIPCRAFT_OPENAI_MODEL", "gpt-4o-mini"))
    language: str = "ko"                      # language of captions / scenario text written by the LLM

    # ---- summarisation (scenario recommendation) ----
    n_scenes: int = 8                         # representative segments handed to the LLM
    frames_per_segment: int = 5               # max key frames sent to the captioner per segment
    key_every_sec: float = 8.0                # about one key frame per this many seconds (min 2 per segment)
    frame_max_side: int = 1024                # longest side of a key frame sent to the captioner
    frame_detail: str = "high"                # OpenAI image detail for key frames ("low" | "high")
    mmr_lambda: float = 0.7                   # importance vs. diversity when picking segments

    # ---- highlight ----
    highlight_ratio: float = 0.10             # target length = ratio * duration, clamped to [min, max]
    highlight_min_sec: float = 15.0
    highlight_max_sec: float = 60.0
    clip_min_sec: float = 3.0
    clip_gap_sec: float = 2.0                 # minimum gap between two selected clips

    # ---- one window length per video: the summary's fixed segment length and the highlight's maximum clip length ----
    window_table: tuple = ((360, 12.0), (1200, 20.0), (float("inf"), 30.0))   # (video shorter than this many seconds, window seconds)

    # ---- importance = w_sal * saliency + w_audio * audio events (rank-normalised parts; the weights sum to 1) ----
    w_saliency: float = 0.7
    w_audio: float = 0.3
    smooth_sec: int = 3
    event_weights: Dict[str, float] = field(default_factory=lambda: {
        "laughter": 1.0, "applause": 1.0, "music": 0.6, "speech": 0.5, "ambient": 0.1, "silence": -0.5})
    event_temperature: float = 30.0           # CLAP cosine -> softmax over event classes

    def window_sec(self, duration: float) -> float:
        """Window length in seconds for a video of `duration` seconds: under 6 min 12 s, under 20 min 20 s, longer 30 s
        (`window_table`). The summary cuts the video into windows of this length; the highlight never grows a clip beyond it."""
        return next(w for limit, w in self.window_table if duration < limit)
