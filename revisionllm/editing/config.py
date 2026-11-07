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
    min_segment_sec: float = 5.0
    frames_per_segment: int = 2               # key frames sent to the captioner per segment
    mmr_lambda: float = 0.7                   # importance vs. diversity when picking segments

    # ---- highlight ----
    highlight_ratio: float = 0.10             # target length = ratio * duration, clamped to [min, max]
    highlight_min_sec: float = 15.0
    highlight_max_sec: float = 60.0
    clip_min_sec: float = 3.0
    clip_max_sec: float = 12.0
    clip_gap_sec: float = 2.0                 # minimum gap between two selected clips

    # ---- importance = w_sal * saliency + w_audio * audio events + w_motion * visual change (rank-normalised parts) ----
    w_saliency: float = 0.5
    w_audio: float = 0.3
    w_motion: float = 0.2
    smooth_sec: int = 3
    event_weights: Dict[str, float] = field(default_factory=lambda: {
        "laughter": 1.0, "applause": 1.0, "music": 0.6, "speech": 0.5, "ambient": 0.1, "silence": -0.5})
    event_temperature: float = 30.0           # CLAP cosine -> softmax over event classes
