"""Which scorer produces the saliency curve of the editing module. Default: the UniVTG + Adapter scorer.

    univtg (default)  released UniVTG (CLIP-B/32 checkpoint, frozen) on the cached CLIP-L/14 features through the aligned Linear
                      Adapter 768->512 (checkpoints/univtg/). Query = video title text. See checkpoints/univtg/README.md.
    head              the audio-visual UniVTG-style head trained in this repository (univtg_head.py, checkpoints/editing_head/head.pt)

Select with make_scorer("head") or the environment variable CLIPCRAFT_SCORER=head; the CLI flag is --scorer. Scores of the two
scorers are cached under different keys (scorer.version), so switching never reuses the other one's results.
"""
import os
from typing import Optional

DEFAULT_SCORER = "univtg"
SCORERS = ("univtg", "head")


def make_scorer(kind: Optional[str] = None, head_ckpt: Optional[str] = None):
    kind = (kind or os.environ.get("CLIPCRAFT_SCORER") or DEFAULT_SCORER).lower()
    if kind not in SCORERS:
        raise ValueError(f"unknown scorer {kind!r}; choose one of {SCORERS}")
    if kind == "univtg":
        from revisionllm.editing.univtg.scorer import UniVTGScorer
        return UniVTGScorer()
    from revisionllm.editing.config import EditConfig
    from revisionllm.editing.legacy.univtg_head import HeadScorer
    ckpt = head_ckpt or EditConfig.head_ckpt
    if not ckpt or not os.path.exists(ckpt):
        print(f"[editing] no head checkpoint at {ckpt}: using training-free saliency (representativeness)")
        return None
    return HeadScorer(ckpt)
