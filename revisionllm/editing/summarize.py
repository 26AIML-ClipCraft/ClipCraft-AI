"""Video summarisation (UniVTG's summarisation task): pick the segments that represent the whole video, with time and content."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from revisionllm.editing.scoring import rank_norm


@dataclass
class Segment:
    id: int
    start: float
    end: float
    importance: float
    tags: Dict[str, float] = field(default_factory=dict)       # dominant audio events, class -> mean probability
    key_times: List[float] = field(default_factory=list)       # seconds of the frames that describe the segment
    caption: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _fixed_spans(T: int, win: int) -> List[tuple]:
    """Cut [0, T) into windows of `win` seconds; a leftover shorter than half a window joins the last window."""
    edges = list(range(0, T, win)) + [T]
    if len(edges) > 2 and T - edges[-2] < win / 2:
        edges.pop(-2)
    return list(zip(edges[:-1], edges[1:]))


def summarize(av, importance: np.ndarray, events: np.ndarray, classes, cfg) -> List[Segment]:
    """Cut the video into fixed windows (length grows with the video, see EditConfig.window_sec), then pick `cfg.n_scenes`
    by importance with maximal-marginal-relevance so the picks are different scenes. Returned in time order."""
    T = len(av)
    end_all = min(float(T), av.duration) if av.duration > 0 else float(T)
    spans = _fixed_spans(T, max(int(round(cfg.window_sec(end_all))), 1))
    n_want = max(1, min(cfg.n_scenes, len(spans)))
    x = av.vision if not av.has_audio else np.hstack([av.vision, 0.5 * av.audio])
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    cent, imp = [], []
    for s, e in spans:
        m = x[s:e].mean(0)
        cent.append(m / max(np.linalg.norm(m), 1e-8))
        top = np.sort(importance[s:e])[::-1]
        imp.append(float(top[: max(1, len(top) // 2)].mean()))
    cent, imp = np.array(cent), np.array(imp)
    score = rank_norm(imp) if len(imp) > 1 else np.ones(1)

    picked: List[int] = [int(np.argmax(score))]
    while len(picked) < min(n_want, len(spans)):
        redundancy = (cent @ cent[picked].T).max(1)
        mmr = cfg.mmr_lambda * score - (1 - cfg.mmr_lambda) * redundancy
        mmr[picked] = -np.inf
        picked.append(int(np.argmax(mmr)))

    segs = []
    for n, k in enumerate(sorted(picked)):
        s, e = spans[k]
        p = events[s:e].mean(0)
        tags = {classes[i]: round(float(p[i]), 3) for i in np.argsort(-p)[:2] if p[i] > 0.15}
        peak = s + int(np.argmax(importance[s:e]))
        n_key = int(min(cfg.frames_per_segment, max(2, np.ceil((e - s) / cfg.key_every_sec))))
        keys = [s + (e - s) * (i + 0.5) / n_key for i in range(n_key)] + [peak + 0.5]   # evenly spread + importance peak
        keys = [round(min(t, end_all), 2) for t in keys]
        uniq = []
        for t in keys:                                    # drop a key frame that is (almost) the same moment
            if all(abs(t - u) >= 2.0 for u in uniq):
                uniq.append(t)
        uniq = sorted(uniq)
        if len(uniq) > cfg.frames_per_segment:            # keep evenly spaced ones when over the cap
            idx = np.linspace(0, len(uniq) - 1, cfg.frames_per_segment).round().astype(int)
            uniq = [uniq[i] for i in idx]
        segs.append(Segment(n, float(s), float(min(e, end_all)), round(imp[k], 4), tags, uniq))
    return segs
