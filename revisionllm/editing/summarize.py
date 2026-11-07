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


def _boundaries(x: np.ndarray, n_seg: int, min_len: int, win: int) -> List[int]:
    """Change points: seconds where the mean feature of the next `win` seconds differs most from the previous `win`."""
    T = len(x)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    idx = np.arange(1, T)
    lo, hi = np.maximum(idx - win, 0), np.minimum(idx + win, T)
    prev = (cs[idx] - cs[lo]) / (idx - lo)[:, None]
    nxt = (cs[hi] - cs[idx]) / (hi - idx)[:, None]
    cos = (prev * nxt).sum(1) / np.maximum(np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1), 1e-8)
    nov = 1.0 - cos
    chosen: List[int] = []
    for j in np.argsort(-nov):
        t = int(idx[j])
        if t < min_len or T - t < min_len or any(abs(t - c) < min_len for c in chosen):
            continue
        chosen.append(t)
        if len(chosen) >= n_seg - 1:
            break
    return sorted(chosen)


def summarize(av, importance: np.ndarray, events: np.ndarray, classes, cfg) -> List[Segment]:
    """Over-segment at feature change points (2x the wanted count), then pick `cfg.n_scenes` by importance with
    maximal-marginal-relevance so the picks are different scenes. Returned in time order."""
    T = len(av)
    end_all = min(float(T), av.duration) if av.duration > 0 else float(T)
    min_len = max(int(cfg.min_segment_sec), 1)
    n_want = max(1, min(cfg.n_scenes, T // min_len))
    x = av.vision if not av.has_audio else np.hstack([av.vision, 0.5 * av.audio])
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    cuts = _boundaries(x, min(2 * n_want, max(T // min_len, 1)), min_len, win=int(np.clip(T // 100, 3, 15)))
    edges = [0] + cuts + [T]
    spans = list(zip(edges[:-1], edges[1:]))

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
        keys = [(s + e) / 2.0, peak + 0.5]
        keys = [round(min(t, end_all), 2) for t in keys]
        uniq = []
        for t in keys:                                    # drop a key frame that is (almost) the same moment
            if all(abs(t - u) >= 2.0 for u in uniq):
                uniq.append(t)
        segs.append(Segment(n, float(s), float(min(e, end_all)), round(imp[k], 4), tags, uniq[: cfg.frames_per_segment]))
    return segs
