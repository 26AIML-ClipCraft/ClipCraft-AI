"""Per-second importance of a video: the one curve both the summary and the highlight selection read."""
import numpy as np


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1]: comparable across components and robust to outliers."""
    if len(x) < 2:
        return np.zeros(len(x))
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r / (len(x) - 1)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def visual_change(vision: np.ndarray) -> np.ndarray:
    """1 - cosine to the previous second (rows are unit vectors): large at cuts and fast motion."""
    d = np.zeros(len(vision))
    d[1:] = 1.0 - (vision[1:] * vision[:-1]).sum(1)
    return d


def representativeness(vision: np.ndarray) -> np.ndarray:
    """Cosine to the video's mean feature: the training-free stand-in for saliency when no head is available."""
    m = vision.mean(0)
    return vision @ (m / max(np.linalg.norm(m), 1e-8))


def importance_curve(av, events: np.ndarray, classes, cfg, saliency: np.ndarray = None):
    """-> (importance (T,) in [0,1], parts). Components are rank-normalised, smoothed and mixed with cfg weights (summing to 1).
    saliency = the head's per-second score (s + f; only its ranks matter); None uses representativeness; a video without audio uses the saliency alone."""
    sal_src = "head:s+f" if saliency is not None else "representativeness"
    sal = saliency if saliency is not None else representativeness(av.vision)
    w = (cfg.w_saliency, cfg.w_audio) if av.has_audio else (1.0, 0.0)   # no audio track: saliency alone
    ev_w = np.array([cfg.event_weights.get(c, 0.0) for c in classes])
    parts = {"saliency": rank_norm(smooth(sal, cfg.smooth_sec)),
             "audio": rank_norm(smooth(events @ ev_w, cfg.smooth_sec))}
    imp = w[0] * parts["saliency"] + w[1] * parts["audio"]
    return imp, {**parts, "saliency_source": sal_src}
