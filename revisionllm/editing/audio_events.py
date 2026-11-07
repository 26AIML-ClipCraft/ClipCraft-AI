"""Zero-shot audio events from the cached CLAP embeddings (no audio is decoded again).

CLAP's audio embeddings share a space with its text encoder, so each second is scored against a few prompts per class and the
class scores are softmaxed. Only the (tiny) prompt embeddings need the CLAP text tower; they are cached next to the features.
"""
import hashlib
import json
import os
from typing import Dict, List

import numpy as np

EVENT_PROMPTS: Dict[str, List[str]] = {
    "speech": ["a person talking", "people having a conversation", "a man speaking", "a woman speaking"],
    "music": ["music playing", "a song with instruments and singing", "background music"],
    "laughter": ["people laughing", "a laugh", "audience laughter"],
    "applause": ["audience applauding", "clapping hands", "crowd cheering and applause"],
    "ambient": ["ambient environmental noise", "wind, traffic and background sounds", "water flowing, birds and nature sounds"],
    "silence": ["silence", "quiet room tone"],
}


def prompt_embeddings(prompts=EVENT_PROMPTS, clap_name="laion/clap-htsat-fused", cache_dir=None, device=None) -> np.ndarray:
    """-> (n_classes, 512) mean-of-prompts unit vectors, class order = dict order."""
    tag = hashlib.sha1(json.dumps([prompts, clap_name], sort_keys=False).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"clap_event_text_{tag}.npy") if cache_dir else None
    if path and os.path.exists(path):
        return np.load(path)
    import torch
    from revisionllm.eval.clap_zeroshot import clap_text_embeddings
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    flat = [p for ps in prompts.values() for p in ps]
    emb = clap_text_embeddings(flat, clap_name, device)
    out, i = [], 0
    for ps in prompts.values():
        m = emb[i:i + len(ps)].mean(0)
        out.append(m / max(np.linalg.norm(m), 1e-8))
        i += len(ps)
    out = np.stack(out).astype(np.float32)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, out)
    return out


class AudioEventTagger:
    def __init__(self, text_emb: np.ndarray, classes=tuple(EVENT_PROMPTS), temperature: float = 30.0):
        self.text, self.classes, self.temperature = text_emb, list(classes), temperature

    def probs(self, av) -> np.ndarray:
        """(T, n_classes) class probabilities per second. A video without audio is all 'silence'."""
        T = len(av)
        if not av.has_audio:
            p = np.zeros((T, len(self.classes)), np.float32)
            if "silence" in self.classes:
                p[:, self.classes.index("silence")] = 1.0
            return p
        z = self.temperature * (av.audio @ self.text.T)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(1, keepdims=True)).astype(np.float32)
