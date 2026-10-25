"""Fig. 4 / Fig. 5 flow: video hash -> cached AV features -> per query: semantic cache -> (miss) hierarchical grounding."""
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.multiquery.config import CacheConfig
from revisionllm.multiquery.db import CacheDB
from revisionllm.multiquery.feature_cache import FeatureCache
from revisionllm.multiquery.grounder import norm_sentence


@dataclass
class QueryResult:
    query: str
    start_time: Optional[float]
    end_time: Optional[float]
    source: str                       # "semantic_cache" | "grounding" | "failed"
    similarity: Optional[float] = None
    matched_query: Optional[str] = None


class TextEncoder:
    """CLIP ViT-L/14 text features: token features feed the model, the EOT feature is also the cache embedding
    (the same two tensors the released query-feature LMDBs hold)."""

    def __init__(self, clip_pt, device=None):
        from revisionllm.data.feature_extraction.clip_extractor import ClipFeatureExtractor
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ext = ClipFeatureExtractor(framerate=30, size=224, centercrop=True, model_name_or_path=clip_pt,
                                        device=device)

    def __call__(self, sentence: str):
        tok, eot = self.ext.encode_text([sentence])
        return tok[0].float().cpu().numpy(), eot[0].float().cpu().numpy()

    @staticmethod
    def embedding(cls_feat: np.ndarray) -> np.ndarray:
        e = cls_feat.astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-12)


class MultiQueryGrounder:
    def __init__(self, cfg: CacheConfig, clip_pt: str, grounder, model_version: str, db: CacheDB = None, text=None,
                 query_batch: int = 8):
        self.cfg, self.grounder, self.model_version = cfg, grounder, model_version
        self.db = db or CacheDB(cfg.dsn, cfg.embedding_dim)
        self.features = FeatureCache(cfg, self.db, clip_pt)
        self.text = text or TextEncoder(clip_pt)
        self.n_grounding_calls = 0
        self.query_batch = query_batch                   # rows (query x window) per forward pass: GPU memory cap

    def run(self, video_path: str, queries: List[str], semantic: bool = True, batched: bool = True) -> List[QueryResult]:
        """semantic=False: no query-cache lookup/store. batched=False: one grounding call per query (evaluation switch)."""
        feats = self.features.get(video_path)            # extraction happens at most once per distinct video file
        print(f"[multiquery] video_id={feats.video_id} features={'cache hit' if feats.cache_hit else 'extracted'} "
              f"duration={feats.duration:.1f}s")
        out = [None] * len(queries)
        misses = []                                      # (index, query, sentence, token feats, cls feats, embedding, hit)
        for n, q in enumerate(queries):
            sentence = norm_sentence(q)
            tok, cls = self.text(sentence)
            emb = TextEncoder.embedding(cls)
            hit = self.db.search_query(feats.video_id, emb, self.model_version) if semantic else None
            if hit is not None and hit[0] > self.cfg.similarity_threshold:
                sim, row = hit
                out[n] = QueryResult(q, row["start_time"], row["end_time"], "semantic_cache", sim, row["query_text"])
            else:
                misses.append((n, q, sentence, tok, cls, emb, hit))
        # Misses that are near-duplicates of an earlier miss of this request follow it instead of being grounded again.
        leaders, followers = [], []
        for m in misses:
            best = max(((float(m[5] @ l[5]), l) for l in leaders), key=lambda x: x[0], default=None)
            if semantic and best is not None and best[0] > self.cfg.similarity_threshold:
                followers.append((m, best[1], best[0]))
            else:
                leaders.append(m)
        if leaders:
            prepared = self.grounder.prepare(feats)      # windows built once, shared by every query
            self.n_grounding_calls += len(leaders)
            items = [(m[2], m[3], m[4]) for m in leaders]
            if batched:
                segs = self.grounder.ground_many(prepared, items, self.query_batch)
            else:
                segs = [self.grounder.ground_many(prepared, [it], self.query_batch)[0] for it in items]
            seg_of = {}
            for m, seg in zip(leaders, segs):
                n, q, _, _, _, emb, hit = m
                seg_of[n] = seg
                if seg is None:
                    out[n] = QueryResult(q, None, None, "failed", hit[0] if hit else None)
                    continue
                if semantic:
                    self.db.put_query(feats.video_id, q, emb, seg[0], seg[1], self.model_version)
                out[n] = QueryResult(q, seg[0], seg[1], "grounding", hit[0] if hit else None,
                                     hit[1]["query_text"] if hit else None)
            for m, lead, sim in followers:
                seg = seg_of[lead[0]]
                out[m[0]] = (QueryResult(m[1], None, None, "failed", sim) if seg is None else
                             QueryResult(m[1], seg[0], seg[1], "semantic_cache", sim, lead[1]))
        return out


def mmss(t):
    return "--:--" if t is None else f"{int(t // 60):02d}:{t % 60:05.2f}"


def model_version_of(stage2: str) -> str:
    """Checkpoint identity for the query cache: directory name + adapter size/mtime."""
    f = os.path.join(stage2, "adapter_model.safetensors")
    st = os.stat(f) if os.path.exists(f) else None
    return f"{os.path.basename(os.path.normpath(stage2))}|{st.st_size if st else 0}|{int(st.st_mtime) if st else 0}"
