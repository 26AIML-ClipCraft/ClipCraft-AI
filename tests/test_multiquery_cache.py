"""Cache behaviour of revisionllm.multiquery without a GPU or the 7B model (fake extractor / text encoder / grounder).
Needs a PostgreSQL with pgvector: CLIPCRAFT_TEST_DSN=postgresql://localhost:5433/clipcraft pytest tests/test_multiquery_cache.py"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DSN = os.environ.get("CLIPCRAFT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="CLIPCRAFT_TEST_DSN not set")

from revisionllm.multiquery import feature_cache as fc
from revisionllm.multiquery.config import CacheConfig
from revisionllm.multiquery.db import CacheDB
from revisionllm.multiquery.service import MultiQueryGrounder


class FakeText:
    VEC = {"a person opens the door": [1, 0, 0], "person opening a door": [0.99, 0.1, 0], "person sneezes": [0, 1, 0]}

    def __call__(self, s):
        v = np.zeros(768, np.float32); v[:3] = self.VEC[s]
        return np.zeros((4, 768), np.float32), v


class FakeGrounder:
    def __init__(self): self.calls = []
    def prepare(self, feats): return feats
    def ground_many(self, pv, queries, batch):
        out = []
        for sentence, tok, cls in queries:
            self.calls.append(sentence)
            out.append((1.0 + len(self.calls), 5.0 + len(self.calls)))
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = CacheConfig(dsn=DSN, cache_dir=str(tmp_path / "cache"), similarity_threshold=0.95)
    db = CacheDB(DSN)
    db.conn.execute("TRUNCATE video, query_cache RESTART IDENTITY CASCADE")
    cache = fc.FeatureCache
    monkeypatch.setattr(cache, "_extract", lambda self, p: (np.ones((20, 768), np.float16), np.ones((10, 512), np.float16)))
    monkeypatch.setattr(fc, "probe_duration", lambda p: 10.0)
    g = FakeGrounder()
    svc = MultiQueryGrounder(cfg, "unused", g, "m1", db=db, text=FakeText())
    v = tmp_path / "v.mp4"; v.write_bytes(b"video-bytes")
    return svc, g, str(v), tmp_path


def test_feature_reuse_and_semantic_cache(env):
    svc, g, v, _ = env
    r = svc.run(v, ["a person opens the door", "person sneezes"])
    assert [x.source for x in r] == ["grounding", "grounding"] and svc.features.n_extractions == 1
    # same video again: no extraction; similar query: no grounding; dissimilar-enough query: grounding
    r = svc.run(v, ["person opening a door", "a person opens the door"])
    assert svc.features.n_extractions == 1 and len(g.calls) == 2
    assert [x.source for x in r] == ["semantic_cache", "semantic_cache"]
    assert r[0].start_time == 2.0 and r[0].matched_query == "a person opens the door"


def test_threshold_and_scope(env):
    svc, g, v, tmp = env
    svc.run(v, ["a person opens the door"])
    svc.cfg.similarity_threshold = 0.999                  # 0.995 sim is no longer enough
    assert svc.run(v, ["person opening a door"])[0].source == "grounding"
    other = tmp / "o.mp4"; other.write_bytes(b"another video")
    assert svc.run(str(other), ["a person opens the door"])[0].source == "grounding"   # other video_id: no sharing
    assert svc.features.n_extractions == 2


def test_version_invalidation(env):
    svc, g, v, _ = env
    svc.run(v, ["a person opens the door"])
    svc.cfg.feature_schema += 1                            # extractor settings changed -> re-extract, drop query cache
    assert svc.run(v, ["a person opens the door"])[0].source == "grounding" and svc.features.n_extractions == 2
    svc.model_version = "m2"                               # other checkpoint: old segments not reused
    assert svc.run(v, ["a person opens the door"])[0].source == "grounding"


def test_duplicates_within_one_request_ground_once(env):
    svc, g, v, _ = env
    r = svc.run(v, ["a person opens the door", "person opening a door", "person sneezes"])
    assert g.calls == ["a person opens the door", "person sneezes"]
    assert [x.source for x in r] == ["grounding", "semantic_cache", "grounding"]
    assert r[1].start_time == r[0].start_time
