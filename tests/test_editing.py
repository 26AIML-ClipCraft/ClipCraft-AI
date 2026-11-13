"""revisionllm.editing without a GPU, the 7B model, OpenAI or real video (synthetic features, fake LLM client).
The service test shares multiquery's PostgreSQL: CLIPCRAFT_TEST_DSN=postgresql://localhost:5433/clipcraft pytest tests/test_editing.py"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from revisionllm.editing.audio_events import EVENT_PROMPTS, AudioEventTagger
from revisionllm.editing.config import EditConfig
from revisionllm.editing.highlight import plan_highlight
from revisionllm.editing.scenario_llm import ScenarioLLM, validate_scenario
from revisionllm.editing.scoring import importance_curve
from revisionllm.editing.shared import AVRepresentation, to_second_grid
from revisionllm.editing.summarize import Segment, summarize

CLASSES = list(EVENT_PROMPTS)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_av(T=120, seed=0):
    """Three scenes (0-40, 40-80, 80-120), each with its own visual direction; laughter at 55-62 s."""
    rng = np.random.default_rng(seed)
    base = unit(rng.normal(size=(3, 768)))
    v = unit(np.repeat(base, T // 3, axis=0) + 0.05 * rng.normal(size=(T, 768))).astype(np.float32)
    a = unit(rng.normal(size=(T, 512))).astype(np.float32)
    return AVRepresentation(v, a, float(T), video_id=1, video_hash="h", feature_version="fv")


def make_tagger(av):
    text = unit(np.random.default_rng(1).normal(size=(len(CLASSES), 512))).astype(np.float32)
    text[CLASSES.index("laughter")] = av.audio[55:62].mean(0)
    return AudioEventTagger(unit(text), CLASSES, temperature=60.0)


def test_second_grid():
    v, a = to_second_grid(np.ones((10, 768)), np.ones((4, 512)), 2.0, 1.0)
    assert v.shape == (5, 768) and a.shape == (5, 512)
    assert np.allclose(np.linalg.norm(v, axis=1), 1) and np.allclose(a[4], a[3])        # audio shorter than video: last row held
    _, silent = to_second_grid(np.ones((10, 768)), None)
    assert not silent.any()


def test_service_scores_follow_the_reference(tmp_path):
    """highlight = rank(s + f) mixed with audio events; summary = rank(f) only (UniVTG inference_hl / inference_qfvs)."""
    from revisionllm.editing.service import EditingService
    av, tagger = make_av(), None
    tagger = make_tagger(av)
    T = len(av)
    s, f = np.zeros(T), np.zeros(T)
    s[10], f[90] = 5.0, 5.0
    scorer = SimpleNamespace(predict=lambda a, q=None: {"saliency": s, "foreground": f}, version="v")
    svc = EditingService.__new__(EditingService)
    svc.cfg, svc.scorer, svc._tagger = EditConfig(w_audio=0.0, smooth_sec=1), scorer, tagger
    imp_vs, _, parts_vs = svc._curves(av, "summary")
    imp_hl, _, parts_hl = svc._curves(av, "highlight")
    assert int(np.argmax(imp_vs)) == 90 and parts_vs["saliency_source"] == "head:f"           # f alone
    assert {int(np.argsort(-imp_hl)[0]), int(np.argsort(-imp_hl)[1])} == {10, 90} and parts_hl["saliency_source"] == "head:s+f"


def test_qwen_query_subprocess(tmp_path, monkeypatch):
    """QwenQuery only builds the command and parses the last JSON line; the model itself runs in the other environment."""
    from revisionllm.editing import query_gen
    monkeypatch.setattr(query_gen, "grab_frame", lambda v, t, max_side=512: b"\xff\xd8jpeg")
    ok = tmp_path / "py_ok.sh"; ok.write_text('#!/bin/sh\necho "loading..."\necho \'{"query": "a person cooks pasta"}\'\n'); ok.chmod(0o755)
    bad = tmp_path / "py_bad.sh"; bad.write_text("#!/bin/sh\necho boom >&2\nexit 3\n"); bad.chmod(0o755)
    q = query_gen.QwenQuery(python=str(ok), n_frames=4)
    assert q("v.mp4", 100.0) == "a person cooks pasta"
    assert q.frame_times(100.0) == sorted(q.frame_times(100.0)) and len(q.frame_times(100.0)) == 4
    with pytest.raises(RuntimeError, match="boom"):
        query_gen.QwenQuery(python=str(bad))("v.mp4", 100.0)


def test_summarize_finds_the_scenes():
    av, cfg = make_av(), EditConfig(n_scenes=3)
    tagger = make_tagger(av)
    events = tagger.probs(av)
    imp, parts = importance_curve(av, events, CLASSES, cfg)
    assert parts["saliency_source"] == "representativeness" and 0 <= imp.min() and imp.max() <= 1
    segs = summarize(av, imp, events, CLASSES, cfg)
    assert 1 <= len(segs) <= 3
    assert all(a.end <= b.start + 1e-6 for a, b in zip(segs, segs[1:])) and [s.id for s in segs] == list(range(len(segs)))
    assert all(0 <= s.start < s.end <= av.duration and s.key_times for s in segs)
    win = cfg.window_sec(av.duration)
    assert all(abs((s.end - s.start) - win) < win / 2 + 1e-6 or s.end == av.duration for s in segs)   # fixed-length windows


def test_window_grows_with_the_video():
    cfg = EditConfig()
    assert [cfg.window_sec(d) for d in (60, 359, 360, 1199, 1200, 3600)] == [12, 12, 20, 20, 30, 30]


def test_highlight_prefers_laughter_and_respects_budget():
    av, cfg = make_av(), EditConfig(w_saliency=0.0, w_audio=1.0, highlight_min_sec=10, highlight_max_sec=20)
    tagger = make_tagger(av)
    events = tagger.probs(av)
    imp, _ = importance_curve(av, events, CLASSES, cfg)
    clips = plan_highlight(imp, events, CLASSES, av.duration, cfg)
    assert clips and any(c["start"] <= 58 < c["end"] for c in clips)
    assert sum(c["end"] - c["start"] for c in clips) <= 20 + cfg.window_sec(av.duration)
    assert all(a["end"] <= b["start"] for a, b in zip(clips, clips[1:]))


def test_no_audio_video():
    av = make_av()
    silent = AVRepresentation(av.vision, np.zeros_like(av.audio), av.duration)
    ev = make_tagger(av).probs(silent)
    assert not silent.has_audio and ev[:, CLASSES.index("silence")].min() == 1.0
    imp, _ = importance_curve(silent, ev, CLASSES, EditConfig())
    assert np.isfinite(imp).all()


class FakeClient:
    """Stands in for openai.OpenAI: records the requests, returns canned replies."""
    def __init__(self, scenario):
        self.scenario, self.requests = scenario, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kw):
        self.requests.append((model, messages, kw))
        text = json.dumps(self.scenario) if kw.get("response_format") else "a caption"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def segs3():
    return [Segment(0, 0, 40, .5, {"speech": .6}, [20.0], "intro"), Segment(1, 40, 80, .9, {"laughter": .5}, [60.0], "joke"),
            Segment(2, 80, 120, .4, {}, [100.0], "outro")]


def test_scenario_only_uses_existing_segments():
    raw = {"title": "t", "summary": "s", "scenes": [
        {"order": 2, "segment_id": 0, "start": 0, "end": 40, "description": "d0", "reason": "r"},
        {"order": 1, "segment_id": 1, "start": 30, "end": 200, "description": "d1", "reason": "r"},     # trim outside the segment
        {"order": 3, "segment_id": 9, "start": 0, "end": 5, "description": "made up", "reason": "r"}]}
    client = FakeClient(raw)
    out = ScenarioLLM(EditConfig(), client).recommend(segs3(), 120.0, goal="short vlog")
    assert [s["segment_id"] for s in out["scenes"]] == [0, 1] and [s["order"] for s in out["scenes"]] == [1, 2]   # sorted by time, not by the LLM order
    assert out["scenes"][1]["start"] == 40 and out["scenes"][1]["end"] == 80          # clamped into segment 1
    model, messages, kw = client.requests[0]
    assert model == "gpt-4o-mini" and kw["response_format"] == {"type": "json_object"}
    assert "short vlog" in messages[1]["content"] and "joke" in messages[1]["content"]
    with pytest.raises(ValueError):
        validate_scenario({"scenes": [{"segment_id": 7}]}, segs3())


DSN = os.environ.get("CLIPCRAFT_TEST_DSN")


class FakeText:
    def __call__(self, text):
        v = np.zeros(768, np.float32); v[abs(hash(text)) % 768] = 1.0
        return None, v


class FakeScorer:
    version = "fake"

    def __init__(self): self.queries = []

    def predict(self, av, query=None):
        self.queries.append(query)
        T = len(av)
        return {"saliency": np.linspace(0, 1, T), "foreground": np.linspace(0, 1, T)}


@pytest.mark.skipif(not DSN, reason="CLIPCRAFT_TEST_DSN not set")
def test_three_features_share_one_encoding(tmp_path, monkeypatch):
    from revisionllm.editing import service as svc_mod
    from revisionllm.editing.db import EditDB
    from revisionllm.editing.service import EditingService
    from revisionllm.multiquery import feature_cache as fc
    from revisionllm.multiquery.config import CacheConfig

    av = make_av()
    cache_cfg = CacheConfig(dsn=DSN, cache_dir=str(tmp_path / "cache"))
    db = EditDB(DSN)
    db.conn.execute("TRUNCATE video, query_cache, edit_result RESTART IDENTITY CASCADE")
    raw_v = np.repeat(av.vision, 2, axis=0).astype(np.float16)
    monkeypatch.setattr(fc.FeatureCache, "_extract", lambda self, p: (raw_v, av.audio.astype(np.float16)))
    monkeypatch.setattr(fc, "probe_duration", lambda p: av.duration)
    monkeypatch.setattr(svc_mod, "render_highlight", lambda v, clips, out, has_audio: out)
    monkeypatch.setattr("revisionllm.editing.scenario_llm.grab_frame", lambda v, t, max_side=512: b"\xff\xd8jpeg")
    client = FakeClient({"title": "t", "summary": "s", "scenes": [
        {"order": 1, "segment_id": 0, "start": 0, "end": 40, "description": "d", "reason": "r"}]})
    svc = EditingService(cache_cfg, EditConfig(n_scenes=3), "unused", db=db, llm=ScenarioLLM(EditConfig(), client),
                         tagger=make_tagger(av))
    video = tmp_path / "v.mp4"; video.write_bytes(b"video-bytes")

    scen = svc.recommend(str(video), goal="x")
    plan = svc.highlight(str(video), str(tmp_path / "h.mp4"))
    assert svc.features.n_extractions == 1 and scen["scenes"] and plan["clips"] and plan["output"].endswith("h.mp4")

    n_calls = len(client.requests)
    svc2 = EditingService(cache_cfg, EditConfig(n_scenes=3), "unused", db=db, llm=ScenarioLLM(EditConfig(), client),
                          tagger=make_tagger(av))                                         # a new process: nothing in memory
    assert svc2.recommend(str(video), goal="x") == scen
    assert svc2.highlight(str(video))["clips"] == plan["clips"]
    assert svc2.features.n_extractions == 0 and len(client.requests) == n_calls           # no re-encoding, no new GPT call


@pytest.mark.skipif(not DSN, reason="CLIPCRAFT_TEST_DSN not set")
def test_title_or_vlm_query_conditions_the_head(tmp_path, monkeypatch):
    from revisionllm.editing.db import EditDB
    from revisionllm.editing.service import EditingService
    from revisionllm.multiquery import feature_cache as fc
    from revisionllm.multiquery.config import CacheConfig

    av = make_av()
    cache_cfg = CacheConfig(dsn=DSN, cache_dir=str(tmp_path / "cache"))
    db = EditDB(DSN)
    db.conn.execute("TRUNCATE video, query_cache, edit_result RESTART IDENTITY CASCADE")
    monkeypatch.setattr(fc.FeatureCache, "_extract", lambda self, p: (np.repeat(av.vision, 2, 0).astype(np.float16), av.audio.astype(np.float16)))
    monkeypatch.setattr(fc, "probe_duration", lambda p: av.duration)
    calls = []

    def gen(path, duration):
        calls.append(duration); return "a cooking video"
    gen.model = "stub-vlm"
    video = tmp_path / "v.mp4"; video.write_bytes(b"video-bytes")

    def build(scorer):
        return EditingService(cache_cfg, EditConfig(), "unused", db=db, scorer=scorer, tagger=make_tagger(av), query_gen=gen, text=FakeText())

    sc = FakeScorer()
    svc = build(sc)
    assert svc.highlight(str(video), title="My trip to Jeju")["query"] == "My trip to Jeju" and not calls      # a title: no VLM call
    plan = svc.highlight(str(video))                                                                       # no title: the VLM describes it
    assert plan["query"] == "a cooking video" and len(calls) == 1
    assert not np.allclose(sc.queries[0], sc.queries[1]) and np.isclose(np.linalg.norm(sc.queries[1]), 1.0)  # different text, unit vector
    svc.summary(str(video))
    sc2 = FakeScorer()
    assert build(sc2).highlight(str(video))["clips"] == plan["clips"] and len(calls) == 1 and not sc2.queries   # description and plan cached
    with pytest.raises(RuntimeError):
        EditingService(cache_cfg, EditConfig(), "unused", db=db, scorer=sc, tagger=make_tagger(av), text=FakeText()).highlight(
            str(video), target_sec=20)                                                                        # no title, no generator
