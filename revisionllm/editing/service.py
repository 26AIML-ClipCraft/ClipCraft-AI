"""Scenario Grounding, Scenario Recommendation and Highlight Extraction on ONE cached audio-visual representation.

All three go through EditingService.representation(): FeatureCache hashes the file and encodes it at most once (CLIP + CLAP);
the other calls, and later requests for the same file, read the cached arrays. Results are cached per video in edit_result.
"""
import dataclasses
import hashlib
import json
import os
import sys
from typing import List, Optional

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.editing.audio_events import EVENT_PROMPTS, AudioEventTagger, prompt_embeddings
from revisionllm.editing.config import EditConfig
from revisionllm.editing.db import EditDB
from revisionllm.editing.highlight import plan_highlight, render_highlight, target_seconds
from revisionllm.editing.scenario_llm import ScenarioLLM
from revisionllm.editing.scoring import importance_curve, rank_norm, smooth
from revisionllm.editing.shared import AVRepresentation
from revisionllm.editing.summarize import Segment, summarize
from revisionllm.multiquery.config import CacheConfig
from revisionllm.multiquery.feature_cache import FeatureCache


def unit_vector(cls):
    e = np.asarray(cls, dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-12)


def params_key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class EditingService:
    def __init__(self, cache_cfg: CacheConfig, cfg: EditConfig, clip_pt: str, db: EditDB = None, llm: ScenarioLLM = None,
                 scorer=None, tagger: AudioEventTagger = None, grounder=None, query_gen=None, text=None):
        """scorer: HeadScorer or None (None / missing checkpoint -> training-free saliency). grounder: a MultiQueryGrounder
        for ground(); built by the CLI because it loads the 7B model. query_gen: callable (video_path, duration) -> sentence, used for
        the head's text query when the caller gives no title (query_gen.QwenQuery); text: CLIP text encoder (service.TextEncoder)."""
        self.cache_cfg, self.cfg = cache_cfg, cfg
        self.db = db or EditDB(cache_cfg.dsn, cache_cfg.embedding_dim)
        self.features = grounder.features if grounder is not None else FeatureCache(cache_cfg, self.db, clip_pt)
        self.grounder = grounder
        self.llm = llm or ScenarioLLM(cfg)
        self.scorer, self._tagger = scorer, tagger
        self.query_gen, self._text, self._clip_pt = query_gen, text, clip_pt
        self._rep = {}                                    # video path -> (mtime, AVRepresentation)

    # ---- shared representation ------------------------------------------------------------------------------
    def representation(self, video_path: str) -> AVRepresentation:
        st = os.stat(video_path)
        key = (st.st_mtime_ns, st.st_size)
        hit = self._rep.get(video_path)
        if hit and hit[0] == key:
            return hit[1]
        feats = self.features.get(video_path)             # encodes only if this file was never seen
        av = AVRepresentation.from_features(feats, self.cache_cfg)
        self._rep[video_path] = (key, av)
        return av

    @property
    def tagger(self) -> AudioEventTagger:
        if self._tagger is None:
            emb = prompt_embeddings(EVENT_PROMPTS, self.cache_cfg.clap_name, self.cache_cfg.cache_dir)
            self._tagger = AudioEventTagger(emb, list(EVENT_PROMPTS), self.cfg.event_temperature)
        return self._tagger

    def video_query(self, video_path: str, title: Optional[str] = None):
        """The text the head is conditioned on: the given title, else a VLM description of the video (cached per video).
        -> (text, unit CLIP text vector) or (None, None) when there is no head (the training-free score takes no query)."""
        if self.scorer is None:
            return None, None
        av = self.representation(video_path)
        text = title.strip() if title and title.strip() else None
        if text is None:
            if self.query_gen is None:
                raise RuntimeError("no title given and no query generator: pass title= or build the service with query_gen=QwenQuery()")
            ck = params_key(av.feature_version, "video_query", getattr(self.query_gen, "model", "custom"))
            text = self.db.get_result(av.video_id, "video_query", ck)
            if text is None:
                text = self.query_gen(video_path, av.duration)
                self.db.put_result(av.video_id, "video_query", ck, text)
        if getattr(self.scorer, "uses_text", False):      # UniVTG encodes the text itself (CLIP-B/32)
            return text, None
        if self._text is None:
            from revisionllm.multiquery.service import TextEncoder
            self._text = TextEncoder(self._clip_pt)
        _, cls = self._text(text)
        return text, unit_vector(cls)

    def _key(self, av, *extra) -> str:
        cfg = dataclasses.asdict(self.cfg)
        cfg.pop("openai_model"), cfg.pop("language")      # applied by the LLM steps only (added back there)
        head = self.scorer.version if self.scorer else "none"
        return params_key(av.feature_version, head, cfg, *extra)

    def _curves(self, av, task: str, query=None, qtext: Optional[str] = None):
        """task 'highlight': head score s + f (reference main/inference_hl.py, eval_mode=add) mixed with audio events and motion.
        task 'summary': head score f only (reference inference_qfvs.py picks the top clips by the foreground score).
        Without a head both use the training-free representativeness stand-in."""
        events = self.tagger.probs(av)
        if self.scorer is None:
            head = None
        elif getattr(self.scorer, "uses_text", False):
            head = self.scorer.predict(av, query, text=qtext)
        else:
            head = self.scorer.predict(av, query)
        if head is None:
            imp, parts = importance_curve(av, events, self.tagger.classes, self.cfg)
        elif task == "highlight":
            imp, parts = importance_curve(av, events, self.tagger.classes, self.cfg, head["saliency"] + head["foreground"])
        else:
            imp = rank_norm(smooth(head["foreground"], self.cfg.smooth_sec))
            parts = {"saliency_source": "head:f"}
        return imp, events, parts

    # ---- 1. Scenario Grounding ------------------------------------------------------------------------------
    def ground(self, video_path: str, queries: List[str]):
        if self.grounder is None:
            raise RuntimeError("grounding needs the ReVisionLLM checkpoint: build the service with grounder=MultiQueryGrounder")
        return self.grounder.run(video_path, queries)

    # ---- 2. Scenario Recommendation -------------------------------------------------------------------------
    def summary(self, video_path: str, title: Optional[str] = None) -> List[Segment]:
        av = self.representation(video_path)
        qtext, qvec = self.video_query(video_path, title)
        key = self._key(av, "summary", qtext)
        row = self.db.get_result(av.video_id, "summary", key)
        if row is not None:
            return [Segment(**s) for s in row]
        imp, events, _ = self._curves(av, "summary", qvec, qtext)
        segs = summarize(av, imp, events, self.tagger.classes, self.cfg)
        self.db.put_result(av.video_id, "summary", key, [s.to_dict() for s in segs])
        return segs

    def recommend(self, video_path: str, goal: Optional[str] = None, target_sec: Optional[float] = None,
                  title: Optional[str] = None) -> dict:
        av = self.representation(video_path)
        segs = self.summary(video_path, title)
        for s in segs:                                    # captions are per segment, so a changed goal does not redo them
            ck = params_key(av.feature_version, "caption", s.start, s.end, s.key_times, self.cfg.openai_model, self.cfg.language)
            cap = self.db.get_result(av.video_id, "caption", ck)
            if cap is None:
                cap = self.llm.caption(video_path, s)
                self.db.put_result(av.video_id, "caption", ck, cap)
            s.caption = cap
        sk = params_key(self._key(av, "scenario"), [s.to_dict() for s in segs], goal, target_sec, self.cfg.openai_model,
                        self.cfg.language)
        out = self.db.get_result(av.video_id, "scenario", sk)
        if out is None:
            out = self.llm.recommend(segs, av.duration, goal, target_sec)
            out["segments"] = [s.to_dict() for s in segs]
            self.db.put_result(av.video_id, "scenario", sk, out)
        return out

    # ---- 3. Highlight Extraction ----------------------------------------------------------------------------
    def highlight(self, video_path: str, out_path: Optional[str] = None, target_sec: Optional[float] = None,
                  title: Optional[str] = None) -> dict:
        av = self.representation(video_path)
        qtext, qvec = self.video_query(video_path, title)
        key = self._key(av, "highlight", target_sec, qtext)
        plan = self.db.get_result(av.video_id, "highlight", key)
        if plan is None:
            imp, events, parts = self._curves(av, "highlight", qvec, qtext)
            clips = plan_highlight(imp, events, self.tagger.classes, av.duration, self.cfg, target_sec)
            plan = {"target_sec": target_seconds(av.duration, self.cfg, target_sec), "saliency_source": parts["saliency_source"],
                    "query": qtext, "clips": clips}
            self.db.put_result(av.video_id, "highlight", key, plan)
        if out_path:
            render_highlight(video_path, plan["clips"], out_path, av.has_audio)
            plan = {**plan, "output": out_path}
        return plan
