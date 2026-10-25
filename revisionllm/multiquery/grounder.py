"""Query-dependent part: the existing dense (single-stage) grounding of eval_nlq_negative.py, run on cached features.

The window construction, prompt, ``inference`` call (alternate model -> iteration_step 1), answer parsing (``iou``) and
cosine/top-k candidate scoring are the ones of eval_nlq_negative.eval(); only the I/O (LMDB -> cached arrays) differs.
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.eval.audio_utils import AudioStore
from revisionllm.eval.eval_nlq_negative import iou, questions, parse_args
from revisionllm.eval.similarity import _topk_pooling
from revisionllm.inference import inference
from revisionllm.model.adapter.tensor_utils import pad_sequences_1d
from revisionllm.model.builder import load_pretrained_model
from revisionllm.utils import disable_torch_init

DEBUG_WINDOW = 500      # scripts/audio/env.sh WINDOW
NUM_FRAMES = 250        # scripts/audio/env.sh FRAMES
BATCH = 8               # windows per forward pass (DENSE_BATCH in eval_row.sh)


def norm_sentence(sent: str) -> str:
    """Same normalisation as charades_sta.norm_sentence (used for the released test query features)."""
    sent = sent.strip().lower()
    return sent[:-1] if sent.endswith(".") else sent


def model_args(clip_pt, vicuna, stage2, stage1_proj, clip_adapter, cfg, attn="sdpa"):
    """The argument namespace load_pretrained_model() needs, built from eval_nlq_negative's parser exactly like
    scripts/audio/eval_row.sh calls it for the audio-visual row."""
    argv = ["x", "--clip_path", clip_pt, "--model_base", vicuna, "--stage2", stage2,
            "--pretrain_mm_mlp_adapter", stage1_proj, "--pretrain_clip_adapter", clip_adapter,
            "--attn_implementation", attn, "--adapter_input_dim", "768", "--feature_fps", str(cfg.feature_fps),
            "--cross_attn", "True", "--clip_adapter_text", "True", "--clip_adapter_feature", "alternate",
            "--audio_fusion", "True", "--audio_feat_folder", "unused", "--audio_dim", str(cfg.audio_dim),
            "--audio_hop_sec", str(cfg.audio_hop_sec), "--debug_window", str(DEBUG_WINDOW),
            "--num_frames", str(NUM_FRAMES)]
    old, sys.argv = sys.argv, argv
    try:
        return parse_args()
    finally:
        sys.argv = old


class PreparedVideo:
    """Windows of one video (query independent): built once, shared by every query."""

    def __init__(self, feats, duration, device, dtype, cfg):
        T = len(feats.vision)
        vision, rows = feats.vision, np.arange(T)
        self.short = duration <= DEBUG_WINDOW
        if self.short:      # whole video = one window, stretched onto NUM_FRAMES frames
            idx = np.linspace(0, T - 1, NUM_FRAMES, dtype=np.int32)
            vision, rows = vision[idx], rows[idx]
        ctx_l = len(vision)
        clip_length = DEBUG_WINDOW * cfg.feature_fps
        widx = [0] if self.short else list(range(math.ceil(ctx_l / (clip_length // 2)) - 1))
        store = AudioStore(None, audio_dim=cfg.audio_dim, hop_sec=cfg.audio_hop_sec, feature_fps=cfg.feature_fps)
        audio = None if feats.audio is None else np.asarray(feats.audio, dtype=np.float32)
        v, a = [], []
        for i in widx:
            start = max(i * clip_length // 2, 0)
            end = min(i * clip_length // 2 + clip_length, ctx_l - 1)
            s = np.linspace(start, end, NUM_FRAMES, dtype=np.int32)
            v.append(vision[s])
            a.append(store.align(audio, rows[s]))
        self.features = torch.from_numpy(np.array(v)).to(dtype=dtype, device=device)
        self.audio = torch.from_numpy(np.array(a, dtype=np.float32)).to(dtype=dtype, device=device)
        self.duration = duration
        self.num_frames_video = NUM_FRAMES if self.short else int(duration * NUM_FRAMES / DEBUG_WINDOW)


class Grounder:
    def __init__(self, args, cfg, topk_pool=True):
        disable_torch_init()
        self.cfg, self.args, self.topk_pool = cfg, args, topk_pool
        self.tokenizer, self.model, _ = load_pretrained_model(args, args.stage2, None)
        if torch.cuda.is_available():
            self.model = self.model.bfloat16().cuda()
            self.dtype, self.device = torch.bfloat16, "cuda"
        else:
            self.model.to(torch.float32)
            self.dtype, self.device = torch.float32, "cpu"
        self.model.eval()

    def prepare(self, feats) -> PreparedVideo:
        return PreparedVideo(feats, feats.duration, self.device, self.dtype, self.cfg)

    def _prompt(self, sentence):
        return "<video>\n" + questions["mad_grounding"].format(sentence)

    @torch.no_grad()
    def ground_many(self, pv: PreparedVideo, queries, batch: int = BATCH):
        """queries: [(sentence, token_feats, cls_feats)] -> [(start, end) | None] in the same order.

        Rows of one forward pass are (query, window) pairs of the SAME shared windows; prompts of different token
        length are left-padded (the model keeps the padding masked during generation). Rows are ordered by prompt
        length so a pass pads as little as possible."""
        qs = [(sent, torch.as_tensor(t).to(device=self.device, dtype=self.dtype),
               torch.as_tensor(c).to(device=self.device, dtype=self.dtype)) for sent, t, c in queries]
        self.model.config.tokenizer_padding_side = "left"
        order = sorted(range(len(qs)), key=lambda i: len(self.tokenizer(self._prompt(qs[i][0])).input_ids))
        W = pv.features.shape[0]
        answers = [[None] * W for _ in qs]
        rows = [(qi, w) for qi in order for w in range(W)]
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            widx = [w for _, w in chunk]
            feat, aud = pv.features[widx], pv.audio[widx]
            qt = pad_sequences_1d([qs[qi][1] for qi, _ in chunk], dtype=self.dtype, device=self.device,
                                  fixed_length=None)
            prompts = [self._prompt(qs[qi][0]) for qi, _ in chunk]
            ans, _ = inference(self.model, feat, qt, prompts, self.tokenizer, return_list=True,
                               audio_feats=aud, iteration_step=1)
            for (qi, w), a in zip(chunk, ans):
                answers[qi][w] = a
        return [self._best(pv, answers[qi], qs[qi][2]) for qi in range(len(qs))]

    def ground(self, pv: PreparedVideo, sentence: str, token_feats, cls_feats, batch: int = BATCH):
        """Single query (kept for reference / comparison with ground_many)."""
        return self.ground_many(pv, [(sentence, token_feats, cls_feats)], batch)[0]

    def _best(self, pv, answers, qc):
        """(start_sec, end_sec) of the best-scoring window prediction, or None if no window predicted an interval."""
        frames, _, _ = iou(answers, (0.0, 1.0), NUM_FRAMES, pv.num_frames_video, [], False)
        best, best_score = None, -1e9
        scores = []
        for k, (f, t) in frames.items():
            p = pv.features[k][f:t + 1]
            p = p / p.norm(dim=0, keepdim=True)
            if self.topk_pool:
                p = _topk_pooling(qc[None], p[None], min(p.shape[0], 3))[0]
                sc = torch.einsum('bd,d->b', p, qc)
            else:
                sc = torch.einsum('bd,d->b', p, qc).mean()
            scores.append((k, f, t, sc.item()))
        if not scores:
            return None
        m = max(s[3] for s in scores)       # eval normalises by the max; ranking is unchanged
        k, f, t, _ = max(scores, key=lambda s: s[3] / m)
        g0, g1 = k * NUM_FRAMES // 2 + f, k * NUM_FRAMES // 2 + t
        to_sec = lambda g: float(min(max(g / pv.num_frames_video * pv.duration, 0.0), pv.duration))
        return to_sec(g0), to_sec(g1)
