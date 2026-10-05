"""
Unit checks for the ClipCraft audio branch (CPU only, no model weights needed).

  python -m pytest tests/test_audio_fusion.py -q

1. AudioFusion with zero-initialised gate/output is an exact identity  => step-0 == released model
2. AudioFusion handles (b,t,d) and (b,v,t,d) alike (hierarchy path)
3. Loader-side alignment: frame index -> nearest 1-s audio row, consistent between
   the training loader (LazySupervisedDataset.align_audio_to_frames) and the eval AudioStore
4. Collator stacks audio with the same rule as images and modality dropout zeroes exactly one modality
"""
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from revisionllm.model.adapter.audio_fusion import AudioFusion
from revisionllm.eval.audio_utils import AudioStore


def test_identity_at_init():
    torch.manual_seed(0)
    m = AudioFusion(audio_dim=512, visual_dim=768, gate_init=0.0)
    v = torch.randn(2, 250, 768)
    a = torch.randn(2, 250, 512)
    out = m(v, a)
    assert torch.equal(out, v), "zero-init fusion must be an exact identity"
    assert torch.equal(m(v, None), v)


def test_hierarchy_shape_and_gradient():
    m = AudioFusion(audio_dim=512, visual_dim=768, gate_init=0.1)
    v = torch.randn(1, 33, 250, 768)
    a = torch.randn(1, 33, 250, 512)
    out = m(v, a)
    assert out.shape == v.shape
    out.sum().backward()
    assert m.audio_fc_in.weight.grad is not None and m.audio_gate.grad is not None
    # zero-init output layer: fc_out gets gradient even though its weight is 0
    assert m.audio_fc_out.weight.grad.abs().sum() > 0


def _grads_after_steps(gate_init, steps=5):
    torch.manual_seed(0)
    m = AudioFusion(audio_dim=512, visual_dim=768, gate_init=gate_init)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    v, a = torch.randn(2, 20, 768), torch.randn(2, 20, 512)
    for _ in range(steps):
        opt.zero_grad()
        m(v, a).pow(2).mean().backward()
        opt.step()
    opt.zero_grad()
    m(v, a).pow(2).mean().backward()
    return m


def test_stage_a_gate_init_escapes_zero_gradient():
    # gate=0 and fc_out=0 at once => every audio parameter has exactly zero gradient, forever.
    dead = _grads_after_steps(0.0)
    assert float(dead.audio_gate) == 0.0 and dead.audio_fc_out.weight.abs().sum() == 0
    # Stage A / B default (gate_init 0.1): still identity at step 0, but gate and fc_out must start moving.
    v, a = torch.randn(2, 20, 768), torch.randn(2, 20, 512)
    assert torch.equal(AudioFusion(512, 768, gate_init=0.1)(v, a), v)
    live = _grads_after_steps(0.1)
    assert live.audio_fc_out.weight.abs().sum() > 0
    assert live.audio_gate.grad.abs() > 0 and live.audio_fc_in.weight.grad.abs().sum() > 0


def test_alignment_matches_between_train_and_eval():
    from revisionllm.train.dataset import LazySupervisedDataset
    fps, hop = 2.0, 1.0
    audio = np.arange(30, dtype=np.float32)[:, None].repeat(4, 1)  # 30 s, row value == second
    frames = np.linspace(0, 59, 25, dtype=np.int32)                # 2 fps frame indices over 30 s
    store = AudioStore(None, audio_dim=4, hop_sec=hop, feature_fps=fps)
    eval_rows = store.align(audio, frames)[:, 0]
    ds = LazySupervisedDataset.__new__(LazySupervisedDataset)
    ds.data_args = types.SimpleNamespace(feature_fps=fps, audio_hop_sec=hop, audio_dim=4)
    train_rows = ds.align_audio_to_frames(audio, frames)[:, 0]
    np.testing.assert_array_equal(eval_rows, train_rows)
    np.testing.assert_array_equal(eval_rows, np.floor(frames / fps))
    # missing audio => zeros of the right shape
    assert store.align(None, frames).shape == (25, 4)


def test_collator_audio_and_modality_dropout():
    from revisionllm.train.dataset import DataCollatorForSupervisedDataset
    import random
    tok = types.SimpleNamespace(pad_token_id=0, model_max_length=64)
    col = DataCollatorForSupervisedDataset(tokenizer=tok, modality_dropout_visual=1.0, modality_dropout_audio=0.0)
    inst = [dict(input_ids=torch.tensor([1, 2, 3]), labels=torch.tensor([1, 2, 3]),
                 image=torch.ones(250, 768), audio=torch.ones(250, 512)) for _ in range(2)]
    random.seed(0)
    batch = col(inst)
    assert batch['images'].shape == (2, 250, 768) and batch['audio_feats'].shape == (2, 250, 512)
    assert batch['images'].abs().sum() == 0 and batch['audio_feats'].abs().sum() > 0
