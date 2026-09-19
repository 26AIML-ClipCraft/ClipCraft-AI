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
