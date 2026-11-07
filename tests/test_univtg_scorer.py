"""UniVTG scorer on cached CLIP-L/14 features: parity with the original code, output contract, service wiring.
Needs checkpoints/univtg/* and the CLIP-B/32 text model in the HF cache; the parity test also needs the original repo
(UNIVTG_ROOT, default ~/UniVTG) and its checkpoint (UNIVTG_FT_DIR, default ~/univtg_assets/ckpt_ft_clip)."""
import os
import sys
import types
import warnings

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from revisionllm.editing.audio_events import EVENT_PROMPTS, AudioEventTagger
from revisionllm.editing.config import EditConfig
from revisionllm.editing.service import EditingService
from revisionllm.editing.shared import AVRepresentation
from revisionllm.editing.univtg.scorer import DEFAULT_ADAPTER, DEFAULT_CKPT, UniVTGScorer

CLASSES = list(EVENT_PROMPTS)
UNIVTG_ROOT = os.environ.get("UNIVTG_ROOT", os.path.expanduser("~/UniVTG"))
FT_DIR = os.environ.get("UNIVTG_FT_DIR", os.path.expanduser("~/univtg_assets/ckpt_ft_clip")) + "/"
HAVE_WEIGHTS = os.path.exists(DEFAULT_CKPT) and os.path.exists(DEFAULT_ADAPTER)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_av(T, seed=0):
    rng = np.random.default_rng(seed)
    v = unit(rng.normal(size=(T, 768))).astype(np.float32)
    return AVRepresentation(v, np.zeros((T, 512), np.float32), float(T), video_id=1, video_hash="h", feature_version="fv")


@pytest.fixture(scope="module")
def scorer():
    if not HAVE_WEIGHTS:
        pytest.skip("checkpoints/univtg not present")
    s = UniVTGScorer(device="cpu")
    try:
        s.text_feature("a dog runs")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"CLIP-B/32 text model unavailable: {e}")
    return s


def test_parity_with_original_univtg(scorer):
    """The vendored Model gives the same outputs as showlab/UniVTG's own code on the same weights."""
    if not (os.path.isdir(UNIVTG_ROOT) and os.path.exists(FT_DIR + "model_best.ckpt")):
        pytest.skip("original UniVTG repo / checkpoint not available")
    import json
    stub = types.ModuleType("model.matcher"); stub.build_matcher = None          # training-only, imports scipy
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "model" or k.startswith("model.")}
    sys.modules["model.matcher"] = stub
    sys.path.insert(0, UNIVTG_ROOT)
    try:
        from model.position_encoding import build_position_encoding as bpe
        from model.transformer_encoder_droppath import build_transformer as bt
        from model.univtg import Model as Orig
        opt = json.load(open(FT_DIR + "opt.json")); args = types.SimpleNamespace(**opt)
        orig = Orig(bt(args), *bpe(args), txt_dim=512, vid_dim=514, input_dropout=opt["input_dropout"], span_loss_type=opt["span_loss_type"],
                    use_txt_pos=opt["use_txt_pos"], n_input_proj=opt["n_input_proj"])
        sd = torch.load(FT_DIR + "model_best.ckpt", map_location="cpu")["model"]
        orig.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=True)
    finally:
        sys.path.remove(UNIVTG_ROOT)
        for k in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
    orig.eval()
    g = torch.Generator().manual_seed(0)
    v = torch.randn(2, 40, 514, generator=g); q = torch.randn(2, 9, 512, generator=g)
    mv, mq = torch.ones(2, 40, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool)
    with torch.no_grad():
        a = orig(src_txt=q, src_txt_mask=mq, src_vid=v, src_vid_mask=mv)
        b = scorer.model(src_txt=q, src_txt_mask=mq, src_vid=v, src_vid_mask=mv)
    for k in ("saliency_scores", "pred_logits", "pred_spans"):
        assert torch.equal(a[k], b[k]), k


def test_output_contract_and_grid(scorer):
    for T in (1, 2, 101, 200):
        av = make_av(T)
        out = scorer.predict(av, None, text="a dog runs in the park")
        assert set(out) == {"saliency", "foreground", "logit"}
        for k, x in out.items():
            assert x.shape == (T,) and np.isfinite(x).all(), (T, k)
        assert np.array_equal(out["saliency"][:2 * (T // 2):2], out["saliency"][1:2 * (T // 2):2])      # a 2 s clip covers two seconds


def test_text_is_required_and_changes_scores(scorer):
    av = make_av(120)
    with pytest.raises(ValueError):
        scorer.predict(av, None, text="  ")
    a = scorer.predict(av, None, text="a dog runs in the park")["saliency"]
    b = scorer.predict(av, None, text="a plate of pasta")["saliency"]
    assert not np.allclose(a, b)                                  # query-conditioned, not a fixed curve
    assert np.allclose(a, scorer.predict(av, None, text="a dog runs in the park")["saliency"])          # deterministic


def test_long_video_chunk_path(scorer):
    scorer._warned = False
    av = make_av(2 * 330 + 120)                                    # 390 clips > max_clips -> two equal chunks
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = scorer.predict(av, None, text="a dog runs in the park")
    assert out["saliency"].shape == (len(av),) and np.isfinite(out["saliency"]).all()
    assert any("not been validated" in str(x.message) for x in w)


def test_service_passes_text_and_skips_clip_l14_text_encoder(scorer):
    av = make_av(120)
    svc = object.__new__(EditingService)
    svc.cfg, svc.scorer = EditConfig(), scorer
    svc.representation = lambda path: av
    svc._tagger = AudioEventTagger(unit(np.random.default_rng(1).normal(size=(len(CLASSES), 512))).astype(np.float32), CLASSES, 30.0)
    svc._text = None
    text, vec = svc.video_query("x.mp4", "a dog runs in the park")
    assert text == "a dog runs in the park" and vec is None and svc._text is None      # no CLIP-L/14 text encoder was loaded
    for task in ("highlight", "summary"):
        imp, events, parts = svc._curves(av, task, vec, text)
        assert imp.shape == (len(av),) and np.isfinite(imp).all() and 0 <= imp.min() and imp.max() <= 1
    assert svc._key(av, "highlight", None, text) != svc._key(av, "highlight", None, "another title")
