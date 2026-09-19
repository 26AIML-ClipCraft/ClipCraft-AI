"""
Additive audio-visual fusion (ClipCraft v1).

    v_tilde_t = v_t + alpha * LN( W2 GELU(W1 a_t + b1) + b2 )

* ``v_t``  : CLIP ViT-L/14 frame feature (768-d), already time-aligned.
* ``a_t``  : CLAP audio feature (512-d) aligned to the same frame grid by the
             dataset loader (nearest 1-second hop).
* ``W2``/``b2`` and ``alpha`` are zero-initialised, so at step 0 the module is
  an exact identity on the visual features and the model behaves like the
  released ReVisionLLM checkpoint.

The module is shape preserving: it accepts ``(..., T, Dv)`` visual tensors with
audio ``(..., T, Da)`` of the same leading shape, so both the dense path
(``b t d``) and the hierarchical path (``b v t d``) work unchanged.
"""
import torch
import torch.nn as nn


class AudioFusion(nn.Module):
    def __init__(self, audio_dim: int = 512, visual_dim: int = 768, hidden_dim: int = None,
                 gate_init: float = 0.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or visual_dim
        # Names are deliberately unique (``audio_fc_in`` / ``audio_fc_out``) so
        # they can never collide with LoRA ``target_modules`` saved in the
        # released adapter_config.json (q_proj, k_proj, ...).
        self.audio_fc_in = nn.Linear(audio_dim, hidden_dim)
        self.audio_fc_out = nn.Linear(hidden_dim, visual_dim)
        self.audio_norm = nn.LayerNorm(visual_dim)
        self.audio_dropout = nn.Dropout(dropout)
        self.audio_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.audio_fc_in.weight)
        nn.init.zeros_(self.audio_fc_in.bias)
        # zero-init the output projection => exact identity at step 0
        nn.init.zeros_(self.audio_fc_out.weight)
        nn.init.zeros_(self.audio_fc_out.bias)

    def audio_branch(self, audio: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.audio_fc_in(audio))
        h = self.audio_fc_out(self.audio_dropout(h))
        return self.audio_norm(h)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor = None) -> torch.Tensor:
        """
        visual: (..., T, Dv)
        audio : (..., T, Da) or None. None => identity.
        """
        if audio is None:
            return visual
        if audio.shape[:-1] != visual.shape[:-1]:
            raise ValueError(
                f"AudioFusion: audio leading shape {tuple(audio.shape[:-1])} must match "
                f"visual leading shape {tuple(visual.shape[:-1])}")
        audio = audio.to(dtype=self.audio_fc_in.weight.dtype, device=visual.device)
        fused = self.audio_branch(audio).to(visual.dtype)
        return visual + self.audio_gate.to(visual.dtype) * fused

    def extra_repr(self) -> str:
        return f"audio_dim={self.audio_dim}, visual_dim={self.visual_dim}, gate={float(self.audio_gate):.4f}"


def load_audio_fusion_weights(module: AudioFusion, path: str) -> None:
    """Load ``audio_fusion.*`` weights out of a ``non_lora_trainables.bin``-style file."""
    state = torch.load(path, map_location='cpu')
    picked = {}
    for k, v in state.items():
        if 'audio_fusion.' in k:
            picked[k.split('audio_fusion.')[-1]] = v
    if not picked:
        raise ValueError(f"No audio_fusion.* keys found in {path}")
    missing, unexpected = module.load_state_dict(picked, strict=False)
    print(f"load audio_fusion: {path} (missing={list(missing)}, unexpected={list(unexpected)})")
