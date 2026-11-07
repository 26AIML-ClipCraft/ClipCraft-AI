"""Inference-only copy of UniVTG's model (https://github.com/showlab/UniVTG, model/univtg.py, MIT License, see LICENSE).

Kept: Model.forward and the layers it uses, byte-for-byte, so the released checkpoints load with strict=True and give identical
outputs (tests/test_univtg_scorer.py compares against the original code). Dropped: SetCriterion, the matcher and build_model
(training only; they pull in scipy)."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from revisionllm.editing.univtg.transformer import build_transformer
from revisionllm.editing.univtg.position_encoding import build_position_encoding

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

class Model(nn.Module):
    """ This is the UniVTG module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 input_dropout, aux_loss=False,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2

        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)

        # Conv projector
        self.span_embed = Conv(hidden_dim, hidden_dim, span_pred_dim, 3, kernel_size=3)
        self.class_embed = Conv(hidden_dim, hidden_dim, 1, 3, kernel_size=3)  # 0: background, 1: foreground

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # MLP Projector
        self.weightedpool = WeightedPool(hidden_dim)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_cls=None, src_cls_mask=None):
        bs = src_vid.shape[0]
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        if src_cls is not None:
            src_cls = self.input_txt_proj(src_cls)
        device_id = src_vid.device

        # type token.
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        if src_cls is not None:
            src_cls = src_cls + self.token_type_embeddings(torch.zeros_like(src_cls_mask.long()))

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        memory = self.transformer(src, ~mask, pos)
        vid_mem = memory[:, :src_vid.shape[1], :]  # (bsz, L_vid, d)

        outputs_class = self.class_embed(vid_mem).sigmoid()  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(vid_mem)  # (#layers, bsz, #queries, 2 or max_v_l * 2)

        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).to(device_id)
            idx_mask = idx_mask.repeat(outputs_coord.shape[0], outputs_coord.shape[1], 1)
            outputs_coord = outputs_coord * idx_mask
        else:
            raise NotImplementedError

        out = {'pred_logits': outputs_class, 'pred_spans': outputs_coord,
               'src_vid_mask': src_vid_mask}

        vid_mem_proj = src_vid

        # word-level -> sentence-level
        txt_mem_proj = self.weightedpool(src_txt, src_txt_mask).unsqueeze(1)
        sim = F.cosine_similarity(vid_mem_proj, txt_mem_proj, dim=-1) + (src_vid_mask + 1e-45).log()

        out["vid_mem_proj"] = vid_mem_proj
        out["txt_mem_proj"] = txt_mem_proj
        if src_cls is not None:
            cls_mem_proj = self.weightedpool(src_cls, src_cls_mask)
            out["cls_mem_proj"] = cls_mem_proj
        out["saliency_scores"] = sim
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Conv(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        x = x.permute(0,2,1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.permute(0, 2, 1)

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

