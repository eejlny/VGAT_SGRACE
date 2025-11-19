# desktop_cifar_graph_train.py
"""
Ultra96 PS-only equivalent (no PL) with a modern GAT pipeline and GPU optimizations.

What's inside:
- PS-only naming preserved: RPYNQ, FPYNQ_GAT (stub), GraphAttention_pynq, GAT_PYNQ.
- Mixed precision (AMP: fp16/bf16/auto), TF32 toggle, optional torch.compile.
- Tokenization options:
    * Raw 2x2 RGB patches (baseline), OR
- Graph connectivity:
    * 4-neighbor grid edges, plus optional feature-space k-NN edges.
    * DropEdge regularization.
- Training recipe:
    * RandAugment + RandomErasing, label smoothing, MixUp/CutMix, warmup+cosine LR.
    * EMA of weights.
    * TTA (flip) at evaluation.
- Pure PyTorch global mean pool and segment softmax fallback (no torch-scatter needed).

Usage (recommended CLI recipe):
  python desktop_cifar_graph_train_opt.py \
    --device cuda --amp auto --tf32 \
  --epochs 100 --batch-size 128 \
  --backbone cnn --embed-dim 96 \
  --hidden 160 --heads 6 --num-blocks 3 \
  --lr 6e-3 --weight-decay 5e-4 --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --knn 0 --edge-drop 0 \
  --mixup 0 --cutmix 0 \
  --randaug --ra-n 2 --ra-m 9 \
  --random-erasing 0.1 \
  --ema 0 --tta \
  --early-stop 0
"""

import time
import argparse
import random
from functools import lru_cache
import numpy as np
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv

# Try to import PyG's softmax; provide a pure-torch fallback if unavailable
try:
    from torch_geometric.utils import softmax as pyg_softmax
    _HAS_PYG_SOFTMAX = True
except Exception:
    pyg_softmax = None
    _HAS_PYG_SOFTMAX = False


# -----------------------------
# Utilities: seed, device, AMP
# -----------------------------
def set_seed(seed: int = 12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(arg: str = "auto"):
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        else:
            return torch.device("cpu"), "cpu"
    return torch.device(arg), arg

def resolve_amp_dtype(device, amp_flag: str):
    """
    Map --amp flag to an autocast dtype (or None). Scaler is only used for fp16 on CUDA.
      amp_flag: 'off' | 'fp16' | 'bf16' | 'auto'
    """
    amp_flag = (amp_flag or "off").lower()
    if amp_flag == "off":
        return None

    if device.type == "cuda":
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if amp_flag == "bf16":
            return torch.bfloat16 if bf16_ok else torch.float16
        if amp_flag == "fp16":
            return torch.float16
        if amp_flag == "auto":
            return torch.bfloat16 if bf16_ok else torch.float16
        return None

    if device.type == "mps":
        if amp_flag in ("fp16", "auto"):
            return torch.float16
        return None

    if device.type == "cpu" and amp_flag == "bf16":
        return torch.bfloat16
    return None


# -----------------------------
# Graph helpers
# -----------------------------
@lru_cache(maxsize=None)
def _base_edges(nH: int, nW: int) -> torch.Tensor:
    """Directed 4-neighbor edges for ONE image (on CPU). Shape [2, E]."""
    edges = []
    for r in range(nH):
        for c in range(nW):
            u = r * nW + c
            if r > 0:        edges.append([u, (r - 1) * nW + c])
            if r < nH - 1:   edges.append([u, (r + 1) * nW + c])
            if c > 0:        edges.append([u, r * nW + (c - 1)])
            if c < nW - 1:   edges.append([u, r * nW + (c + 1)])
    return torch.tensor(edges, dtype=torch.long).t()  # [2, E]

@lru_cache(maxsize=None)
def _node_xy(nH: int, nW: int) -> torch.Tensor:
    """Positional features (y,x) in [0,1] for ONE image, shape [nH*nW, 2] on CPU."""
    y = torch.arange(nH, dtype=torch.float32)
    x = torch.arange(nW, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if nH > 1: yy = yy / (nH - 1)
    if nW > 1: xx = xx / (nW - 1)
    return torch.stack([yy, xx], dim=-1).reshape(-1, 2)

def segment_softmax_torch(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Pure-Torch segment softmax over edges grouped by 'index' (dst), numerically stable.
    src  : [E] float32
    index: [E] long in [0, num_nodes-1]
    returns [E] float32
    """
    # Max per segment
    max_per = torch.full((num_nodes,), -float("inf"), device=src.device, dtype=src.dtype)
    if hasattr(max_per, "index_reduce_"):
        max_per.index_reduce_(0, index, src, reduce="amax")
    else:
        # Fallback using scatter_reduce (torch>=1.12/2.0)
        tmp = torch.full_like(max_per, -float("inf"))
        tmp = tmp.scatter_reduce(0, index, src, reduce="amax", include_self=True)
        max_per = tmp

    x = src - max_per.index_select(0, index)
    expx = torch.exp(x)
    sum_per = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
    sum_per.index_add_(0, index, expx)
    denom = sum_per.index_select(0, index) + 1e-12
    return expx / denom

def edge_softmax_float32(e: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax over incoming edges per 'dst' node, enforced float32, with PyG fallback.
    """
    if _HAS_PYG_SOFTMAX:
        return pyg_softmax(e, dst, num_nodes=num_nodes).float()
    return segment_softmax_torch(e.float(), dst, num_nodes)

def drop_edge(edge_index: torch.Tensor, p: float = 0.0, training: bool = True):
    if (not training) or p <= 0:
        return edge_index
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > p
    return edge_index[:, keep].contiguous()


# -----------------------------
# Tokenization
# -----------------------------
def build_cifar_graph(images: torch.Tensor, patch_size: int = 2, add_xy: bool = True):
    """
    Vectorized conversion of batch [B,3,32,32] to:
      x:          [B*num_nodes, 12 (+2 if add_xy)]
      edge_index: [2, B*E]  (directed 4-neighbors)
      batch:      [B*num_nodes]
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    nH = H // patch_size
    nW = W // patch_size
    num_nodes = nH * nW

    ps = patch_size
    patches = images.unfold(2, ps, ps).unfold(3, ps, ps)            # [B,C,nH,ps,nW,ps]
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()        # [B,nH,nW,C,ps,ps]
    x = patches.view(B * num_nodes, C * ps * ps)                    # [B*num_nodes, 12]

    if add_xy:
        xy_cpu = _node_xy(nH, nW)                                   # CPU fp32
        xy = xy_cpu.to(images.device, dtype=x.dtype, non_blocking=True).repeat(B, 1)
        x = torch.cat([x, xy], dim=1)                               # [B*num_nodes, 14]

    batch = torch.arange(B, device=images.device, dtype=torch.long).repeat_interleave(num_nodes)

    base_edges = _base_edges(nH, nW).to(images.device, non_blocking=True)   # [2,E]
    E = base_edges.size(1)
    offsets = (torch.arange(B, device=images.device).view(B, 1, 1) * num_nodes)
    edges_batched = base_edges.unsqueeze(0) + offsets                         # [B,2,E]
    edge_index = edges_batched.permute(1, 0, 2).reshape(2, B * E).contiguous()
    return x, edge_index, batch

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    # with residual connection
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)


        # 입력과 출력의 차원이 다른 경우 Residual connection을 위한 1x1 컨볼루션
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        out = out + x  # Residual connection 추가
        return out

class CustomMobileNet(nn.Module):
    #cfg = [32, 32, 32, (64,2), 64, 64, 64, (128,2), 128, 128, 128, 128, 128, 128, 128, 128, (256,2), 256, 256, 256, (512,2), 512]



    def __init__(self, out_channels=96):
        super(CustomMobileNet, self).__init__()
        self.cfg = [32, 32, 32, (64,2), 64, 64, 64, (128,2), 128, 128, 128, 128, 128, 128, 128, 128, (256,2), 256, 256, 256, (out_channels,2), out_channels]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_planes=16)
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가
    
        # 가중치 초기화 함수 호출
        self._initialize_weights()

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #out = F.avg_pool2d(out, 2)
        #out = out.view(out.size(0), -1)

        # 드롭아웃 추가
        out = self.dropout(out)
        out = F.relu(out)
        return out

    # 가중치 초기화 추가
    # from: https://github.com/2KangHo/mobilenet_cifar10_pytorch/blob/master/mobilenet.py
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 초기화
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 평균이 0이고 표준편차가 sqrt(2/n)인 정규 분포로 초기화
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 배치 정규화 레이어의 가중치를 1로 초기화
                m.weight.data.fill_(1)
                # 배치 정규화 레이어의 편향을 0으로 초기화
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                # FC 레이어의 가중치를 평균이 0이고 표준편차가 0.01인 정규 분포로 초기화
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class CNNBackbone2(nn.Module):
    """
    Light CNN to produce 16x16 tokens with overlap.
    Input:  [B,3,32,32] -> [B,C,16,16]
    """
    def __init__(self, out_channels=96):
        super().__init__()
        c = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, c, 3, stride=2, padding=1, bias=False),   # 32->16
            nn.BatchNorm2d(c), 
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNNBackbone(nn.Module):
    """
    Light CNN to produce 16x16 tokens with overlap.
    Input:  [B,3,32,32] -> [B,C,16,16]
    """

    def __init__(self, out_channels=96):
        super().__init__()
        c = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, c, 3, stride=2, padding=1, bias=False),   # 32->16
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def build_graph_from_backbone(images: torch.Tensor, backbone: nn.Module, add_xy: bool = True):
    """
    images[B,3,32,32] -> backbone -> feats[B,C,16,16]
    returns x_with_xy[B*H*W, C(+2)], edge_grid[2, B*E4], batch[B*H*W], hw=(H,W), feat_flat[B*H*W,C]
    """
    feats = backbone(images)                                # [B,C,16,16]
    B, C, H, W = feats.shape
    num_nodes = H * W

    feat_flat = feats.permute(0, 2, 3, 1).contiguous().view(B * num_nodes, C)  # [B*N, C]
    x = feat_flat
    if add_xy:
        y = torch.arange(H, device=images.device, dtype=x.dtype)
        z = torch.arange(W, device=images.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, z, indexing="ij")
        xy = torch.stack([yy/(H-1 if H>1 else 1), xx/(W-1 if W>1 else 1)], dim=-1).view(-1, 2)
        x = torch.cat([x, xy.repeat(B, 1)], dim=1)  # [B*N, C+2]

    base_edges = _base_edges(H, W).to(images.device, non_blocking=True)
    E = base_edges.size(1)
    offsets = (torch.arange(B, device=images.device).view(B, 1, 1) * num_nodes)
    edge_grid = (base_edges.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, B * E).contiguous()

    batch = torch.arange(B, device=images.device, dtype=torch.long).repeat_interleave(num_nodes)
    return x, edge_grid, batch, (H, W), feat_flat


def add_knn_edges(feat_flat: torch.Tensor, batch: torch.Tensor, hw, k=8):
    """
    k-NN edges in feature space, computed per-image for memory efficiency.
    feat_flat: [B*H*W, C]  (do NOT include positional (x,y) in features)
    batch    : [B*H*W]
    hw       : (H, W)
    returns edge_index_knn [2, B*k*H*W] (directed)
    """
    if k <= 0:
        return None
    H, W = hw
    N_per = H * W
    B = int(batch.max().item()) + 1
    rows = []
    cols = []
    xn = F.normalize(feat_flat, dim=1)
    for b in range(B):
        s = b * N_per
        e = s + N_per
        xb = xn[s:e]                                    # [N_per, C]
        sim = xb @ xb.T                                 # [N_per, N_per] cosine sim
        sim.fill_diagonal_(-1e9)
        idx = sim.topk(k, dim=1).indices                # [N_per, k]
        src = torch.arange(N_per, device=feat_flat.device).unsqueeze(1).expand_as(idx).reshape(-1)
        dst = idx.reshape(-1)
        rows.append(src + s)
        cols.append(dst + s)
    src_all = torch.cat(rows)
    dst_all = torch.cat(cols)
    return torch.stack([src_all, dst_all], dim=0).contiguous()


# -----------------------------
# PS-only stubs to keep names
# -----------------------------
class RPYNQ(torch.autograd.Function):
    """PS-only ReLU-like autograd demo (kept for name compatibility)."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clone()
    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[inp <= 0] = 0
        return grad

class FPYNQ_GAT(torch.autograd.Function):
    """Placeholder; real 'acceleration' replaced with the PS-only module below."""
    @staticmethod
    def forward(ctx, self_ref, adj_or_edge_index, input_, weights, attention, out_features, dropout):
        raise RuntimeError("FPYNQ_GAT forward is a placeholder in PS-only build; use GraphAttention_pynq().")
    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 7


# -----------------------------
# GAT layers (AMP-safe)
# -----------------------------
class GraphAttentionHead(nn.Module):
    """
    Single-head GAT (edge-wise, sparse).
    - Attention logits & softmax computed in fp32 for stability.
    - Alpha cast back to message dtype before aggregation (avoids dtype mismatch).
    """
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.1, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.W   = nn.Linear(in_features, out_features, bias=False)
        self.a_l = nn.Parameter(torch.empty(out_features))
        self.a_r = nn.Parameter(torch.empty(out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_p = dropout
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        with torch.no_grad():
            fan_in = out_features
            bound = np.sqrt(6.0 / (fan_in + 1.0))
            self.a_l.uniform_(-bound, bound)
            self.a_r.uniform_(-bound, bound)

    def forward(self, x, edge_index):
        # x: [N, Fin], edge_index: [2, E] (src, dst)
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        Wh = self.W(x)  # [N, Fout], dtype may be fp32/fp16/bf16

        # attention in fp32
        Wh_f = Wh.float()
        e = self.leakyrelu(
            (Wh_f[src] * self.a_l.float()).sum(dim=1) + (Wh_f[dst] * self.a_r.float()).sum(dim=1)
        )  # [E], float32

        alpha_f = edge_softmax_float32(e, dst, num_nodes=N)  # float32
        if self.training and self.dropout_p > 0:
            alpha_f = F.dropout(alpha_f, p=self.dropout_p, training=True)
        alpha = alpha_f.to(Wh.dtype)  # cast back to message dtype

        out = torch.zeros((N, self.out_features), device=x.device, dtype=Wh.dtype)
        out.index_add_(0, dst, alpha.unsqueeze(1) * Wh[src])

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class MultiHeadGAT(nn.Module):
    """Multi-head GAT using the PS-only heads above."""
    def __init__(self, in_features, out_features, heads=4, alpha=0.2, dropout=0.1, concat=True, bias=True):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionHead(in_features, out_features, alpha=alpha, dropout=dropout, bias=bias)
            for _ in range(max(1, heads))
        ])
        self.concat = concat

    def forward(self, x, edge_index):
        outs = [h(x, edge_index) for h in self.heads]
        return torch.cat(outs, dim=1) if self.concat else torch.stack(outs, dim=0).mean(dim=0)


class NormDropRes(nn.Module):
    """BatchNorm + ReLU + Dropout + Residual wrapper (if dims match)."""
    def __init__(self, dim, p_drop=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x, y):
        y = self.bn(y)            # runs in fp32 under autocast
        y = F.relu(y, inplace=True)
        y = self.drop(y)
        if x.shape == y.shape:
            return x + y
        return y


def global_mean_pool_torch(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch global mean pool over graph batches.
    x:     [N, F]
    batch: [N] with graph ids in [0, B-1]
    returns: [B, F]
    """
    if batch is None or batch.numel() == 0:
        return x.mean(dim=0, keepdim=True)
    B = int(batch.max().item()) + 1
    out = torch.zeros((B, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    count = torch.bincount(batch, minlength=B).clamp(min=1).to(x.dtype).unsqueeze(1)
    return out / count


class GraphAttention_pynq(nn.Module):
    """PS-only 'hardware' GAT layer interface preserved."""
    def __init__(self, in_features, out_features, head_count=4, alpha=0.2, dropout=0.1, concat=True, bias=True):
        super().__init__()
        #self.gat = MultiHeadGAT(in_features, out_features, heads=head_count,
        #                        alpha=alpha, dropout=dropout, concat=concat, bias=bias)
        self.gat = GATConv(in_features, out_features, heads=head_count,
                                negative_slope=alpha, add_self_loops= False, dropout=dropout, concat=concat, bias=bias)

        #self.gat = SAGEConv(in_features, out_features, add_self_loops= False, bias=bias)
        #self.gat = SAGEConv(in_features, out_features, bias=bias)

    def forward(self, acc_flag, dense_flag, relu_flag, x, edge_index, nnz_adj,
                rowPtr_fea_buffer=None, columnIndex_fea_buffer=None, values_fea_buffer=None,
                rowPtr_adj_buffer=None, columnIndex_adj_buffer=None, values_adj_buffer=None,
                B_buffer=None, D_buffer=None, E_buffer=None, S_buffer=None):
        out = self.gat(x, edge_index)
        return F.relu(out, inplace=True) if relu_flag else out


class RPYNQ_ReLU(nn.Module):
    """Wrapper using the RPYNQ autograd hook (kept for name parity)."""
    def forward(self, x):
        return RPYNQ.apply(x)


class GraphAttentionCPU(nn.Module):
    """Single-head GAT block with residual, norm, and dropout."""
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.1, residual=True):
        super().__init__()
        #self.att = MultiHeadGAT(in_features, out_features, heads=1, alpha=alpha, dropout=dropout, concat=True, bias=True)
        self.att = GATConv(in_features, out_features, heads=1,
                                negative_slope=alpha, add_self_loops= False, dropout=dropout, concat=True, bias=True)
        self.ndr = NormDropRes(out_features, p_drop=dropout)
        self.residual = residual
    def forward(self, x, edge_index):
        y = self.att(x, edge_index)
        #return y
        base = x if self.residual and x.shape == y.shape else torch.zeros_like(y)
        return self.ndr(base, y)


# -----------------------------
# Model (PS layout preserved)
# -----------------------------
class GAT_PYNQ(nn.Module):
    """
    1) "Hardware" GAT layer (multi-head, concat) + ReLU
    2) Projection GAT block to hidden
    3) (num_blocks-1) residual GAT blocks at hidden
    4) Global mean pool + MLP head
    """
    def __init__(self, hidden_channels, head_count, in_features, out_classes,
                 dropout=0.2, alpha=0.2, num_blocks=3):
        super().__init__()
        self.att_pynq = GraphAttention_pynq(in_features, hidden_channels,
                                            head_count=head_count, alpha=alpha, dropout=dropout,
                                            concat=True, bias=True)
        self.relu_pynq = nn.ReLU() #RPYNQ_ReLU()


        self.proj = GraphAttentionCPU(hidden_channels * head_count, hidden_channels,
                                      alpha=alpha, dropout=dropout, residual=True)
        blocks = []
        for _ in range(max(0, num_blocks - 1)):
            blocks.append(GraphAttentionCPU(hidden_channels, hidden_channels,
                                            alpha=alpha, dropout=dropout, residual=True))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_classes)
        )

    def forward(self, acc_flag, x, edge_index, batch,
                rowPtr_fea_buffer=None, columnIndex_fea_buffer=None, values_fea_buffer=None,
                rowPtr_adj_buffer=None, columnIndex_adj_buffer=None, values_adj_buffer=None,
                B_buffer=None, D_buffer=None, E_buffer=None, S_buffer=None):
        def isSparse(array, m, n): 
         counter = 0
         # Count number of zeros
         # in the matrix
         for i in range(0, m):
            for j in range(0, n):
               if (array[i][j] == 0):
                   counter = counter + 1
         print("total values ",m*n)
         print("zero values ",counter)
         return (counter > ((m * n) // 2))
        
        nnz_adj = edge_index.size(1) if edge_index is not None else 0
        dense = 0
        relu  = 1
        #print("input shape")
        #print(x.shape)

        #print("input sparsity")
        #isSparse(x, x.shape[0], x.shape[1])

        #print("edge index shape")
        #print(edge_index.shape)

        x1 = self.att_pynq(acc_flag, dense, relu, x, edge_index, nnz_adj)
        x1 = self.relu_pynq(x1)
        h = self.proj(x1, edge_index)
        for blk in self.blocks:
            h = blk(h, edge_index)
        if batch is not None:
            h = global_mean_pool_torch(h, batch)
        return self.head(h)


# -----------------------------
# Training helpers: MixUp/CutMix, EMA, TTA
# -----------------------------
def rand_bbox(W, H, lam):
    cut_w = int(W * (1 - lam) ** 0.5)
    cut_h = int(H * (1 - lam) ** 0.5)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(images, labels, mixup_alpha=0.2, cutmix_alpha=0.1, p_cutmix=0.5):
    B, C, H, W = images.shape
    perm = torch.randperm(B, device=images.device)
    images2 = images[perm]
    labels2 = labels[perm]

    if torch.rand(1, device=images.device).item() < p_cutmix and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        images[:, :, y1:y2, x1:x2] = images2[:, :, y1:y2, x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1) / (W*H))
    elif mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images2
    else:
        lam = 1.0

    return images, labels, labels2, lam

class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = None
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        new_state = self.backup.copy()
        new_state.update(self.shadow)
        model.load_state_dict(new_state, strict=False)
    @torch.no_grad()
    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None


# -----------------------------
# Train / Eval loops
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, build_graph_fn, autocast_kwargs=None):
    model.eval()
    total = 0
    correct = 0
    autocast_kwargs = autocast_kwargs or {"enabled": False}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(**autocast_kwargs):
            x, edge_index, batch = build_graph_fn(images, training=False)
            logits = model(0, x, edge_index, batch)
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return correct / max(1, total)

@torch.no_grad()
def evaluate_tta(model, loader, device, build_graph_fn, autocast_kwargs=None, use_tta=True):
    model.eval()
    total = 0; correct = 0
    autocast_kwargs = autocast_kwargs or {"enabled": False}
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = []
        flips = [False, True] if use_tta else [False]
        for flip in flips:
            imgs = torch.flip(images, dims=[3]) if flip else images
            with torch.autocast(**autocast_kwargs):
                x, edge_index, batch = build_graph_fn(imgs, training=False)
                logits = model(0, x, edge_index, batch)
            preds.append(logits)
        logits = torch.stack(preds).mean(0)
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return correct / max(1, total)

def train_one_epoch(model, loader, device, optimizer, criterion, scheduler,
                    build_graph_fn, grad_clip=1.0, scaler=None, autocast_kwargs=None,
                    mixup_alpha=0.2, cutmix_alpha=0.1):
    model.train()
    running = 0.0
    autocast_kwargs = autocast_kwargs or {"enabled": False}
    use_scaler = bool(scaler) and getattr(scaler, "is_enabled", lambda: False)()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp/CutMix (image-space) before tokenization
        do_mix = (mixup_alpha > 0) or (cutmix_alpha > 0)
        if do_mix:
            images, labels1, labels2, lam = apply_mixup_cutmix(
                images, labels, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, p_cutmix=0.5
            )

        with torch.autocast(**autocast_kwargs):
            x, edge_index, batch = build_graph_fn(images, training=True)
            logits = model(0, x, edge_index, batch)
            if do_mix:
                loss = lam * criterion(logits, labels1) + (1 - lam) * criterion(logits, labels2)
            else:
                loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running += loss.item()

    return running / len(loader)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # Core & model
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--attn-drop", type=float, default=0.1)  # kept as 'drop' for alpha probs in heads
    ap.add_argument("--out-classes", type=int, default=10)

    # Tokenization
    ap.add_argument("--backbone", choices=["none", "cnn"], default="cnn")
    ap.add_argument("--embed-dim", type=int, default=96)
    ap.add_argument("--patch-size", type=int, default=2)  # used when backbone=none
    ap.add_argument("--no-xy", action="store_true", help="disable (y,x) positional features")

    # Graph connectivity
    ap.add_argument("--knn", type=int, default=8, help="k for feature-space k-NN edges (0 to disable)")
    ap.add_argument("--edge-drop", type=float, default=0.05, help="DropEdge prob during training")

    # Optim & schedule
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--save", type=str, default="cifar10_ultra96_ps_best.pth")
    ap.add_argument("--ema", type=float, default=0.9995, help="EMA decay (0 disables)")

    # Data aug
    ap.add_argument("--randaug", action="store_true", help="Enable RandAugment")
    ap.add_argument("--ra-n", type=int, default=2)
    ap.add_argument("--ra-m", type=int, default=9)
    ap.add_argument("--random-erasing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.1)

    # AMP/compile/TF32/TTA
    ap.add_argument("--amp", choices=["off", "fp16", "bf16", "auto"], default="auto")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tta", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    device, device_str = pick_device(args.device)

    # Performance knobs
    if device_str == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    amp_dtype = resolve_amp_dtype(device, args.amp)
    autocast_kwargs = dict(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None))

    # GradScaler (new API; fallback to old if needed)
    scaler = None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device_str == "cuda" and amp_dtype == torch.float16))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device_str == "cuda" and amp_dtype == torch.float16))

    print(f"[Device] {device_str.upper()}  |  AMP: {('off' if amp_dtype is None else str(amp_dtype).split('.')[-1])}"
          f"{'  |  TF32:on' if (device_str=='cuda' and args.tf32) else ''}"
          f"{'  |  compile:on' if args.compile else ''}"
          f"{'  |  PyG softmax:' + ('on' if _HAS_PYG_SOFTMAX else 'fallback')}")

    # ----------------- Data -----------------
    # Train transform
    train_transforms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]
    if args.randaug and hasattr(T, "RandAugment"):
        train_transforms.append(T.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m))
    train_transforms.extend([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    if args.random_erasing > 0:
        train_transforms.append(T.RandomErasing(p=args.random_erasing, scale=(0.02, 0.2),
                                               ratio=(0.3, 3.3), value='random'))
    transform_train = T.Compose(train_transforms)

    # Test transform (NO random aug)
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    root = "./cifar_data"
    train_set = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    pin_mem = (device_str == "cuda")
    persistent = args.workers > 0
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_mem, persistent_workers=persistent,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_mem, persistent_workers=persistent,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        drop_last=False,
    )

    # ----------------- Model -----------------
    add_xy = (not args.no_xy)
    print("add_xy is: ", add_xy)
    if args.backbone == "cnn":
        in_features = args.embed_dim + (0 if not add_xy else 2)
    else:
        in_features = (12 if args.patch_size == 2 else (3 * args.patch_size * args.patch_size)) + (0 if not add_xy else 2)

    model = GAT_PYNQ(
        hidden_channels=args.hidden, head_count=args.heads,
        in_features=in_features, out_classes=args.out_classes,
        dropout=args.drop, alpha=args.alpha, num_blocks=args.num_blocks
    ).to(device)

    # Optional torch.compile
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[compile] disabled due to: {e}")

    # Backbone (if used)
    backbone = None
    if args.backbone == "cnn":
        #backbone = timm.create_model("mobilenetv1_100.ra4_e3600_r224_in1k", pretrained=True)
        backbone = CNNBackbone2(out_channels=args.embed_dim).to(device)
        #backbone = CustomMobileNet(out_channels=args.embed_dim).to(device)
        #model_path = "/media/josnu02/hd1/josnu02/cuda_performance/vision_transformer_cifar10/ini_weights.pkt" #load best model
        #backbone.load_state_dict(torch.load(model_path),strict=False)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optim & Scheduler
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + ([] if backbone is None else list(backbone.parameters())),
        lr=args.lr, weight_decay=args.weight_decay
    )

    #optimizer = torch.optim.AdamW(
    #    list(model.parameters()),
    #    lr=args.lr, weight_decay=args.weight_decay
    #)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * max(1, args.warmup_epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # EMA
    ema = ModelEMA(model, decay=args.ema) if args.ema and args.ema > 0 else None

    # Graph builder (captures args/backbone)
    def build_graph_fn(images, training: bool):
        def isSparse(array, m, n): 
         counter = 0
         # Count number of zeros
         # in the matrix
         for i in range(0, m):
            for j in range(0, n):
               if (array[i][j] == 0):
                   counter = counter + 1
         print("total values ",m*n)
         print("zero values ",counter)
         return (counter > ((m * n) // 2))
        if args.backbone == "cnn":
            x, edge_grid, batch, hw, feat_flat = build_graph_from_backbone(images, backbone, add_xy=add_xy)
            edge_index = edge_grid
            if args.knn and args.knn > 0:
                edge_knn = add_knn_edges(feat_flat, batch, hw, k=args.knn)
                if edge_knn is not None:
                    edge_index = torch.cat([edge_index, edge_knn], dim=1).contiguous()
        else:
            x, edge_index, batch = build_cifar_graph(images, patch_size=args.patch_size, add_xy=add_xy)

        if training and args.edge_drop and args.edge_drop > 0:
            edge_index = drop_edge(edge_index, p=args.edge_drop, training=True)
        #print("Num of nodes in graph: ", x.shape[0]) 
        #print("Num of features in node: ", x.shape[1])
        #print("Num of edges in graph: ", edge_index.shape[1])    
        #print("Sparsity in input: ")
        #isSparse(x, x.shape[0], x.shape[1])
        return x, edge_index, batch

    # ----------------- Train -----------------
    best_acc = 0.0
    best_epoch = 0
    patience = args.early_stop
    start_time = time.time()

    cfg_str = f"epochs={args.epochs} batch={args.batch_size} hidden={args.hidden} heads={args.heads} " \
              f"blocks={args.num_blocks} tok={'cnn' if args.backbone=='cnn' else f'patch{args.patch_size}'} " \
              f"knn={args.knn} edge_drop={args.edge_drop} mixup={args.mixup} cutmix={args.cutmix} xy={add_xy}"
    print(f"[Train] {cfg_str}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model, loader=train_loader, device=device, optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, build_graph_fn=build_graph_fn, grad_clip=1.0, scaler=scaler,
            autocast_kwargs=autocast_kwargs, mixup_alpha=args.mixup, cutmix_alpha=args.cutmix
        )
        if ema is not None:
            ema.update(model)

        # Train accuracy (quick, no TTA)
        train_acc = evaluate(model, train_loader, device, build_graph_fn, autocast_kwargs=autocast_kwargs)

        # Eval (EMA & TTA if requested)
        if ema is not None:
            ema.apply_to(model)
        if args.tta:
            test_acc = evaluate_tta(model, test_loader, device, build_graph_fn, autocast_kwargs=autocast_kwargs, use_tta=True)
        else:
            test_acc = evaluate(model, test_loader, device, build_graph_fn, autocast_kwargs=autocast_kwargs)
        if ema is not None:
            ema.restore(model)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | lr {optimizer.param_groups[0]['lr']:.6f} | "
              f"loss {train_loss:.4f} | train {train_acc:.4f} | test {test_acc:.4f} | {dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            if ema is not None:
                # Save EMA weights
                ema.apply_to(model)
                torch.save(model.state_dict(), args.save)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), args.save)
            print(f"  ✓ New best test acc {best_acc:.4f}. Saved to {args.save}")

        if patience and (epoch - best_epoch) >= patience:
            print(f"[Early Stop] No improvement for {patience} epochs (best at {best_epoch}).")
            break

    total_min = (time.time() - start_time) / 60
    print(f"[Done] Best test acc: {best_acc:.4f} (epoch {best_epoch}) | total {total_min:.1f} min")


if __name__ == "__main__":
    main()
