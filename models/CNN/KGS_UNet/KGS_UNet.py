import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from .kan import KANLinear


def _gn(groups: int, ch: int) -> nn.GroupNorm:
    if groups <= 0:
        groups = 1
    if ch % groups != 0:
        g = math.gcd(ch, groups)
        groups = g if g > 0 else 1
    return nn.GroupNorm(groups, ch)

def _safe(x: torch.Tensor, lo: float = -6.0, hi: float = 6.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo).clamp_(lo, hi)

class UpRefineGN(nn.Module):
    def __init__(self, ch: int, scale: int = 2, groups: int = 4, depthwise: bool = True):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        g = ch if depthwise else 1
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=g, bias=False),
            _gn(groups, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
            _gn(groups, ch),
        )
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ups(x)
        alpha = 0.5 * torch.tanh(self.scale)
        return y + alpha * self.refine(y)

class DSConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=4, depthwise=True, dilation=1, beta=1.0):
        super().__init__()
        self.cv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.n1  = _gn(groups, cout)
        if depthwise:
            self.cv2 = nn.Conv2d(cout, cout, 3, 1, dilation, groups=cout, bias=False)
            self.pw  = nn.Conv2d(cout, cout, 1, 1, 0, bias=False)
        else:
            self.cv2 = nn.Conv2d(cout, cout, 3, 1, dilation, bias=False)
            self.pw  = None
        self.n2  = _gn(groups, cout)
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x0):
        s = self.skip(x0)
        x = self.act(self.n1(self.cv1(x0)))
        x = self.cv2(x)
        if self.pw is not None:
            x = self.pw(x)
        x = self.n2(x)
        return self.act(x + self.beta * s)
class SplineCrossAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4, dim_head: int = 32,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.h, self.dh = heads, dim_head
        inner = heads * dim_head
        self.to_q = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_k = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_v = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_o = nn.Conv2d(inner, ch, 1, bias=False)
        self.kan  = KANLinear(1, 1, grid_size=grid_size, spline_order=spline_order,
                             grid_range=[-6, 6])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        q = self.to_q(x).view(B, self.h, self.dh, N).transpose(-2, -1)  # [B,h,N,dh]
        k = self.to_k(x).view(B, self.h, self.dh, N)                    # [B,h,dh,N]
        v = self.to_v(x).view(B, self.h, self.dh, N).transpose(-2, -1)  # [B,h,N,dh]
        sim = (q @ k) / math.sqrt(self.dh)                               # [B,h,N,N]
        s   = self.kan(_safe(sim.reshape(-1, 1))).view_as(sim)           # 样条整形
        attn= s.softmax(dim=-1)
        out = attn @ v                                                   # [B,h,N,dh]
        out = out.transpose(-2, -1).contiguous().view(B, self.h*self.dh, H, W)
        return self.to_o(out)

class KANSE(nn.Module):
    def __init__(self, ch: int, r: int = 8, grid_size: int = 5, spline_order: int = 3, min_ch: int = 16):
        super().__init__()
        hidden = max(min_ch, ch // r)

        self.fc1 = KANLinear(ch,     hidden, grid_size=grid_size, spline_order=spline_order,
                             grid_range=[-6, 6])
        self.fc2 = KANLinear(hidden, ch,   grid_size=grid_size, spline_order=spline_order,
                             grid_range=[-6, 6])
    def forward(self, x):
        b,c,_,_ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).reshape(b, c)
        s = self.fc2(torch.nn.SiLU(inplace=True)(self.fc1(torch.nan_to_num(s))))
        return x * torch.sigmoid(s).view(b,c,1,1)
class KCB(nn.Module):
    def __init__(
        self,
        ch: int,
        heads: int = 4,
        dim_head: int = 32,
        grid_size: int = 5,
        spline_order: int = 3,
        use_pos: bool = True,
        use_chan: bool = True,
        se_ratio: int = 8,
    ):
        super().__init__()
        self.use_pos = use_pos
        self.use_chan = use_chan

        if use_pos:
            self.pos_attn = SplineCrossAttention(
                ch=ch,
                heads=heads,
                dim_head=dim_head,
                grid_size=grid_size,
                spline_order=spline_order,
            )
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("alpha", None)
        if use_chan:
            self.chan_attn = KANSE(
                ch=ch,
                r=se_ratio,
                grid_size=grid_size,
                spline_order=spline_order,
            )
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        if self.use_pos:
            out = out + self.alpha * self.pos_attn(x)
        if self.use_chan:
            out = out + self.beta * self.chan_attn(x)
        return out


class AttnDown(nn.Module):
    def __init__(self, ch: int, groups: int = 4, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1, bias=False)
        self.gn   = nn.GroupNorm(max(1, min(groups, ch)), ch)
        self.act  = nn.SiLU(inplace=True)
        self.catt = KANSE(ch, r=8, grid_size=grid_size, spline_order=spline_order)
        self.sq   = nn.Conv2d(ch, 1, 1, 1, 0, bias=False)                            # 空间打分
        self.kan  = KANLinear(1, 1, grid_size=grid_size, spline_order=spline_order,
                              grid_range=[-6, 6])
        self.g_floor = nn.Parameter(torch.tensor(0.10))
        self._max_floor = 0.40

    def forward(self, x):
        y = self.act(self.gn(self.conv(x)))
        y = self.catt(y)
        s = self.sq(y)
        B,_,H,W = s.shape
        g = self.kan(torch.nan_to_num(s).reshape(B*H*W, 1)).view(B,1,H,W)
        g = torch.sigmoid(g)
        floor = self.g_floor.clamp(0.0, self._max_floor)
        g = floor + (1.0 - floor) * g
        return y * g


class SAGateKAN(nn.Module):
    def __init__(self, c_x: int, c_g: int, d: int = 16, grid_size: int = 5,
                 spline_order: int = 3, dropout: float = 0.0, grid_update_every: int = 256):
        super().__init__()
        self.theta = nn.Conv2d(c_x, d, 1, bias=False)
        self.phi   = nn.Conv2d(c_g, d, 1, bias=False)
        self.mix   = nn.Conv2d(d*4, d, 1, bias=False)
        self.dp    = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.kan   = KANLinear(d, 1, grid_size=grid_size, spline_order=spline_order,
                               grid_range=[-6, 6])
        self.last_gate: Optional[torch.Tensor] = None
        self.last_token: Optional[torch.Tensor] = None
        self._grid_update_every = int(grid_update_every) if grid_update_every else 0
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2:] != g.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        tx = self.theta(x)
        tg = self.phi(g)
        z  = torch.cat([tx, tg, (tx - tg).abs(), tx * tg], dim=1)
        z  = self.dp(self.mix(z))

        B, d, H, W = z.shape
        z_lin = z.permute(0, 2, 3, 1).reshape(B*H*W, d)
        if self._grid_update_every and self.training:
            self._step += 1
            if int(self._step.item()) % self._grid_update_every == 0:
                try:
                    self.kan.update_grid(z_lin.detach())
                except Exception:
                    pass

        a = torch.sigmoid(self.kan(torch.nan_to_num(z_lin, nan=0.0, posinf=6.0, neginf=-6.0))).view(B, H, W, 1)
        a = a.permute(0, 3, 1, 2).contiguous()
        self.last_gate, self.last_token = a, z
        return x * a, a

    def monotone_reg(self, lam: float = 1e-4, scale: float = 2.0, eps: float = 1e-3) -> torch.Tensor:
        if self.last_token is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        z = self.last_token
        B, d, H, W = z.shape
        z_lin = z.permute(0, 2, 3, 1).reshape(B*H*W, d)
        q = d // 4
        idx = slice(2*q, 3*q)  # |θ-φ|
        z_aug = z_lin.clone()
        z_aug[:, idx] = z_aug[:, idx] * (1.0 + scale * eps)
        a0 = torch.sigmoid(self.kan(torch.nan_to_num(z_lin, nan=0.0, posinf=6.0, neginf=-6.0)))
        a1 = torch.sigmoid(self.kan(torch.nan_to_num(z_aug, nan=0.0, posinf=6.0, neginf=-6.0)))
        penalty = (a1 - a0).clamp(max=0).abs().mean()
        return lam * penalty

class KGS(SAGateKAN):
    def __init__(self, c_x: int, c_g: int, d: int, grid_size: int = 5,
                 spline_order: int = 3, dropout: float = 0.0):
        super().__init__(c_x, c_g, d, grid_size, spline_order, dropout)
        self.noise_estimator = nn.Conv2d(d, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2:] != g.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        tx = self.theta(x)
        tg = self.phi(g)
        z = torch.cat([tx, tg, (tx - tg).abs(), tx * tg], dim=1)
        z = self.dp(self.mix(z))

        noise = torch.sigmoid(self.noise_estimator(z))
        dynamic_floor = 0.1 + 0.3 * noise
        B, d, H, W = z.shape
        z_lin = z.permute(0, 2, 3, 1).reshape(B*H*W, d)
        a = torch.sigmoid(self.kan(torch.nan_to_num(z_lin))).view(B, H, W, 1).permute(0, 3, 1, 2)

        a = dynamic_floor + (1 - dynamic_floor) * a
        self.last_gate, self.last_token = a, z
        return x * a, a

class LiteSPP(nn.Module):
    def __init__(self, ch, r=(1,2,4)):
        super().__init__()
        self.proj = nn.Conv2d(ch * (len(r)+1), ch, 1, bias=False)
        self.norm = nn.GroupNorm( min(8, math.gcd(ch, 8)) or 1, ch )
        self.act  = nn.SiLU(inplace=True)
        self.r = r
    def forward(self, x):
        feats = [x]
        for k in self.r:
            p = F.adaptive_avg_pool2d(x, (max(1, x.size(2)//k), max(1, x.size(3)//k)))
            p = F.interpolate(p, size=x.shape[-2:], mode='bilinear', align_corners=False)
            feats.append(p)
        y = self.proj(torch.cat(feats, dim=1))
        return self.act(self.norm(y))
class MultiScaleSEM(nn.Module):
    def __init__(self, channels: int,
                 dilation_rates: Tuple[int, ...] = (1, 2, 4),
                 reduction: int = 8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(
                channels, channels,
                kernel_size=3,
                padding=d,
                dilation=d,
                groups=channels,
                bias=False
            )
            for d in dilation_rates
        ])
        self.pointwise = nn.Conv2d(
            channels * len(dilation_rates),
            channels,
            kernel_size=1,
            bias=False
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_ch = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        feats = [branch(x) for branch in self.branches]
        ms = torch.cat(feats, dim=1)
        ms = self.pointwise(ms)


        w = self.avg_pool(ms)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))

        return x * w + x

from torch_geometric.nn import GATv2Conv


class GraphEnhancedMSSEM(nn.Module):
    def __init__(self, channels: int, dilation_rates=(1, 2, 4), reduction=8):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d)
            for d in dilation_rates
        ])
        self.pointwise = nn.Conv2d(channels * len(dilation_rates), channels, kernel_size=1)
        self.node_proj = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.graph_attn = GATv2Conv(
            in_channels=channels // 4,
            out_channels=channels // 4,
            heads=1,
            concat=False
        )
        self.graph_channel_up = nn.Conv2d(channels // 4, channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def _build_adj(self, H: int, W: int, device) -> torch.Tensor:
        N = H * W
        adj = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if i > 0: adj.append([idx, (i - 1) * W + j])
                if i < H - 1: adj.append([idx, (i + 1) * W + j])
                if j > 0: adj.append([idx, i * W + (j - 1)])
                if j < W - 1: adj.append([idx, i * W + (j + 1)])
        adj = torch.tensor(adj, dtype=torch.long, device=device).T
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        feats = [branch(x) for branch in self.branches]
        ms = torch.cat(feats, dim=1)
        ms = self.pointwise(ms)
        nodes = self.node_proj(x).view(B, C // 4, -1).transpose(1, 2)
        data_list = []
        for i in range(B):

            adj = self._build_adj(H, W, x.device)
            data = Batch(x=nodes[i], edge_index=adj)
            data_list.append(data)
        batch_data = Batch.from_data_list(data_list)

        graph_feat_batch = self.graph_attn(batch_data.x, batch_data.edge_index)
        graph_feat = graph_feat_batch.view(B, N, C // 4)
        graph_feat = graph_feat.transpose(1, 2).view(B, C // 4, H, W)
        graph_feat = self.graph_channel_up(graph_feat)
        ms = ms + graph_feat
        w = self.avg_pool(ms).view(B, C)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(B, C, 1, 1)

        return x * w + x


class EdgeAttentionHead(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            _gn(groups=4, ch=mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_logits = self.conv(x)
        edge_mask   = torch.sigmoid(edge_logits)
        x_refine    = x * (1.0 + edge_mask)
        return x_refine, edge_logits

class TextureEdgeAttentionHead(EdgeAttentionHead):
    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__(in_ch, mid_ch)
        self.texture_conv = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        nn.init.xavier_uniform_(self.texture_conv.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_logits = self.conv(x)
        texture_feat = self.texture_conv(x)
        texture_conf = torch.sigmoid(F.adaptive_avg_pool2d(texture_feat, 1).mean(dim=1, keepdim=True))
        edge_mask = torch.sigmoid(edge_logits) * (1 + 0.5 * texture_conf)
        x_refine = x * (1.0 + edge_mask)
        return x_refine, edge_logits

class ClassBalancedHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, grid_size: int = 5):
        super().__init__()
        self.head = nn.Conv2d(in_ch, num_classes, 1)
        self.kan_balance = KANLinear(num_classes, num_classes, grid_size=grid_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        if self.training:
            prob = torch.softmax(logits, dim=1)
            avg_prob = prob.mean(dim=(2, 3))
            balance_weight = torch.sigmoid(self.kan_balance(avg_prob))
            logits = logits * balance_weight.unsqueeze(-1).unsqueeze(-1)
        return logits
    @property
    def weight(self):
        return self.head.weight
    @property
    def bias(self):
        return self.head.bias
class AttentionScheduler(nn.Module):
    def __init__(self, in_ch: int, num_attention: int = 3, grid_size: int = 5):
        super().__init__()
        self.stat_proj = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.kan_scheduler = KANLinear(16, num_attention, grid_size=grid_size)

    def forward(self, x: torch.Tensor, attentions: List[torch.Tensor]) -> torch.Tensor:
        stats = self.stat_proj(x)  # [B, 16]
        weights = torch.softmax(self.kan_scheduler(stats), dim=1)
        fused = sum(w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * att
                   for w, att in zip(weights.unbind(1), attentions))
        return fused

class KGS_UNet(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 num_classes: int = 2,
                 base_ch: int = 64,
                 groups: int = 4,
                 depthwise: bool = True,
                 gate_dims: Tuple[int, int, int, int] = (4, 8, 12, 16),
                 gate_dropout: float = 0.05,
                 kan_grid_size: int = 5,
                 kan_spline_order: int = 3,
                 use_ssa: bool = False,
                 ssa_heads: int = 4,
                 ssa_dim_head: int = 32,
                 ssa_grid_size: int = 5,
                 ssa_spline_order: int = 3,
                 deep_supervision: bool = False,
                 logit_clip: float = 50.0,
                 use_ms_sem: bool = True,
                 use_edge_attn: bool = False
                 ):
        super().__init__()

        C = base_ch
        c1, c2, c3, c4, c5 = C, 2*C, 4*C, 8*C, 16*C

        # ---- Encoder ----

        self.down1 = AttnDown(c1, groups=groups)
        self.down2 = AttnDown(c2, groups=groups)
        self.down3 = AttnDown(c3, groups=groups)
        self.down4 = AttnDown(c4, groups=groups)

        self.enc1 = ConvBlock(in_ch, c1, groups=groups, depthwise=depthwise)
        self.enc2 = ConvBlock(c1, c2, groups=groups, depthwise=depthwise)
        self.enc3 = ConvBlock(c2, c3, groups=groups, depthwise=depthwise)
        self.enc4 = ConvBlock(c3, c4, groups=groups, depthwise=depthwise)
        self.enc5 = ConvBlock(c4, c5, groups=groups, depthwise=depthwise)

        self.dec4 = ConvBlock(c4 + c5, c4, groups=groups, depthwise=depthwise)
        self.dec3 = ConvBlock(c3 + c4, c3, groups=groups, depthwise=depthwise)
        self.dec2 = ConvBlock(c2 + c3, c2, groups=groups, depthwise=depthwise)
        self.dec1 = ConvBlock(c1 + c2, c1, groups=groups, depthwise=depthwise)
        # ---- Bottleneck ----
        self.ssa = KCB(
            ch=c5,
            heads=ssa_heads,
            dim_head=ssa_dim_head,
            grid_size=ssa_grid_size,
            spline_order=ssa_spline_order,
            use_pos=use_ssa,
            use_chan=True,
            se_ratio=8,
        )
        self.ssa_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_ssa_enabled", torch.tensor(1.0 if use_ssa else 0.0))
        self.spp = LiteSPP(c5, r=(2, 4, 8))
        self.use_ms_sem = bool(use_ms_sem)
        if self.use_ms_sem:
            self.ms_sem = GraphEnhancedMSSEM(c5, dilation_rates=(1, 2, 4), reduction=8)
        else:
            self.ms_sem = nn.Identity()
        # ---- Decoder + SAGate-KAN ----
        g1, g2, g3, g4 = gate_dims
        self.up4 = UpRefineGN(c5, groups=groups, depthwise=depthwise)
        self.up3 = UpRefineGN(c4, groups=groups, depthwise=depthwise)
        self.up2 = UpRefineGN(c3, groups=groups, depthwise=depthwise)
        self.up1 = UpRefineGN(c2, groups=groups, depthwise=depthwise)

        self.g4   = KGS(c_x=c4, c_g=c5, d=g4, grid_size=kan_grid_size,
                              spline_order=kan_spline_order, dropout=gate_dropout)
        self.g3   = KGS(c_x=c3, c_g=c4, d=g3, grid_size=kan_grid_size,
                              spline_order=kan_spline_order, dropout=gate_dropout)
        self.g2   = KGS(c_x=c2, c_g=c3, d=g2, grid_size=kan_grid_size,
                              spline_order=kan_spline_order, dropout=gate_dropout)
        self.g1   = KGS(c_x=c1, c_g=c2, d=g1, grid_size=kan_grid_size,
                              spline_order=kan_spline_order, dropout=gate_dropout)
        self.attn_scheduler = AttentionScheduler(
            in_ch=c5,
            num_attention=3,
            grid_size=kan_grid_size
        )
        self.use_edge_attn = bool(use_edge_attn)
        if self.use_edge_attn:
            self.edge_head = TextureEdgeAttentionHead(in_ch=c1, mid_ch=max(16, c1 // 2))
            self.register_buffer("_has_edge", torch.tensor(1.0), persistent=False)
            self.last_edge_logits: Optional[torch.Tensor] = None
        else:
            self.edge_head = None
            self.register_buffer("_has_edge", torch.tensor(0.0), persistent=False)
            self.last_edge_logits = None
        # ---- Heads ----
        self.deep_supervision = bool(deep_supervision)
        self.head = ClassBalancedHead(
            in_ch=c1,
            num_classes=num_classes,
            grid_size=kan_grid_size
        )

        if self.deep_supervision:
            self.aux1 = ClassBalancedHead(in_ch=c2, num_classes=num_classes, grid_size=kan_grid_size)
            self.aux2 = ClassBalancedHead(in_ch=c3, num_classes=num_classes, grid_size=kan_grid_size)

        with torch.no_grad():
            neg_b = -2.2
            if self.head.bias is not None:
                    nn.init.constant_(self.head.bias, neg_b)
            if self.deep_supervision:
                if self.aux1.bias is not None:
                        nn.init.constant_(self.aux1.bias, neg_b)
                if self.aux2.bias is not None:
                        nn.init.constant_(self.aux2.bias, neg_b)
        self._logit_clip = float(logit_clip)


    @torch.no_grad()
    def toggle_ssa(self, enabled: bool, alpha: Optional[float] = None):
        self._ssa_enabled.fill_(1.0 if enabled else 0.0)
        if enabled and (alpha is not None):
            self.ssa_scale.fill_(float(alpha))

    def forward(self, x: torch.Tensor):
        x = _safe(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        e5 = self.enc5(self.down4(e4))

        if self.use_edge_attn and hasattr(self, 'edge_head'):
            edge_feat_enhanced, edge_logits = self.edge_head(e1)
            if hasattr(self, 'edge_feat_proj'):
                edge_feat = self.edge_feat_proj(edge_feat_enhanced)
            else:
                edge_feat = edge_feat_enhanced
        else:
            edge_feat = torch.zeros_like(e5)
        ssa_out = self._ssa_enabled * self.ssa_scale * self.ssa(e5)
        spp_out = self.spp(e5)
        ms_sem_out = self.ms_sem(spp_out)
        b_fused = self.attn_scheduler(x, [ssa_out, ms_sem_out, edge_feat])
        b = e5 + b_fused
        d4 = self.up4(b);s4, _ = self.g4(e4, d4);d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4);s3, _ = self.g3(e3, d3);d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3);s2, _ = self.g2(e2, d2);d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2);s1, _ = self.g1(e1, d1);d1 = self.dec1(torch.cat([d1, s1], dim=1))
        y  = self.head(d1)
        if self.training:
            y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            y = torch.nan_to_num(y, nan=0.0, posinf=self._logit_clip, neginf=-self._logit_clip)
            y = y.clamp_(-self._logit_clip, self._logit_clip)
        if not self.deep_supervision:
            return y
        a1 = self.aux1(d2)
        a2 = self.aux2(d3)
        if self.training:
            a1 = torch.nan_to_num(a1)
            a2 = torch.nan_to_num(a2)
        else:
            def _sanitize(t):
                t = torch.nan_to_num(t, nan=0.0, posinf=self._logit_clip, neginf=-self._logit_clip)
                return t.clamp_(-self._logit_clip, self._logit_clip)

            a1 = _sanitize(a1)
            a2 = _sanitize(a2)
        a1 = F.interpolate(a1, size=y.shape[-2:], mode='bilinear', align_corners=False)
        a2 = F.interpolate(a2, size=y.shape[-2:], mode='bilinear', align_corners=False)
        return [a2, a1, y]

    def gate_mono_reg(self, lam: float = 1e-4) -> torch.Tensor:
        reg = 0.0
        for g in [self.g1, self.g2, self.g3, self.g4]:
            reg = reg + g.monotone_reg(lam=lam)
        return reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=next(self.parameters()).device)


#The following code is solely for supporting the U_Bench training framework and is not part of the model code.
class KGS_UNet_Compat(nn.Module):
    def __init__(self,

                 in_ch: int = 1,
                 num_classes: int = 2,
                 base_ch: int = 32,
                 groups: int = 2,
                 depthwise: bool = True,

                 use_ssa: bool = False,
                 gate_dims: Tuple[int,int,int,int] = (4,8,12,16),
                 gate_dropout: float = 0.05,
                 kan_grid_size: int = 5,
                 kan_spline_order: int = 3,
                 ssa_heads: int = 2,
                 ssa_dim_head: int = 16,
                 ssa_grid_size: int = 5,
                 ssa_spline_order: int = 3,
                 deep_supervision: bool = False,
                 logit_clip: float = 50.0,

                 input_channel: Optional[int] = None,
                 return_deeps: Optional[bool] =  None
                 ):

        super().__init__()
        if input_channel is not None:
            in_ch = int(input_channel)
        self.return_deeps = bool(deep_supervision) if (return_deeps is None) else bool(return_deeps)

        self.core = KGS_UNet(
            in_ch=in_ch, num_classes=num_classes, base_ch=base_ch, groups=groups,
            depthwise=depthwise, gate_dims=gate_dims, gate_dropout=gate_dropout,
            kan_grid_size=kan_grid_size, kan_spline_order=kan_spline_order,
            use_ssa=use_ssa, ssa_heads=ssa_heads, ssa_dim_head=ssa_dim_head,
            ssa_grid_size=ssa_grid_size, ssa_spline_order=ssa_spline_order,
            deep_supervision=deep_supervision,
            logit_clip=logit_clip
        )
    def _pack(self, y: torch.Tensor, a1: Optional[torch.Tensor] = None, a2: Optional[torch.Tensor] = None):
        if not self.return_deeps:
            return y
        a1 = a1 if (a1 is not None) else y
        a2 = a2 if (a2 is not None) else y

        return [a2, a1, y]

    def forward(self, x: torch.Tensor):
        out = self.core(x)

        # 1) list/tuple (returns [a2, a1, y] when our deep_supervision=True)
        if isinstance(out, (list, tuple)):
            if len(out) == 3 and all(torch.is_tensor(t) for t in out):
                a2, a1, y = out
                return self._pack(y, a1, a2)
            if len(out) >= 1 and torch.is_tensor(out[0]):
                y  = out[-1]
                a1 = out[-2] if len(out) >= 2 and torch.is_tensor(out[-2]) else y
                a2 = out[-3] if len(out) >= 3 and torch.is_tensor(out[-3]) else y
                return self._pack(y, a1, a2)
            raise TypeError(f"Unexpected list/tuple output from core: {[type(t) for t in out]}")

        # 2) direct tensor
        if torch.is_tensor(out):
            return out if not self.return_deeps else self._pack(out)

        # 3) dict
        if isinstance(out, dict):
            y = out.get("logits", None)
            aux = out.get("aux", None)
            if y is None:
                for k in ("out", "pred", "y"):
                    if k in out:
                        y = out[k]; break
            if not self.return_deeps:
                return y
            a1 = aux[0] if (isinstance(aux, (list, tuple)) and len(aux) > 0 and torch.is_tensor(aux[0])) else y
            a2 = aux[1] if (isinstance(aux, (list, tuple)) and len(aux) > 1 and torch.is_tensor(aux[1])) else y
            return self._pack(y, a1, a2)
        raise TypeError(f"Unsupported output type from core: {type(out)}")

    def load_state_dict(self, state_dict, strict: bool = True):
        info = super().load_state_dict(state_dict, strict=False)
        drop_prefix = ("core.ssa.", "core.ssa_scale", "core._ssa_enabled")
        miss = [k for k in info.missing_keys if not k.startswith(drop_prefix)]
        unex = [k for k in info.unexpected_keys if not k.startswith(drop_prefix)]
        if strict and (miss or unex):
            raise RuntimeError(f"Incompatible keys (non-SSA): missing={miss}, unexpected={unex}")
        return info

def KGS_UNet_F(
    input_channel:Optional[int] = None,
    in_ch: int = 3,
    num_classes: int = 1,
    base_ch: int =64,#DRIVE recommended 48
    groups: int = 4,
    depthwise: bool = True,
    use_ssa: bool = False,
    gate_dims: Tuple[int,int,int,int] = (4, 8, 12, 16),
    gate_dropout: float = 0.05,#DRIVE recommended0.02
    kan_grid_size: int = 5,
    kan_spline_order: int = 3,
    ssa_heads: int = 4,
    ssa_dim_head: int = 32,
    ssa_grid_size: int = 5,
    ssa_spline_order: int = 3,
    deep_supervision: bool = False,
    logit_clip: float = 50,
    return_deeps: Optional[bool] = None,
) -> KGS_UNet_Compat:
    if input_channel is not None:
        in_ch = int(input_channel)
    return KGS_UNet_Compat(
        in_ch=in_ch,
        num_classes=num_classes,
        base_ch=base_ch,
        groups=groups,
        depthwise=depthwise,
        use_ssa=use_ssa,
        gate_dims=gate_dims,
        gate_dropout=gate_dropout,
        kan_grid_size=kan_grid_size,
        kan_spline_order=kan_spline_order,
        ssa_heads=ssa_heads,
        ssa_dim_head=ssa_dim_head,
        ssa_grid_size=ssa_grid_size,
        ssa_spline_order=ssa_spline_order,
        deep_supervision=deep_supervision,
        logit_clip=logit_clip,
        return_deeps=return_deeps,
    )

