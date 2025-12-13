import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # ---- Parameters ----
        self.base_weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.enable_standalone_scale_spline = bool(enable_standalone_scale_spline)
        if self.enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.empty(out_features, in_features)
            )

        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.base_activation = base_activation()
        self.grid_eps = float(grid_eps)

        self.reset_parameters()

    def reset_parameters(self):
        # --- Initialize base_weight (BUGFIX: it was uninitialized -> NaNs in MmBackward) ---
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            self.base_weight.mul_(self.scale_base)

        # --- Initialize spline path weights ---
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # a positive-ish small scale is safer/stable; keep learnable
                torch.nn.init.constant_(self.spline_scaler, self.scale_spline)

    # -----------------------------
    # Numerically safer helpers
    # -----------------------------
    @staticmethod
    def _finite(x: torch.Tensor, lo: float = -6.0, hi: float = 6.0) -> torch.Tensor:
        # clamp after converting NaN/Inf; do NOT do in-place to keep autograd happy
        return torch.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo).clamp(lo, hi)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x (torch.Tensor): (batch_size, in_features).
        Returns:
            (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            den1 = (grid[:, k:-1] - grid[:, :-(k + 1)]).clamp_min(1e-6)
            den2 = (grid[:, k + 1:] - grid[:, 1:(-k)]).clamp_min(1e-6)

            bases = ((x - grid[:, :-(k + 1)]) / den1) * bases[:, :, :-1] +                     ((grid[:, k + 1:] - x) / den2) * bases[:, :, 1:]

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        Args:
            x (torch.Tensor): (batch_size, in_features).
            y (torch.Tensor): (batch_size, in_features, out_features).
        Returns:
            (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor):
            assert x.dim() == 2 and x.size(1) == self.in_features
            x = self._finite(x)


            grid_min, grid_max = self.grid_range[0], self.grid_range[1]
            x = x.clamp(grid_min - 0.1, grid_max + 0.1)
            x = (x - grid_min) / (grid_max - grid_min + 1e-8)
            x = x * (grid_max - grid_min) + grid_min

            base_output = F.linear(self.base_activation(x), self.base_weight)
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )
            y = base_output + spline_output
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
            return y

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        x_mean = x_sorted.mean(dim=0, keepdim=True)
        x_std = x_sorted.std(dim=0, keepdim=True)
        # 动态边界：覆盖均值±3σ，至少保留原范围
        new_min = torch.min(torch.cat([x_mean - 3 * x_std, self.grid_range[0].unsqueeze(0)], dim=0))
        new_max = torch.max(torch.cat([x_mean + 3 * x_std, self.grid_range[1].unsqueeze(0)], dim=0))
        uniform_step = (new_max - new_min + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
                * uniform_step
                + new_min
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0,regularize_smooth=1.0):
        """
        Compute the regularization loss.
        L1 on spline weights + entropy term on their magnitudes.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-12)
        regularization_loss_entropy = -torch.sum(p * (p + 1e-12).log())
        smooth_loss = (self.spline_weight[..., 1:] - self.spline_weight[..., :-1]).abs().mean()
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
            + regularize_smooth * smooth_loss
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
class SharedKANForward(nn.Module):
    """
    仅转发到一个“共享的” KANLinear（共享全部参数）；用于零开销共享。
    """
    def __init__(self, core: "KANLinear"):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)

class LoRAKANAdapter(nn.Module):
    """
    在共享 KANLinear 之上叠加极小的低秩增量（仅改“基线性路径”的一阶项），
    等价： y = scale_out * core(x) + α/r * B(A(x)) 。
    - 适合同形状的 KAN（如 d→1 或 1→1）在多处共享主体，再做细微域适配；
    - 不动样条网格与样条系数，数值最稳。
    """
    def __init__(self, core: "KANLinear", rank: int = 2, alpha: float = 1.0):
        super().__init__()
        assert isinstance(core, nn.Module), "core must be a KANLinear"
        self.core = core
        self.rank = int(rank)
        self.alpha = float(alpha) / max(1, self.rank)

        in_f, out_f = core.in_features, core.out_features
        if self.rank > 0:
            self.lora_A = nn.Linear(in_f, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, out_f, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A, self.lora_B = None, None

        # 允许每个适配器轻微缩放输出（近似调节样条路径强度）
        self.scale_out = nn.Parameter(torch.ones(out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(torch.nan_to_num(x))
        y = y * self.scale_out
        if self.rank > 0:
            y = y + self.alpha * self.lora_B(self.lora_A(torch.nan_to_num(x)))
        return y