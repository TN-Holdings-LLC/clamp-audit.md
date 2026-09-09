import torch
import torch.nn as nn
import torch.nn.functional as F


class GPCLayer(nn.Module):
    """Original pasted version (with the x_proj*gate fix already applied)."""

    def __init__(self, lam=0.15, sigma=1.0, eps=1e-6):
        super().__init__()
        self.lam = lam
        self.sigma = sigma
        self.eps = eps

    def _projective_clamp(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (x / (norm + self.eps)) * self.sigma * torch.tanh(norm / self.sigma)

    def _hopf_projection(self, x: torch.Tensor) -> torch.Tensor:
        pad_len = (4 - x.shape[-1] % 4) % 4
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='constant')
        q = x.view(*x.shape[:-1], -1, 4)
        q = F.normalize(q, dim=-1, eps=self.eps)
        z = q[..., 0] ** 2 + q[..., 3] ** 2 - q[..., 1] ** 2 - q[..., 2] ** 2
        return z.mean(dim=-1, keepdim=True)

    def _causal_eit_smoothing(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() >= 3:
            z_smoothed = torch.zeros_like(z)
            z_curr = z[:, 0, :]
            z_smoothed[:, 0, :] = z_curr
            for t in range(1, z.shape[1]):
                z_curr = (1 - self.lam) * z_curr + self.lam * z[:, t, :]
                z_smoothed[:, t, :] = z_curr
            return z_smoothed
        else:
            return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self._projective_clamp(x)
        z_hopf = self._hopf_projection(x_proj)
        z_eit = self._causal_eit_smoothing(z_hopf)
        gate = torch.sigmoid(z_eit * 5.0)
        return x_proj * gate


class SafeModel(nn.Module):
    def __init__(self, base_model: nn.Module, lam=0.15, sigma=1.0):
        super().__init__()
        self.gpcl = GPCLayer(lam=lam, sigma=sigma)
        self.base_model = base_model

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x_safe = self.gpcl(x)
        return self.base_model(x_safe, *args, **kwargs)
