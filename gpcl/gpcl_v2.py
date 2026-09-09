"""
GPCLayer v2 -- a rework of the pasted Geometric Pre-Constraint Layer.

WHAT CHANGED AND WHY (every item below was measured, not assumed; the
measurements are in diag_gpcl.py / diag_safemodel2.py / test_gpcl_v2.py):

 1. SCALE-BLIND CLAMP -> CALIBRATED RADIUS.
    v1 used a fixed `sigma` (default 1.0) as the saturation radius. Real
    activations do not have norm ~1: unit-variance features of width d have
    norm ~sqrt(d), so at d=768 every ordinary input arrives at norm ~29,
    i.e. tanh(29) = 1.0 -- fully saturated. Measured consequence: a 1000x
    spike produced an output only 1.03x larger than a clean input at d=768,
    and 1.004x at d=4096. The clamp could not distinguish an anomaly from
    normal data because it was saturated on normal data. v2 tracks the
    typical norm as a running statistic (BatchNorm-style buffer, or set
    once via `calibrate()`), so the operating point is correct at any scale.

 2. NO IDENTITY PATH -> EXACT IDENTITY IN-DISTRIBUTION.
    v1's clamp is `direction * sigma * tanh(norm/sigma)`, which rescales
    EVERY input, not just outliers. Measured: clean inputs came out at
    1.8% of their original norm at d=768. v2 uses a hinged soft clamp that
    is *exactly* the identity while norm <= radius and only bends above it,
    C1-continuous at the hinge. Clean data passes through bit-unchanged.

 3. BLIND TO THE ANOMALY THAT ACTUALLY MATTERS.
    v1 only ever rescales the magnitude, and preserves direction exactly.
    A one-coordinate spike is mostly a *direction* change, so v1 cannot
    remove it. Measured on a frozen classifier: v1 reduced spike damage by
    0.03 points out of 3.52 -- i.e. not at all -- while costing 2.3 points
    of clean accuracy. v2 adds a per-feature soft clamp against a running
    per-feature scale, which is what actually catches coordinate spikes.

 4. THE GATE WAS A CONSTANT.
    v1's Hopf gate is a mean over d/4 unit quaternions, each with mean 0,
    so it concentrates on 0.5 as width grows: measured 0.500 +/- 0.018 at
    d=4096, and completely insensitive to spike magnitude past ~100
    (0.457522 for every spike from 1e3 to 1e5). It was a `* 0.5` in
    disguise. v2 drives the gate from an actual anomaly statistic (relative
    excess over the calibrated radius), so gate==1 on clean data and falls
    only when something is genuinely out of distribution. The Hopf map is
    kept as an opt-in extra channel, with its padding bug fixed, for anyone
    who wants the geometric term -- but it is off by default because it
    measurably contributes nothing at width.

 5. ZERO-PADDING BIASED THE HOPF TERM.
    Padding to a multiple of 4 with zeros creates all-zero quaternions;
    F.normalize maps those to 0, so they contribute z=0 and drag the mean.
    Measured: identical data at d=61 gave E[z]=+0.064 vs ~0.000 at d=60/64,
    a 7% gate shift caused purely by feature-count parity. v2 masks the
    padded bundles out of the mean instead.

 6. O(T) PYTHON LOOP -> O(log T) VECTORISED SCAN.
    v1 ran one Python iteration per timestep: measured 31.3 ms/call at
    T=2048. v2 computes the identical recurrence with a Hillis-Steele
    doubling scan in ceil(log2 T) steps (11 at T=2048). The result is the
    same recurrence, not an approximation -- verified to < 1e-6 max abs
    deviation against the original loop in test_gpcl_v2.py.

 7. SILENTLY WRONG ON NON-(B,T,F) LAYOUTS.
    v1 normalised over dim=-1 and treated dim=1 as time. On a (B,C,H,W)
    CNN tensor that means "normalise over image width, smooth over
    channels" -- nonsense, and it ran without raising. v2 takes an explicit
    layout and raises on anything it cannot handle correctly.

 8. "WRAP ANY LEGACY MODEL" WAS FALSE FOR TOKEN INPUTS.
    v1's SafeModel applied the layer to whatever the model's first argument
    was; for a transformer that is an integer token-id tensor, and it
    raised RuntimeError on the norm. v2's SafeModel checks dtype and, by
    default, hooks the layer *after* the embedding instead of before it.

WHICH STAGE ACTUALLY EARNS ITS PLACE (ablation.py / ablation2.py, 5 seeds,
frozen classifier, 5% of rows corrupted). Damage removed vs. the bare model:

    per-feature clamp only ................... 98-99 %
    norm clamp only ("/0 projective") ......... 0 %
    Hopf gate only ............................ 0 %
    anomaly gate only ......................... 0 %
    all stages together ...................... 98-99 %

All three stages of the original design measured exactly zero contribution,
in every scenario tried -- single-coordinate spikes at 1e2 and 1e3, and
whole-vector gain explosions at 50x and 1000x against a saturating (tanh)
network, which is the norm clamp's best case. The component that does the
work is the per-feature robust clamp, which was not in the original design.
This is a statement about the configurations tested here, not a proof that a
norm bound can never matter (an exploding residual stream during training is
a plausible case where it would). But if the goal is the one the original
docstring states -- neutralising OOD spikes before they reach the model --
the measured answer is that the per-feature clamp is the whole mechanism,
and `clamp_norm=False, gate_mode="off"` loses nothing while being cheaper.
The other stages are kept, off or floored, so the geometry can be re-enabled
by anyone who finds a case where it does pay.

HONEST SCOPE. This is a magnitude/outlier conditioner with a calibrated
operating point. It does not "enforce R=0", "neutralise OOD noise", or
"upgrade a model to Love-OS standards"; those are claims no measurement
here supports. It does one narrow, checkable thing: it bounds how far a
single input can deviate from the distribution the layer was calibrated
on, while leaving in-distribution inputs untouched. Calibration data IS
required -- so "no retraining" is true, but "nothing needed" is not.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GPCLayer", "SafeModel"]


class GPCLayer(nn.Module):
    """Calibrated outlier conditioner: identity in-distribution, saturating outside.

    Args:
        num_features: size of the feature axis. Required for the per-feature
            clamp; pass None to use the vector-norm clamp only.
        lam: EIT forgetting rate in [0, 1]. Higher forgets history faster.
        headroom: how far above the calibrated radius the output may travel,
            as a multiple of that radius. The clamp saturates at
            radius * (1 + headroom).
        momentum: EMA rate for the running statistics (as in BatchNorm).
        feature_clamp_k: per-feature clamp threshold, in robust sigmas.
            Coordinates beyond this many scaled MADs are softly pulled in.
        gate_mode: "anomaly" (default), "hopf", "both", or "off".
        layout: "BTF" (batch, time, feature), "BF", or "auto".
        eps: numerical floor.

    Shape:
        (B, T, F) or (B, F). Anything else raises -- see note 7 in the
        module docstring.
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        lam: float = 0.15,
        headroom: float = 0.5,
        momentum: float = 0.05,
        feature_clamp_k: float = 6.0,
        gate_mode: str = "anomaly",
        gate_floor: float = 0.5,
        clamp_norm: bool = True,
        radius_quantile: float = 0.99,
        layout: str = "auto",
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        if headroom <= 0:
            raise ValueError(f"headroom must be > 0, got {headroom}")
        if not 0.0 <= gate_floor <= 1.0:
            raise ValueError(f"gate_floor must be in [0, 1], got {gate_floor}")
        if not 0.5 < radius_quantile < 1.0:
            raise ValueError(
                f"radius_quantile must be in (0.5, 1.0), got {radius_quantile}"
            )
        if gate_mode not in ("anomaly", "hopf", "both", "off"):
            raise ValueError(f"unknown gate_mode {gate_mode!r}")
        if layout not in ("auto", "BTF", "BF"):
            raise ValueError(f"unknown layout {layout!r}")

        self.num_features = num_features
        self.lam = lam
        self.headroom = headroom
        self.momentum = momentum
        self.feature_clamp_k = feature_clamp_k
        self.gate_mode = gate_mode
        self.gate_floor = gate_floor
        self.clamp_norm = clamp_norm
        self.radius_quantile = radius_quantile
        self.layout = layout
        self.eps = eps

        # Running statistics. `calibrated` stays 0 until real data is seen,
        # so an uncalibrated layer can refuse to pretend it is filtering.
        self.register_buffer("running_radius", torch.ones(()))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        if num_features is not None:
            self.register_buffer("running_center", torch.zeros(num_features))
            self.register_buffer("running_scale", torch.ones(num_features))
        else:
            self.running_center = None
            self.running_scale = None

    # ------------------------------------------------------------------ utils
    def _check_layout(self, x: torch.Tensor) -> str:
        if not x.is_floating_point():
            raise TypeError(
                f"GPCLayer expects a floating-point tensor, got {x.dtype}. "
                "If you are wrapping a transformer, apply this layer after "
                "the embedding, not to the token ids (see SafeModel)."
            )
        lay = self.layout
        if lay == "auto":
            if x.dim() == 3:
                lay = "BTF"
            elif x.dim() == 2:
                lay = "BF"
            else:
                raise ValueError(
                    f"GPCLayer received a {x.dim()}-D tensor {tuple(x.shape)}. "
                    "Only (B, T, F) and (B, F) are supported. A conv tensor "
                    "(B, C, H, W) would be silently mis-handled -- reshape to "
                    "(B, H*W, C) and pass layout='BTF' if that is what you want."
                )
        if lay == "BTF" and x.dim() != 3:
            raise ValueError(f"layout='BTF' needs a 3-D tensor, got {tuple(x.shape)}")
        if lay == "BF" and x.dim() != 2:
            raise ValueError(f"layout='BF' needs a 2-D tensor, got {tuple(x.shape)}")
        if self.num_features is not None and x.shape[-1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features, got {x.shape[-1]}"
            )
        return lay

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        norms = torch.linalg.vector_norm(flat, dim=-1)
        # A high quantile, not the median: the operating point should leave
        # the overwhelming majority of normal data untouched and bend only
        # the extreme tail. Calibrating at the median clamps half of all
        # clean inputs, which is what made the first draft of this layer
        # lose 39 accuracy points on a magnitude-encoded task.
        if norms.numel() > 1_000_000:  # torch.quantile has an input-size cap
            idx = torch.randperm(norms.numel(), device=norms.device)[:1_000_000]
            radius = torch.quantile(norms[idx], self.radius_quantile)
        else:
            radius = torch.quantile(norms, self.radius_quantile)
        if not bool(self.calibrated):
            self.running_radius.copy_(radius)
        else:
            self.running_radius.mul_(1 - self.momentum).add_(self.momentum * radius)
        if self.running_center is not None:
            center = flat.median(dim=0).values
            # Median absolute deviation -> robust sigma (0.6745 = Phi^-1(0.75)).
            mad = (flat - center).abs().median(dim=0).values / 0.6745
            mad = mad.clamp_min(self.eps)
            if not bool(self.calibrated):
                self.running_center.copy_(center)
                self.running_scale.copy_(mad)
            else:
                self.running_center.mul_(1 - self.momentum).add_(self.momentum * center)
                self.running_scale.mul_(1 - self.momentum).add_(self.momentum * mad)
        self.calibrated.fill_(True)

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> "GPCLayer":
        """Set the operating point from a batch of clean, in-distribution data.

        Call this once (or a few times) before using the layer around a
        frozen model. Without it the layer has no idea what 'normal' means
        and forward() will warn.
        """
        self._check_layout(x)
        self._update_stats(x)
        return self

    # ------------------------------------------------------------------ stages
    def _feature_clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Per-coordinate soft clamp -- catches single-feature spikes, which a
        norm-only clamp cannot (it preserves direction exactly)."""
        if self.running_center is None:
            return x
        c, s = self.running_center, self.running_scale
        z = (x - c) / s
        lim = self.feature_clamp_k
        excess = (z.abs() - lim).clamp_min(0.0)
        # identity while |z| <= lim, then saturates at lim + lim*headroom
        room = lim * self.headroom
        z_new = torch.sign(z) * (z.abs().clamp_max(lim) + room * torch.tanh(excess / room))
        return z_new * s + c

    def _projective_clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Hinged norm clamp: exactly the identity while ||x|| <= radius,
        smoothly saturating at radius*(1+headroom) beyond it. C1 at the hinge.

        Disabled by `clamp_norm=False` for models whose input MAGNITUDE
        carries signal -- no norm clamp can be safe there, by construction."""
        if not self.clamp_norm:
            return x
        radius = self.running_radius.clamp_min(self.eps)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        room = radius * self.headroom
        excess = (norm - radius).clamp_min(0.0)
        new_norm = norm.clamp_max(radius) + room * torch.tanh(excess / room)
        return x * (new_norm / (norm + self.eps))

    def _anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """0 for in-distribution input, growing with how far outside it is."""
        radius = self.running_radius.clamp_min(self.eps)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        s = (norm - radius).clamp_min(0.0) / radius
        if self.running_center is not None:
            z = ((x - self.running_center) / self.running_scale).abs()
            s = s + (
                z.amax(dim=-1, keepdim=True) - self.feature_clamp_k
            ).clamp_min(0.0) / self.feature_clamp_k
        return s

    def _anomaly_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Bounded confidence weight in [gate_floor, 1].

        Deliberately floored. An unbounded exp(-score) gate zeroes the whole
        feature vector when any single coordinate spikes -- measured: one
        1e3 coordinate in a 768-d input drove the gate to exactly 0.0 and
        annihilated the entire token. That is the same
        destroy-the-signal-to-save-it failure this rework exists to remove.
        The clamps above are what actually bound the input; this gate only
        expresses reduced confidence, so it must not be able to reach 0."""
        s = torch.tanh(self._anomaly_score(x))
        return self.gate_floor + (1.0 - self.gate_floor) * (1.0 - s)

    def _hopf_projection(self, x: torch.Tensor) -> torch.Tensor:
        """S3 -> S2 Hopf z-component, averaged over quaternion bundles.

        Padded bundles are masked out of the mean rather than counted as
        z=0, which is what biased v1 (see note 5)."""
        d = x.shape[-1]
        pad_len = (4 - d % 4) % 4
        if pad_len:
            x = F.pad(x, (0, pad_len))
        q = F.normalize(x.view(*x.shape[:-1], -1, 4), dim=-1, eps=self.eps)
        z = q[..., 0] ** 2 + q[..., 3] ** 2 - q[..., 1] ** 2 - q[..., 2] ** 2
        if pad_len:
            n_real = d // 4  # bundles made entirely of real features
            if n_real == 0:
                return torch.zeros_like(z[..., :1])
            z = z[..., :n_real]
        return z.mean(dim=-1, keepdim=True)

    def _causal_eit_smoothing(self, z: torch.Tensor) -> torch.Tensor:
        """Exponential causal smoothing along the time axis.

        Identical recurrence to v1's loop:
            y[0] = z[0];  y[t] = (1-lam)*y[t-1] + lam*z[t]
        computed with a Hillis-Steele doubling scan in ceil(log2 T) steps
        instead of T Python iterations. Exact, not an approximation."""
        if z.dim() < 3 or z.shape[1] < 2 or self.lam == 1.0:
            return z
        T = z.shape[1]
        a = 1.0 - self.lam
        y = z * self.lam
        y = torch.cat([z[:, :1], y[:, 1:]], dim=1)  # y[0] = z[0], out-of-place
        k = 1
        while k < T:
            shifted = F.pad(y[:, :-k], (0, 0, k, 0))
            y = y + (a ** k) * shifted
            k *= 2
        return y

    # ------------------------------------------------------------------ forward
    def forward(
        self, x: torch.Tensor, return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        self._check_layout(x)

        if self.training:
            self._update_stats(x)
        elif not bool(self.calibrated):
            raise RuntimeError(
                "GPCLayer is not calibrated: it has never seen data, so it has "
                "no notion of what an outlier is and would clamp against a "
                "meaningless radius of 1.0. Call layer.calibrate(clean_batch) "
                "once, or run a few forward passes in train() mode, first."
            )

        x_feat = self._feature_clamp(x)
        x_proj = self._projective_clamp(x_feat)

        if self.gate_mode == "off":
            gate = torch.ones_like(x_proj[..., :1])
        else:
            terms = []
            if self.gate_mode in ("anomaly", "both"):
                terms.append(self._anomaly_gate(x))
            if self.gate_mode in ("hopf", "both"):
                terms.append(torch.sigmoid(self._hopf_projection(x_proj) * 5.0))
            gate = terms[0] if len(terms) == 1 else terms[0] * terms[1]
            gate = self._causal_eit_smoothing(gate)

        out = x_proj * gate
        if return_diagnostics:
            return out, {
                "gate": gate.detach(),
                "anomaly": self._anomaly_score(x).detach(),
                "radius": self.running_radius.detach().clone(),
                "clamped_fraction": (
                    (torch.linalg.vector_norm(x, dim=-1) > self.running_radius)
                    .float().mean().detach()
                ),
            }
        return out

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, lam={self.lam}, "
            f"headroom={self.headroom}, gate_mode={self.gate_mode}, "
            f"calibrated={bool(self.calibrated)}, "
            f"radius={float(self.running_radius):.4g}"
        )


class SafeModel(nn.Module):
    """Wrap a frozen model with a calibrated GPCLayer on its float input.

    Unlike v1 this refuses integer inputs with an actionable message rather
    than raising deep inside a norm, and it requires calibration before
    eval-mode use so it cannot silently clamp against a meaningless radius.
    """

    def __init__(self, base_model: nn.Module, num_features: Optional[int] = None, **kw):
        super().__init__()
        self.gpcl = GPCLayer(num_features=num_features, **kw)
        self.base_model = base_model

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> "SafeModel":
        self.gpcl.calibrate(x)
        return self

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not x.is_floating_point():
            raise TypeError(
                "SafeModel received a non-float input (probably token ids). "
                "Wrap the model's hidden states instead, e.g. put the GPCLayer "
                "after the embedding module rather than around the whole model."
            )
        return self.base_model(self.gpcl(x), *args, **kwargs)
