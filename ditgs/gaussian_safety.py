"""Local Gaussian repair; low contribution is never a batch rejection."""
import math
import torch
from tokengs.models.activations import ClipActivationHead

def stable_quaternion(q):
    """Normalize in FP32; zero/subnormal magnitude uses a local identity fallback."""
    q = q.float()
    magnitude = q.detach().abs().amax(dim=-1, keepdim=True)
    # A subnormal divisor can overflow the FP32 normalization Jacobian.
    # This is a dtype limit, not a learned-geometry threshold.
    valid = torch.isfinite(q).all(-1, keepdim=True) & (magnitude >= torch.finfo(q.dtype).tiny)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    finite = torch.where(valid, q, identity)
    divisor = torch.where(valid, magnitude, torch.ones_like(magnitude))
    scaled = finite / divisor
    return scaled / scaled.square().sum(-1, keepdim=True).sqrt(), ~valid.squeeze(-1)

def distribution(value):
    x = value.detach().reshape(-1).cpu()
    if x.dtype != torch.float64:
        x = x.float()
    finite = x[torch.isfinite(x)]
    result = {"count": x.numel(), "nonfinite_count": int((~torch.isfinite(x)).sum())}
    if finite.numel():
        qs = torch.quantile(finite, torch.tensor([0., .001, .01, .5, .99, .999, 1.], dtype=finite.dtype))
        result.update(zip(("min", "p001", "p01", "p50", "p99", "p999", "max"), qs.tolist()))
    return result

def prepare_gaussians(gaussians, scale_cap=.075, detailed=False):
    """Quarantine invalid points; every finite positive tiny scale survives."""
    g = gaussians.float()
    detached = g.detach()
    scales, q = detached[..., 4:7], detached[..., 7:11]
    finite = torch.isfinite(detached).all(-1)
    # Native position activation bound plus z offset, not a scene-quality bound.
    position_bad = (detached[..., :3].abs() > math.expm1(15.) + 2.).any(-1)
    scale_bad = ((scales < 0) | (scales > scale_cap * (1 + 1e-5))).any(-1)
    opacity_bad = (detached[..., 3] < 0) | (detached[..., 3] > 1)
    color_bad = ((detached[..., 11:] < 0) | (detached[..., 11:] > 1)).any(-1)
    inactive = ~finite | position_bad | scale_bad | opacity_bad | color_bad
    placeholder = torch.zeros_like(g)
    placeholder[..., 4:7] = .01
    placeholder[..., 7] = 1
    safe = torch.where(inactive.unsqueeze(-1), placeholder, g)
    fixed_q, q_fallback = stable_quaternion(safe[..., 7:11])
    fixed_s = torch.where(safe[..., 4:7] == 0,
                          torch.full_like(safe[..., 4:7], torch.finfo(torch.float32).tiny),
                          safe[..., 4:7])
    result = torch.cat((safe[..., :4], fixed_s, fixed_q, safe[..., 11:]), -1)
    total = inactive.numel()
    masks = {"nonfinite_points": ~finite, "position_contract_points": position_bad,
             "invalid_scale_points": scale_bad, "invalid_opacity_points": opacity_bad,
             "invalid_color_points": color_bad, "quarantined_points": inactive,
             "zero_scale_points": (scales == 0).any(-1) & finite,
             "tiny_positive_scale_points": ((scales > 0) & (scales < 1e-8)).any(-1) & finite,
             "quaternion_fallback_points": q_fallback & ~inactive}
    counts = torch.stack([v.sum() for v in masks.values()]).cpu().tolist()
    stats = {"points": total, **dict(zip(masks, map(int, counts)))}
    stats["active_points"] = total - stats["quarantined_points"]
    stats["quarantined_fraction"] = stats["quarantined_points"] / max(total, 1)
    if detailed:
        stats["scale_axes"] = [distribution(scales[..., i]) for i in range(3)]
        stats["log10_positive_scales"] = distribution(scales[scales > 0].log10())
        stats["quaternion_norm_before"] = distribution(q.cpu().double().norm(dim=-1))
        stats["quaternion_norm_after"] = distribution(fixed_q.detach().cpu().norm(dim=-1))
        stats["quaternion_norm_error"] = distribution((fixed_q.detach().cpu().norm(dim=-1) - 1).abs())
    return result, ~inactive, stats

class LocalSafeActivationHead(ClipActivationHead):
    """Same learned tensors/readout layout; sanitize before unsafe nonlinearities."""
    detailed_statistics = False

    def forward(self, x, rays_os=None, rays_ds=None, return_logits=False):
        raw = self.deconv(x.float()).reshape(x.shape[0], -1, self.output_dims)
        nonfinite = ~torch.isfinite(raw.detach()).all(-1)
        clean = torch.where(nonfinite.unsqueeze(-1), torch.zeros_like(raw), raw)
        recorder = getattr(self, "activation_recorder", None)
        if recorder is not None:
            recorder("gs_head/logits", clean)
        pos, rgb, scaling, rotation, opacity = clean.split([3, 3, 3, 4, 1], -1)
        q, fallback = stable_quaternion(rotation)
        gs = torch.cat((self.pos_act(pos), self.opacity_act(opacity),
                        self.scale_act(scaling), q, self.rgb_act(rgb)), -1)
        gs = torch.cat((gs[..., :3],
                        torch.where(nonfinite.unsqueeze(-1), torch.zeros_like(gs[..., 3:4]), gs[..., 3:4]),
                        gs[..., 4:]), -1)
        self.local_invalid_mask = nonfinite.detach()
        q_magnitude = rotation.detach().abs().amax(-1)
        self.gaussian_statistics = {
            "nonfinite_logit_points": int(nonfinite.sum().item()),
            "position_logit_saturated_points": int((pos.detach().abs() >= 15.).any(-1).sum().item()),
            "opacity_below_raster_threshold_points": int((gs.detach()[..., 3] < 1/255.).sum().item()),
            "raw_quaternion_fallback_points": int((fallback & ~nonfinite).sum().item()),
            "raw_zero_quaternion_points": int(((q_magnitude == 0) & ~nonfinite).sum().item()),
            "raw_subnormal_quaternion_points": int(((q_magnitude > 0) &
                (q_magnitude < torch.finfo(torch.float32).tiny) & ~nonfinite).sum().item())}
        if self.detailed_statistics:
            self.gaussian_statistics["raw_quaternion_norm"] = distribution(
                raw.detach()[..., 9:13].cpu().double().norm(dim=-1))
            self.gaussian_statistics["scale_logits"] = distribution(scaling.detach() - self.scale_shift)
        return (gs, raw) if return_logits else gs

def repair_renderer_gradient(gradient):
    """A nonfinite derivative at one Gaussian cannot poison the other rows."""
    valid = torch.isfinite(gradient).all(-1, keepdim=True)
    return torch.where(valid, gradient, torch.zeros_like(gradient)), int((~valid).sum().item())
