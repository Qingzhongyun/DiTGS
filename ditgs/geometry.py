"""Camera validation and reference-point Pluecker rays on the VAE grid."""
import torch
from torch.nn import functional as F

def validate_cameras(c2w, intrinsics):
    if c2w.shape[-2:] != (4, 4) or intrinsics.shape != (*c2w.shape[:-2], 4):
        raise ValueError("Malformed camera/intrinsics shape")
    if not torch.isfinite(c2w).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("Nonfinite camera/intrinsics")
    if (intrinsics[..., :2] <= 0).any():
        raise ValueError("Nonpositive focal length")
    bottom = c2w.new_tensor([0, 0, 0, 1]).expand_as(c2w[..., 3, :])
    rotation = c2w[..., :3, :3].float()
    identity = torch.eye(3, device=rotation.device).expand_as(rotation)
    if not torch.allclose(c2w[..., 3, :], bottom, atol=1e-5, rtol=0):
        raise ValueError("Invalid homogeneous camera row")
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-2, rtol=0):
        raise ValueError("Camera rotation is not orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-2, rtol=0):
        raise ValueError("Camera rotation is singular/reflected")
    if (torch.linalg.vector_norm(c2w[..., :3, 3].double(), dim=-1) > 1e9).any():
        raise ValueError("Absurd camera translation")

def camera_raymaps(c2w, intrinsics, height, width, latent_height, latent_width):
    """[B,V,4,4], pixel K -> [B,6,V,h,w]; camera convention is +z forward."""
    with torch.autocast(device_type=c2w.device.type, enabled=False):
        c2w, intrinsics = c2w.float(), intrinsics.float()
        yy, xx = torch.meshgrid(
            (torch.arange(latent_height, device=c2w.device) + 0.5) * height / latent_height,
            (torch.arange(latent_width, device=c2w.device) + 0.5) * width / latent_width,
            indexing="ij")
        fx, fy, cx, cy = intrinsics.unbind(-1)
        x = (xx - cx[..., None, None]) / fx[..., None, None]
        y = (yy - cy[..., None, None]) / fy[..., None, None]
        local = F.normalize(torch.stack((x, y, torch.ones_like(x)), dim=-1), dim=-1)
        direction = torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3], local)
        direction = F.normalize(direction, dim=-1)
        origin = c2w[..., :3, 3][:, :, None, None, :].expand_as(direction)
        closest = origin - (origin * direction).sum(-1, keepdim=True) * direction
        return torch.cat((direction, closest), dim=-1).permute(0, 4, 1, 2, 3).contiguous()


def tokengs_plucker_maps(c2w, intrinsics, height, width):
    """[B,V,6,H,W] in TokenGS order [o cross d, d], for every scene/view.

    Generate at RGB pixel centers; do not interpolate the DiT's low-res [d,p].
    All input cameras already use the dataloader's reference frame and scale.
    """
    rays = camera_raymaps(c2w, intrinsics, height, width, height, width)
    direction = rays[:, :3].permute(0, 2, 3, 4, 1)
    origin = c2w.float()[..., :3, 3][:, :, None, None, :].expand_as(direction)
    moment = torch.cross(origin, direction, dim=-1)
    return torch.cat((moment, direction), dim=-1).permute(0, 1, 4, 2, 3).contiguous()


def flashworld_camera_raymaps(c2w, intrinsics, height, width, latent_height,
                             latent_width, scene_scale=0.15, normalization_eps=0.01):
    """DiT-only released FlashWorld [direction, closest-point] convention.

    Input is TokenGS OpenCV first-camera gauge with translation already multiplied
    by scene_scale. FlashWorld uses OpenGL cameras, first-camera gauge, and
    t_raw / (max_view_norm(t_raw) + 0.01). Let S=diag(1,-1,-1).
    OpenGL relative rotations are S R_cv S; local rays are S r_cv, hence the
    DiT-world direction is S d_cv. Closest points are S p_cv divided by
    max_view_norm(t_scaled) + scene_scale*0.01.
    No GS camera, target K, render matrix or adapter Pluecker map is changed.
    """
    if scene_scale <= 0 or normalization_eps <= 0:
        raise ValueError("Positive source camera scale and normalization epsilon required")
    if not torch.isfinite(c2w).all():
        raise ValueError("Nonfinite DiT camera")
    identity = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand_as(c2w[:, 0])
    if not torch.allclose(c2w[:, 0], identity, atol=1e-4, rtol=0):
        raise ValueError("FlashWorld conversion expects the first-camera gauge")
    rays = camera_raymaps(c2w, intrinsics, height, width, latent_height, latent_width)
    with torch.autocast(device_type=c2w.device.type, enabled=False):
        basis = rays.new_tensor([1., -1., -1.]).view(1, 3, 1, 1, 1)
        radius = torch.linalg.vector_norm(c2w.float()[..., :3, 3], dim=-1).amax(dim=1)
        denominator = (radius + scene_scale * normalization_eps).view(-1, 1, 1, 1, 1)
        return torch.cat((rays[:, :3] * basis, rays[:, 3:] * basis / denominator), dim=1)
