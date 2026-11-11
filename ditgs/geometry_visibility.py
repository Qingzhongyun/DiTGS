"""Visibility geometry independent of rasterizer/opacity/scale culling."""
import torch
from torch.nn import functional as F


def frustum_visibility(xyz, cameras_transposed, intrinsics, height, width,
                       near_plane, far_plane, distance_scale):
    """Mean log1p distance to the closest camera frustum, in GS world units.

    Six signed plane distances: left/right/top/bottom/near/far. Camera matrices
    use the renderer's transposed world-to-camera contract. No division by z,
    projected-UV buffer, opacity, radius, Gaussian scale or upper clamp is used.
    Every finite point outside ALL frustums has a nonzero geometric subgradient
    in camera space, including points behind near or beyond far. log1p keeps
    far-out finite values manageable without a hard zero-gradient plateau.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("Require nonempty [points,3] means")
    if cameras_transposed.ndim != 3 or cameras_transposed.shape[1:] != (4,4):
        raise ValueError("Require [views,4,4] transposed W2C matrices")
    if intrinsics.shape != (cameras_transposed.shape[0],4):
        raise ValueError("Intrinsics/view count mismatch")
    if min(height,width,near_plane,distance_scale) <= 0 or far_plane <= near_plane:
        raise ValueError("Invalid raster/depth/distance contract")
    with torch.autocast(device_type=xyz.device.type,enabled=False):
        world_to_camera = cameras_transposed.float().transpose(-1,-2)
        camera_xyz = torch.einsum("vij,nj->vni",world_to_camera[:,:3,:3],xyz.float())
        camera_xyz = camera_xyz + world_to_camera[:,:3,3][:,None]
        x,y,z = camera_xyz.unbind(-1)
        fx,fy,cx,cy = (v[:,None] for v in intrinsics.float().unbind(-1))
        signed = torch.stack((
            (-fx*x-cx*z)/torch.sqrt(fx.square()+cx.square()),
            (fx*x-(width-cx)*z)/torch.sqrt(fx.square()+(width-cx).square()),
            (-fy*y-cy*z)/torch.sqrt(fy.square()+cy.square()),
            (fy*y-(height-cy)*z)/torch.sqrt(fy.square()+(height-cy).square()),
            near_plane-z,z-far_plane),dim=-1)
        distances = F.relu(signed).sum(-1)
        nearest = distances.min(dim=0).values
        loss = torch.log1p(nearest/distance_scale).mean()
        with torch.no_grad():
            stats = {"geometry_inside_any_frustum_fraction":float((signed<=0).all(-1).any(0).float().mean()),
                     "geometry_nearest_frustum_distance_mean":float(nearest.mean()),
                     "geometry_nearest_frustum_distance_max":float(nearest.max())}
    return loss,stats
