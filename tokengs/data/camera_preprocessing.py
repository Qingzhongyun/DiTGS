# Modified for DiTGS: static reconstruction integration and calibrated-camera support.
"""DL3DV raw optics -> actual-raster pinhole RGB/K, before TokenGS crop/resize.

No OpenCV dependency: inverse image sampling uses the forward OpenCV lens model
and torch.grid_sample. Rectification preserves camera orientation; only K changes.
Metadata is authoritative: never guess that images are undistorted from a folder
name or silently discard distortion. Already-pinhole/zero-distortion data bypass
resampling exactly. Both training and evaluation use this module.
"""
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F

from tokengs.camera_policy import CAMERA_PREPROCESSING
COEFFICIENTS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
MODELS = {"PINHOLE", "SIMPLE_PINHOLE", "OPENCV", "FULL_OPENCV",
          "SIMPLE_RADIAL", "RADIAL", "OPENCV_FISHEYE"}


def _camera_metadata(data, frame=None):
    result = dict(data)
    if frame is not None:
        row = data["frames"][int(frame)]
        # Intrinsics/optics may vary per frame; poses/file paths do not enter K.
        for name in ("w", "h", "fl_x", "fl_y", "cx", "cy", "camera_model", *COEFFICIENTS):
            if name in row:
                result[name] = row[name]
    return result


def intrinsics_for_raster(data, source_hw=None, frame=None):
    """Scale metadata K to decoded (H,W), with TokenGS's native edge convention."""
    meta = _camera_metadata(data, frame)
    old_w, old_h = float(meta["w"]), float(meta["h"])
    h, w = (old_h, old_w) if source_hw is None else map(int, source_hw)
    k = np.asarray([meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]], dtype=np.float64)
    if (not np.isfinite([old_w, old_h, h, w, *k]).all()
            or min(old_w, old_h, h, w, *k[:2]) <= 0):
        raise ValueError("Invalid camera calibration or actual raster dimensions")
    k *= np.asarray([w/old_w, h/old_h, w/old_w, h/old_h])
    return k.astype(np.float32)


def _spec(data, source_hw, frame):
    meta = _camera_metadata(data, frame)
    k = tuple(float(x) for x in intrinsics_for_raster(data, source_hw, frame))
    coeff = tuple(float(meta.get(name, 0.0)) for name in COEFFICIENTS)
    if not np.isfinite(coeff).all():
        raise ValueError("Nonfinite lens distortion")
    model = str(meta.get("camera_model", "")).upper()
    if not model:
        if any(coeff):
            raise ValueError("Nonzero distortion requires an explicit camera_model")
        model = "PINHOLE"
    if model not in MODELS:
        raise ValueError(f"Unsupported camera_model {model!r}; refusing pinhole fallback")
    permitted = {
        "PINHOLE": (), "SIMPLE_PINHOLE": (), "SIMPLE_RADIAL": (0,), "RADIAL": (0, 1),
        "OPENCV": (0, 1, 2, 3, 4), "FULL_OPENCV": tuple(range(8)),
        "OPENCV_FISHEYE": (0, 1, 4, 5)}[model]
    # Fisheye coefficients are k1,k2,k3,k4, not tangential p1,p2.
    if any(value != 0.0 for i, value in enumerate(coeff) if i not in permitted):
        raise ValueError(f"Coefficients inconsistent with camera model {model}")
    return tuple(map(int, source_hw)), k, model, coeff


def _distort(x, y, model, coeff):
    k1, k2, p1, p2, k3, k4, k5, k6 = coeff
    r2 = x*x + y*y
    if model == "OPENCV_FISHEYE":
        r = np.sqrt(r2)
        theta = np.arctan(r)
        t2 = theta*theta
        theta_d = theta*(1 + t2*(k1 + t2*(k2 + t2*(k3 + t2*k4))))
        factor = np.divide(theta_d, r, out=np.ones_like(r), where=r > 1e-12)
        return x*factor, y*factor
    radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
    if model == "FULL_OPENCV":
        denominator = 1 + r2*(k4 + r2*(k5 + r2*k6))
        if np.any(np.abs(denominator) < 1e-8):
            raise ValueError("Singular rational distortion model")
        radial = radial/denominator
    return (x*radial + 2*p1*x*y + p2*(r2+2*x*x),
            y*radial + p1*(r2+2*y*y) + 2*p2*x*y)


@lru_cache(maxsize=4)
def _rectification(source_hw, k, model, coeff):
    """Output K and a destination-to-source map. All output pixels remain valid.

    Use the original principal point and the smallest inward focal zoom >=1
    whose entire perimeter maps inside the source. Check the full final grid as
    well (reject folded/nonphysical calibrations). This is a specified pinhole
    policy, not an assertion of byte equality to an upstream undistorted export.
    """
    h, w = source_hw
    if model != "OPENCV_FISHEYE" and not any(coeff):
        return np.asarray(k, np.float32), None, 1.0
    fx, fy, cx, cy = k
    if min(h, w) < 3 or not (0 < cx < w-1 and 0 < cy < h-1):
        raise ValueError("Rectification requires an interior principal point and raster >=3")
    # OpenCV remaps use integer-indexed pixel centers. K is kept in the same
    # metadata convention as native TokenGS; no ad-hoc half-pixel K correction.
    bx = np.concatenate((np.arange(w), np.arange(w), np.zeros(h), np.full(h, w-1)))
    by = np.concatenate((np.zeros(w), np.full(w, h-1), np.arange(h), np.arange(h)))
    margin = 0.01  # subpixel support margin against FP32 boundary roundoff
    def project(u, v, zoom):
        x, y = (u-cx)/(fx*zoom), (v-cy)/(fy*zoom)
        xd, yd = _distort(x, y, model, coeff)
        return fx*xd+cx, fy*yd+cy
    def inside(u, v):
        return (np.isfinite(u).all() and np.isfinite(v).all()
                and u.min() >= margin and u.max() <= w-1-margin
                and v.min() >= margin and v.max() <= h-1-margin)
    zoom = 1.0
    if not inside(*project(bx, by, zoom)):
        lo, hi = 1.0, 2.0
        while not inside(*project(bx, by, hi)):
            hi *= 2
            if hi > 16:
                raise ValueError("Cannot form a valid pinhole raster within 16x rectification zoom")
        for _ in range(40):
            mid = (lo+hi)/2
            if inside(*project(bx, by, mid)):
                hi = mid
            else:
                lo = mid
        zoom = hi*(1+2e-6)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing="ij")
    u, v = project(xx, yy, zoom)
    if not inside(u, v):
        raise ValueError("Nonphysical/folded lens map: interior falls outside valid source")
    # A bounded map alone can still fold over itself; reject nonpositive local
    # image Jacobians rather than feed a nonphysical raster to a pinhole model.
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    jacobian = du_dx*dv_dy - du_dy*dv_dx
    if not np.isfinite(jacobian).all() or (jacobian <= 1e-8).any():
        raise ValueError("Lens rectification has a folded/singular local Jacobian")
    # align_corners=False: integer source center u corresponds to (u+.5)/W.
    grid = np.stack((2*(u+.5)/w-1, 2*(v+.5)/h-1), axis=-1).astype(np.float32)
    output_k = np.asarray([fx*zoom, fy*zoom, cx, cy], dtype=np.float32)
    return output_k, torch.from_numpy(grid).unsqueeze(0), float(zoom)


def project_source_pixels(points_camera, data, source_hw, frame=None):
    """Project positive-z COLMAP points into the declared raw lens raster."""
    _, k, model, coeff = _spec(data, source_hw, frame)
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or (points[:, 2] <= 0).any():
        raise ValueError("Expected positive-z [N,3] camera points")
    xd, yd = _distort(points[:, 0]/points[:, 2], points[:, 1]/points[:, 2], model, coeff)
    return k[0]*xd+k[2], k[1]*yd+k[3]


def rectified_intrinsics(data, source_hw, frame=None):
    spec = _spec(data, source_hw, frame)
    k, _, _ = _rectification(*spec)
    return k.copy()


def rectify_camera_batch(images, data, frame_indices=None, *, mode="bilinear"):
    """Rectify RGB or aligned depth; return raster, per-frame PINHOLE K and audit.

    Call once at decode, before crop/resize, for context AND supervised/scored
    images. Depth uses mode='nearest' to avoid averaging across discontinuities.
    No zero border padding becomes a supervised/scored image region.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("Expected float [N,C,H,W] camera raster")
    frames = list(range(len(images))) if frame_indices is None else list(map(int, frame_indices))
    if len(frames) != len(images):
        raise ValueError("Camera frame count differs from decoded raster batch")
    if mode not in ("bilinear", "nearest"):
        raise ValueError("Unsupported camera raster interpolation")
    groups, intrinsics, audit = {}, [], []
    for i, frame in enumerate(frames):
        spec = _spec(data, images.shape[-2:], frame)
        k, grid, zoom = _rectification(*spec)
        intrinsics.append(k)
        groups.setdefault(spec, []).append(i)
        audit.append({"policy": CAMERA_PREPROCESSING, "frame": frame,
            "source_hw": list(images.shape[-2:]), "input_model": spec[2],
            "input_coefficients": dict(zip(COEFFICIENTS, spec[3])),
            "source_intrinsics": list(spec[1]), "output_model": "PINHOLE",
            "rectification_zoom": zoom, "rectified": grid is not None})
    result = images.clone() if any(spec[2] == "OPENCV_FISHEYE" or any(spec[3]) for spec in groups) else images
    for spec, positions in groups.items():
        _, grid, _ = _rectification(*spec)
        if grid is not None:
            device_grid = grid.to(device=images.device)
            for start in range(0, len(positions), 8):
                chunk = positions[start:start+8]
                selected = images[chunk].float()
                mapped = F.grid_sample(selected, device_grid.expand(len(chunk), -1, -1, -1),
                                       mode=mode, padding_mode="zeros", align_corners=False)
                result[chunk] = mapped.to(dtype=images.dtype)
    return result, torch.from_numpy(np.stack(intrinsics)).to(images.device), audit
