"""DL3DV ZIP images and corrected per-frame cameras."""
from io import BytesIO
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import torch
from tokengs.data.datafield import DF_CAMERA_C2W_TRANSFORM,DF_CAMERA_INTRINSICS,DF_DEPTH,DF_IMAGE_RGB
from tokengs.data.camera_preprocessing import intrinsics_for_raster,rectify_camera_batch

class DL3DVArchive:
    """Official DL3DV zip reader with flat/rooted archive compatibility."""

    def __init__(self, root_path, subset, resolution="960p", **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported Dense16 DL3DV kwargs: {sorted(kwargs)}")
        self.root_path = Path(root_path)
        self.subset = tuple(subset)
        if resolution != "960p":
            raise ValueError("Dense16 DL3DV uses official images_4 at 960p")
        self.sample_list = []
        for group in self.subset:
            pattern = "*.zip" if group == "140" else f"{group}/*.zip"
            self.sample_list.extend(sorted(self.root_path.glob(pattern)))
        if not self.sample_list:
            raise FileNotFoundError(f"No DL3DV scene archives under {self.root_path}")
        self.is_static = True

    def __len__(self):
        return len(self.sample_list)

    def count_cameras(self, video_idx):
        del video_idx
        return 1

    @staticmethod
    def _layout(zip_handle, archive_path):
        stem = archive_path.stem
        for transforms_path, prefix in (
            ("transforms.json", ""),
            (f"{stem}/transforms.json", f"{stem}/"),
            (f"{stem}/gaussian_splat/transforms.json", f"{stem}/gaussian_splat/"),
        ):
            try:
                with zip_handle.open(transforms_path, "r") as handle:
                    return prefix, json.load(handle)
            except KeyError:
                continue
        raise FileNotFoundError(f"{archive_path} has no supported transforms.json layout")

    @staticmethod
    def _intrinsics(data, source_hw=None, frame=None):
        # Compatibility utility: no nominal 960x540 assumption.
        return intrinsics_for_raster(data, source_hw, frame)

    @staticmethod
    def _cameras(data):
        transforms = []
        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            transforms.append(c2w)
        cameras = np.stack(transforms, axis=0)
        if cameras.shape[1:] != (4, 4) or not np.isfinite(cameras).all():
            raise ValueError("DL3DV archive has malformed or non-finite camera transforms")
        # Validate the entire archive, not just this draw: otherwise a corrupt
        # scene is accepted until its exceptional frames happen to be sampled.
        # Use float64 so the validation norm itself cannot overflow in float32.
        translation_norm = np.linalg.norm(cameras[:, :3, 3].astype(np.float64), axis=-1)
        invalid = np.flatnonzero(translation_norm > 1e9)
        if invalid.size:
            raise ValueError(
                f"DL3DV archive has absurd camera translation: "
                f"max_norm={translation_norm.max():.6g}, frames={invalid[:16].tolist()}")
        return cameras

    @staticmethod
    def _image_bytes(zip_handle, prefix, file_path):
        file_path = str(file_path).lstrip("./")
        paths = [file_path]
        if file_path.startswith("images/"):
            paths.append("images_4/" + file_path[len("images/"):])
        candidates = []
        for path in paths:
            candidates.append(path)
            if prefix:
                candidates.append(f"{prefix}{path}")
        for candidate in candidates:
            try:
                with zip_handle.open(candidate, "r") as handle:
                    return handle.read()
            except KeyError:
                continue
        raise FileNotFoundError(f"DL3DV archive has no image file for {file_path}")

    def count_frames(self, idx):
        archive = Path(self.sample_list[idx])
        with zipfile.ZipFile(archive, "r") as zip_handle:
            _, data = self._layout(zip_handle, archive)
        return len(data["frames"])

    def get_data(self, idx, data_fields, frame_indices=None, view_indices=None,
                 camera_convention="opencv", num_depth_frames=None):
        del view_indices, num_depth_frames
        if camera_convention != "opencv":
            raise ValueError(f"Dense16 DL3DV expects OpenCV convention, got {camera_convention}")
        if DF_DEPTH in data_fields:
            raise NotImplementedError("Dense16 does not request DL3DV depth fields")
        archive = Path(self.sample_list[idx])
        with zipfile.ZipFile(archive, "r") as zip_handle:
            prefix, data = self._layout(zip_handle, archive)
            total = len(data["frames"])
            selected = list(range(total)) if frame_indices is None else [int(value) for value in frame_indices]
            if not selected or min(selected) < 0 or max(selected) >= total:
                raise IndexError(f"DL3DV frame selection {selected} is invalid for {total} frames")
            # Reject corrupt metadata before decoding images. Keep sample_list
            # unchanged so checkpoint cursors and deterministic indices retain
            # their meaning; the provider logs and replaces bad training scenes.
            try:
                cameras = self._cameras(data)
            except ValueError as error:
                raise ValueError(f"{archive}: {error}") from error
            images = []
            for frame_index in selected:
                file_path = data["frames"][frame_index]["file_path"]
                with Image.open(BytesIO(self._image_bytes(zip_handle, prefix, file_path))) as image:
                    images.append(torch.from_numpy(
                        np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
        rgbs = torch.stack(images).float().div_(255.0)
        c2ws = torch.from_numpy(cameras[selected]).float().contiguous()
        rgbs, intrinsics, camera_info = rectify_camera_batch(rgbs, data, selected)
        output = {"__key__": archive.stem, "camera_preprocessing": camera_info}
        if DF_IMAGE_RGB in data_fields:
            output[DF_IMAGE_RGB] = rgbs
        if DF_CAMERA_C2W_TRANSFORM in data_fields:
            output[DF_CAMERA_C2W_TRANSFORM] = c2ws
        if DF_CAMERA_INTRINSICS in data_fields:
            output[DF_CAMERA_INTRINSICS] = intrinsics
        return output
