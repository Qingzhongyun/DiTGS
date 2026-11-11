"""Rectified pinhole benchmark; target RGB remains inaccessible until scoring."""
from io import BytesIO
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import torch
from tokengs.data.camera_preprocessing import rectify_camera_batch, rectified_intrinsics
from .eval_protocol import HEIGHT, WIDTH, zip_layout
from .geometry import validate_cameras


class BenchmarkData:
    def __init__(self, root, size=(HEIGHT, WIDTH), scene_scale=.15):
        from tokengs.utils.data import ImageTransform
        self.root = Path(root)
        self.transform = ImageTransform(crop_size=size, sample_size=size, max_crop=True)
        self.size = tuple(size)
        self.scene_scale = scene_scale

    def _read_rgb(self, z, prefix, meta, frames):
        from ditgs.archive import DL3DVArchive
        images = []
        for frame in frames:
            raw = DL3DVArchive._image_bytes(z, prefix, meta["frames"][frame]["file_path"])
            with Image.open(BytesIO(raw)) as image:
                images.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1))
        if len({tuple(x.shape) for x in images}) != 1:
            raise ValueError("Image raster changes within benchmark scene")
        return rectify_camera_batch(torch.stack(images).float()/255, meta, frames)

    def load_condition(self, row):
        from ditgs.archive import DL3DVArchive
        path = self.root/(row["scene_id"]+".zip")
        with zipfile.ZipFile(path) as z:
            prefix, meta = zip_layout(z, path)
            # ONLY released context RGB; all other cameras use metadata alone.
            images, _, camera_info = self._read_rgb(z, prefix, meta, row["context"])
        source_hw = tuple(images.shape[-2:])
        images, shift, scale, _ = self.transform.preprocess_images(images)
        cameras = torch.from_numpy(DL3DVArchive._cameras(meta)[:50]).float()
        cameras = torch.linalg.inv(cameras[row["context"][0]])[None] @ cameras
        cameras[:, :3, 3] *= self.scene_scale
        k = torch.from_numpy(np.stack([rectified_intrinsics(meta, source_hw, i) for i in range(50)]))
        k = torch.stack((k[:, 0]*scale[0], k[:, 1]*scale[1],
            (k[:, 2]+shift[0])*scale[0], (k[:, 3]+shift[1])*scale[1]), -1)
        validate_cameras(cameras, k)
        if images.shape != (2, 3, *self.size) or not torch.isfinite(images).all():
            raise ValueError("Invalid context RGB")
        return {"context_images": images.unsqueeze(0), "c2w": cameras, "intrinsics": k,
                "source_hw": source_hw, "camera_preprocessing": camera_info,
                "crop_shift_xy": list(shift), "crop_scale_xy": list(scale)}

    def load_gt(self, row, expected_source_hw=None):
        path = self.root/(row["scene_id"]+".zip")
        with zipfile.ZipFile(path) as z:
            prefix, meta = zip_layout(z, path)
            rgb, _, _ = self._read_rgb(z, prefix, meta, row["target"])
        if expected_source_hw is not None and tuple(rgb.shape[-2:]) != tuple(expected_source_hw):
            raise ValueError("Context/target source raster changed: "+row["scene_id"])
        images, _, _, _ = self.transform.preprocess_images(rgb)
        if images.shape != (50, 3, *self.size) or not torch.isfinite(images).all():
            raise ValueError("Invalid benchmark GT RGB")
        return images
