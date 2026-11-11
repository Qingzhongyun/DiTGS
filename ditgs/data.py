"""Eight-view DL3DV sampling with logged, unbounded real-scene recovery."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate
from tokengs.options import Options
from tokengs.data.provider import Provider
from tokengs.data.registry import dataset_registry
from tokengs.data.datafield import DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS
from ditgs.archive import DL3DVArchive
from ditgs.ipc import configure_loader_ipc, configure_loader_worker
from .geometry import validate_cameras

REGISTRY_NAME = "ditgs_wan_per_view_dl3dv"
def selection_seed(seed, step, index):
    return int(seed) + int(step)*1000003 + int(index)*97409

def select_ordered_frames(frame_count, seed, views=8, min_gap=7, max_gap=40):
    if frame_count < views or min_gap < views-1 or max_gap < min_gap:
        raise ValueError("Insufficient distinct frames / invalid view interval")
    rng = np.random.default_rng(int(seed))
    gap = int(rng.integers(min(min_gap, frame_count-1), min(max_gap, frame_count-1)+1))
    start = int(rng.integers(0, frame_count-gap))
    return np.rint(np.linspace(start, start+gap, views)).astype(np.int64)

def observed_mask_for(views):
    return tuple(i in (0, views-1) for i in range(views))

def render_indices_for(seed, step, index, views=8, renders=8):
    """Full supervision; seed/address no longer select a render subset."""
    if (views, renders) != (8, 8):
        raise ValueError("This experiment must render all eight training cameras")
    return tuple(range(views))

def sample_addresses(size, seed, step, world_size=16, batch_size=1):
    if size < world_size*batch_size:
        raise ValueError("Insufficient unique scenes for a global batch")
    return random.Random(int(seed)+15485863*int(step)).sample(range(size), world_size*batch_size)

class PinnedArchives(DL3DVArchive):
    def __init__(self, paths):
        self.sample_list = [Path(p) for p in paths]
        if not self.sample_list:
            raise ValueError("Empty training cohort")
        self.is_static = True

    @staticmethod
    def _cameras(data):
        cameras = DL3DVArchive._cameras(data)
        # Validate all rotations, including frames not selected on this draw.
        c = torch.from_numpy(cameras)
        fake_k = c.new_tensor([1, 1, 0, 0]).expand(len(c), -1)
        validate_cameras(c, fake_k)
        return cameras

def load_training_paths(cfg):
    cfg.validate(check_files=True)
    rows=Path(cfg.training_list).read_text().splitlines()
    root=Path(cfg.data_root).expanduser().resolve()
    paths=[]
    for row in rows:
        if not row.strip():continue
        relative=Path(row.strip())
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Training split paths must be relative")
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError("Training path leaves dataset root")
        paths.append(str(path))
    if len(paths)!=len(set(paths)) or len(paths)<cfg.world_size*cfg.batch_size:raise ValueError("Invalid/duplicate/insufficient training scenes")
    return paths


class MultiViewProvider(Provider):
    def __init__(self, cfg):
        self.cfg = cfg.validate()
        paths = load_training_paths(cfg)
        if REGISTRY_NAME in dataset_registry and dataset_registry[REGISTRY_NAME]["cls"] is not PinnedArchives:
            raise RuntimeError("Dataset registration conflict")
        dataset_registry[REGISTRY_NAME] = {"cls": PinnedArchives, "kwargs": {},
            "scene_scale": cfg.scene_scale, "min_gap": cfg.min_gap, "max_gap": cfg.max_gap}
        opt = Options(seed=cfg.seed, workspace=cfg.workspace, project_name="DiTGS-Wan",
            img_size=(cfg.image_size, cfg.image_width), num_input_views=cfg.dense_views,
            num_views=cfg.dense_views, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            camera_normalization_method="first_cam", camera_scale_method="constant",
            random_reflect=False, time_embedding=False, evaluating=True,
            use_input_supervision=False, use_multiscale_encoder=False, use_latent_bottleneck=False,
            dataset_kwargs={"paths": paths})
        super().__init__(REGISTRY_NAME, opt, training=True, num_repeat=1)
        self.scene_scale = cfg.scene_scale
        self.current_step = self.current_rank = 0
        self._selected = None
        self.scene_fingerprint = hashlib.sha256("".join(str(p)+"\n" for p in self.dataset.sample_list).encode()).hexdigest()

    def _get_indices_static(self, idx):
        self._selected = select_ordered_frames(self.dataset.count_frames(idx),
            selection_seed(self.cfg.seed, self.current_step, idx)
            + getattr(self, "_replacement_attempt", 0) * 32452843, self.cfg.dense_views,
            self.cfg.min_gap, self.cfg.max_gap)
        return self._selected, []

    def get_item(self, idx):
        frame_ids, _ = self._get_indices_static(idx)
        raw = self.dataset.get_data(idx, data_fields=[
            DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS],
            frame_indices=frame_ids, view_indices=[])
        rgbs, c2ws, intrinsics = raw[DF_IMAGE_RGB], raw[DF_CAMERA_C2W_TRANSFORM], raw[DF_CAMERA_INTRINSICS]
        validate_cameras(c2ws, intrinsics)
        if not torch.isfinite(rgbs).all() or rgbs.min() < 0 or rgbs.max() > 1:
            raise ValueError("Invalid RGB range / nonfinite RGB")
        # Preserve TokenGS's exact center max-crop / resize and coordinate gauge,
        # without constructing its removed image encoder's RGB+Pluecker input.
        source_hw = list(rgbs.shape[-2:])
        rgbs, shift, scale, _ = self.image_transform.preprocess_images(rgbs)
        intrinsics = torch.stack((intrinsics[..., 0]*scale[0], intrinsics[..., 1]*scale[1],
            (intrinsics[..., 2]+shift[0])*scale[0], (intrinsics[..., 3]+shift[1])*scale[1]), dim=-1)
        c2ws = torch.linalg.inv(c2ws[0]).unsqueeze(0) @ c2ws
        c2ws[:, :3, 3] *= self.scene_scale
        validate_cameras(c2ws, intrinsics)
        h, w = rgbs.shape[-2:]
        focal_ratio = intrinsics[..., :2] / intrinsics.new_tensor([w, h])
        if ((focal_ratio < self.cfg.focal_ratio_min) | (focal_ratio > self.cfg.focal_ratio_max)).any():
            raise ValueError("Finite but implausible focal length / image size")
        principal = intrinsics[..., 2:] / intrinsics.new_tensor([w, h])
        if (principal.abs() > 10).any():
            raise ValueError("Finite but implausible principal point")
        extent = torch.linalg.vector_norm(c2ws[:, :3, 3].double(), dim=-1)
        if extent.max() > self.cfg.camera_translation_max or extent.max() < self.cfg.camera_baseline_min:
            raise ValueError("Finite but extreme/collapsed camera baseline after normalization")
        self._crop_info = {"source_hw": source_hw, "output_hw": list(rgbs.shape[-2:]),
                           "shift_xy": list(shift), "scale_xy": list(scale),
                           "camera_preprocessing": raw["camera_preprocessing"]}
        return {"images_all": rgbs, "c2w": c2ws,
                "cam_view_all": torch.linalg.inv(c2ws).transpose(-1, -2), "intrinsics_all": intrinsics}

    def __getitem__(self, address):
        idx, step, rank = (int(address), 0, 0) if isinstance(address, (int, np.integer)) else map(int, address)
        if not 0 <= idx < len(self.dataset):
            raise IndexError(address)
        self.current_step, self.current_rank = step, rank
        requested_idx = idx
        if len(self.dataset) >= self.cfg.world_size * self.cfg.batch_size:
            reserved = set(sample_addresses(len(self.dataset), self.cfg.seed, step,
                self.cfg.world_size, self.cfg.batch_size)) | {idx}
            candidates = [i for i in range(len(self.dataset)) if i not in reserved]
            random.Random(self.cfg.seed + step*15485863 + 9143521).shuffle(candidates)
            replacements = candidates[rank::self.cfg.world_size]
        else:  # Small synthetic fixtures; real launch preflight requires a full global batch.
            candidates = [i for i in range(len(self.dataset)) if i != idx]
            random.Random(selection_seed(self.cfg.seed, step, idx)+9143521+rank).shuffle(candidates)
            replacements = candidates
        if not replacements:
            replacements = [requested_idx]
        attempt = 0
        while True:
            idx = requested_idx if attempt == 0 else replacements[(attempt - 1) % len(replacements)]
            self._replacement_attempt = attempt
            self.generator.manual_seed(selection_seed(self.cfg.seed, step, idx) % (2**63-1))
            self._selected = None
            archive = str(self.dataset.sample_list[idx])
            try:
                out = self.get_item(idx)
                break
            except Exception as error:
                folder = Path(self.cfg.workspace)/"data_errors"
                folder.mkdir(parents=True, exist_ok=True)
                record = {"protocol": self.cfg.protocol, "step": step, "rank": rank,
                    "requested_index": requested_idx, "actual_index": idx, "source_path": archive,
                    "frame_ids": None if self._selected is None else self._selected.tolist(),
                    "replacement_attempt": attempt, "action": "resample_real_scene_until_valid",
                    "error": f"{type(error).__name__}: {error}"}
                with (folder/f"rank{rank}_pid{os.getpid()}.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")
                    f.flush()
                print(f"Data recovery rank={rank} step={step} attempt={attempt} "
                      f"scene={archive}: {type(error).__name__}: {error}", flush=True)
                attempt += 1
        known = observed_mask_for(self.cfg.dense_views)
        renders = render_indices_for(self.cfg.seed, step, idx, self.cfg.dense_views, self.cfg.gs_render_views)
        noise_seed = selection_seed(self.cfg.seed, step, idx)+49979687
        timestep_seed = selection_seed(self.cfg.seed, step, idx)+67867967
        metadata = {"protocol": self.cfg.protocol, "recipe_step": step, "rank": rank,
            "requested_index": requested_idx, "actual_index": idx, "replacement_attempt": attempt,
            "valid_case": True,
            "source_path": archive, "frame_ids": self._selected.tolist(), "observed_mask": known,
            "gs_render_indices": renders, "noise_seed": noise_seed, "timestep_seed": timestep_seed,
            "scene_fingerprint": self.scene_fingerprint, "quality_manifest_sha256": self.cfg.quality_manifest_sha256,
            "layout": "one_image_one_latent_no_temporal_compression", "scene_scale": self.scene_scale,
            "crop": self._crop_info,
            "intrinsics_xy": out["intrinsics_all"].tolist(), "cam_view_transposed": out["cam_view_all"].tolist()}
        # Rebuild original normalized RGB/rays from these same cropped cameras
        # inside the frozen sparse encoder, selecting only observed RGB first.
        return {"images": out["images_all"], "images_all": out["images_all"],
            "c2w": out["c2w"], "cam_view_all": out["cam_view_all"], "intrinsics_all": out["intrinsics_all"],
            "observed_mask": torch.tensor(known, dtype=torch.bool),
            "gs_render_indices": torch.tensor(renders, dtype=torch.int64),
            "frame_ids": torch.from_numpy(self._selected.copy()), "sample_index": idx,
            "requested_sample_index": requested_idx, "recipe_step": step, "valid_case": True,
            "noise_seed": noise_seed, "timestep_seed": timestep_seed,
            "metadata_json": json.dumps(metadata, sort_keys=True)}

class StepSampler(Sampler):
    def __init__(self, size, cfg, rank, world_size, start_step=0, stop_step=None):
        if world_size != cfg.world_size or not 0 <= rank < world_size:
            raise ValueError("Distributed data topology mismatch")
        self.size, self.cfg, self.rank, self.world_size = size, cfg, rank, world_size
        self.start, self.stop = start_step, cfg.steps if stop_step is None else stop_step
        if not 0 <= self.start <= self.stop:
            raise ValueError("Invalid recipe step range")
    def __iter__(self):
        for step in range(self.start, self.stop):
            ids = sample_addresses(self.size, self.cfg.seed, step, self.world_size, self.cfg.batch_size)
            yield [(i, step, self.rank) for i in ids[self.rank*self.cfg.batch_size:(self.rank+1)*self.cfg.batch_size]]
    def __len__(self):
        return self.stop-self.start

def make_dataset(cfg):
    return MultiViewProvider(cfg)

def make_dataloader(cfg, rank, world_size, start_step=0, stop_step=None, dataset=None):
    configure_loader_ipc()
    dataset = make_dataset(cfg) if dataset is None else dataset
    return DataLoader(dataset, batch_sampler=StepSampler(len(dataset), cfg, rank, world_size, start_step, stop_step),
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=False, collate_fn=default_collate,
        worker_init_fn=configure_loader_worker, prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
        generator=torch.Generator().manual_seed(cfg.seed+rank))
