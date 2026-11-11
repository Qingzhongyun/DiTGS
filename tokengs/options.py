# Modified for DiTGS: static reconstruction integration and calibrated-camera support.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native TokenGS architecture and rendering options used by DiTGS."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any,Literal
import tyro

@dataclass
class Options:
    # --- general
    evaluating: bool = False
    workspace: str = "./workspace"
    resume: str | None = None
    model_type: str = "tokengs"
    seed: int = 42

    # --- wandb / logging
    use_wandb: bool = False
    experiment_name: str = "tokengs"
    out_dir: str = "outputs"
    project_name: str = "TokenGS"

    # --- model architecture
    img_size: tuple[int, int] = (256, 256)
    patch_size: int = 8
    dec_patch_size: int | None = None
    enc_depth: int = 3
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: int = 4
    clip_head_readout_std: float = 0.002
    clip_head_z_init: float | None = None
    dec_init_values: float | None = None

    # --- gaussian splatting
    bg_color: Literal["white", "black", "grey"] = "grey"
    gaussian_scale_cap: float = 0.075
    opacity_bias: float = 2.0
    gaussian_z_offset: float = 1.0
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    gs_token_std: float = 1e-2
    num_dynamic_gs_tokens: int = 0
    init_dynamic_tokens_from_static: bool = False
    init_tokens_from_existing: bool = False
    init_latents_from_existing: bool = False

    # --- dataset
    data_mode: tuple[tuple[str, int], ...] = (("dl3dv_scaled_0.15", 6),)
    num_views: int = 8
    num_input_views: int = 4
    znear: float = 0.025
    zfar: float = 125.0
    camera_normalization_method: Literal["mean_cam", "first_cam"] = "first_cam"
    camera_scale_method: Literal["constant", "distance", "bound", "pointmap"] = "constant"
    pointmap_trim_lo: float = 0.0
    pointmap_trim_hi: float = 1.0
    num_workers: int = 16
    dataset_kwargs: dict[str, str] | None = None

    # --- training
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 30
    max_iters_per_epoch: int = 1_000_000
    lr: float = 4e-4
    pct_start_steps: int = 1000
    final_div_factor: float = 1000.0
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    deferred_bp: bool = False
    use_input_supervision: bool = False
    mean_of_grads: Literal["none", "per-scene", "per-view"] = "none"
    mean_of_grads_scene_chunk_size: int = 1
    mean_of_grads_view_chunk_size: int | None = None

    # --- loss weights
    rgb_loss_type: Literal["l1", "l2"] = "l2"
    lambda_rgb: float = 1.0
    lambda_lpips: float = 0.0
    lambda_mask: float = 0.0
    lambda_ssim: float = 0.2
    lambda_visibility: float = 1.0
    visibility_distance_threshold: float = 1.0
    lambda_opacity: float = 0.0
    lambda_dyn_aux: float = 0.0
    lambda_dyn_aux_warmup_steps: int = 0
    lambda_dyn_aux_decay_steps: int = 0
    lambda_dyn_aux_min: float = 0.0

    # --- logging frequency
    print_freq: int = 10
    log_image_freq: int = 100

    # --- evaluation
    eval_n_media_dumps: int = 0
    strict_checkpoint_loading: bool = True

    # --- test-time training (eval)
    use_ttt_for_eval: bool = False
    ttt_mode: Literal["token-tuning", "scene-latent-tuning", "tokens", "latents"] = "token-tuning"
    ttt_n_steps: int = 50
    ttt_lr: float = 1e-4

    # --- dynamic scenes
    time_embedding: bool = False
    time_embedding_dim: int = 2
    use_interp_target: bool = False

    # --- latent bottleneck release architecture
    use_multiscale_encoder: bool = False
    multiscale_encoder_layers: tuple[int, ...] = (5, 7, 9, 11)
    use_latent_bottleneck: bool = False
    num_latents: int = 4096
    latent_cross_attn_depth: int = 12

    # --- augmentation
    random_reflect: bool = True

    def __post_init__(self) -> None:
        if self.dec_patch_size is None:
            self.dec_patch_size = self.patch_size
        self.validate()

    def validate(self) -> None:
        if self.evaluating:
            assert not self.use_input_supervision, "use_input_supervision must be False when evaluating"
        if self.mean_of_grads not in ("none", "per-scene", "per-view"):
            raise ValueError("mean_of_grads must be one of: none, per-scene, per-view")
        if self.deferred_bp and self.mean_of_grads != "none":
            raise ValueError("deferred_bp and mean_of_grads are mutually exclusive backprop strategies")
        if self.mean_of_grads_scene_chunk_size <= 0:
            raise ValueError("mean_of_grads_scene_chunk_size must be positive")
        if self.mean_of_grads_view_chunk_size is not None and self.mean_of_grads_view_chunk_size <= 0:
            raise ValueError("mean_of_grads_view_chunk_size must be positive")

    def evolve(self, **changes: Any) -> Options:
        """Return a deep copy with the given fields replaced."""
        new_instance = copy.deepcopy(self)
        for key, value in changes.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"Options has no attribute '{key}'")
            setattr(new_instance, key, value)
        new_instance.validate()
        return new_instance
