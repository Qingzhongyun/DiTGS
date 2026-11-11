"""One FM forward -> predicted latent -> frozen VAE decoder -> adapter -> TokenGS."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import MultiViewDiT, PerViewVAE, condition_latents, flow_training_pair
from .geometry import flashworld_camera_raymaps, tokengs_plucker_maps
from .adapter import DecoderFeatureAdapter
from .residual_attention import SequentialDecoderBlock, RESIDUAL_STAT_COLUMNS
from .gaussian_safety import prepare_gaussians
from .gs import GSLossMixin, _new_base_gs, _load_base_gs

def load_pretrained_component(model_class, path):
    model, info = model_class.from_pretrained(path, torch_dtype=torch.float32,
        local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
    issues = {key: info.get(key) for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if issues:
        raise ValueError(f"Pretrained component did not load exactly: {path}: {issues}")
    return model

class MultiViewModel(GSLossMixin, nn.Module):
    def __init__(self, cfg, initialize_weights=True):
        super().__init__()
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from .assets import load_empty_prompt, validate_model_configs
        self.cfg = cfg.validate()
        self.source_initialization = {
            "effective_training_initialization": cfg.init_checkpoint,
            "restore_policy": ("strict corrected warmup model/buffers; fresh optimizer/EMA/LR"
                if cfg.recipe == "joint" else "fresh original FlashWorld and TokenGS base1024 assets"),
            "official_checkpoint": cfg.official_checkpoint,
            "attention": cfg.dit_output_initialization,
            "kv": cfg.dit_kv_initialization,
            "adapter_final_norm": cfg.adapter_norm_initialization,
            "residual_post_norm": cfg.residual_post_norm,
            "layerscale": cfg.dit_residual_scaling,
            "layerscale_initialization": cfg.dit_layerscale_initialization,
            "copied_layers": 12,
            "copy_order": "official-native-load-then-copy-after-to_empty-before-FSDP-sync",
            "parameter_tying": False,
            "runtime_dit_residual_scaling": "learned per-channel gamma in each of12 layers",
            "runtime_dit_scalar_gate": False,
            "runtime_dit_layerscale": True,
            "output_projection_gamma_folded": False}
        self.activation_recorder = None
        dconfig, vconfig = validate_model_configs(cfg.model_dir)
        text = load_empty_prompt(cfg.model_dir) if initialize_weights else None
        root = Path(cfg.model_dir)
        if initialize_weights:
            dit = load_pretrained_component(WanTransformer3DModel, root/"transformer")
            vae = load_pretrained_component(AutoencoderKLWan, root/"vae")
        else:
            with torch.device("meta"):
                dit = WanTransformer3DModel.from_config(dconfig)
                vae = AutoencoderKLWan.from_config(vconfig)
        with nullcontext() if initialize_weights else torch.device("meta"):
            self.dit = MultiViewDiT(dit, cfg.activation_checkpointing)
            self.vae = PerViewVAE(vae, cfg.vae_view_batch, cfg.activation_checkpointing)
            self.adapter = DecoderFeatureAdapter(
                (cfg.image_size, cfg.image_width), self.vae.feature_channels, cfg.feature_dim)
            self.register_buffer("empty_text", torch.zeros(1, 512, 4096), persistent=True)
        if initialize_weights:
            from .flashworld import load_flashworld_transformer, apply_flashworld_latent_normalization
            self.flashworld_initialization = load_flashworld_transformer(self.dit.transformer, cfg)
            apply_flashworld_latent_normalization(self.vae, self.flashworld_initialization)
        with torch.device("meta"):
            self.gs = _new_base_gs(cfg)
        if initialize_weights:
            self.gs.to_empty(device="cpu")
            _load_base_gs(self.gs, self.adapter.ray_patch, cfg.official_checkpoint,
                          adapter_output_norm=self.adapter.output_norm)
            self.empty_text.copy_(text)
        self.configure_trainability()
        self.train()

    def configure_trainability(self):
        self.dit.requires_grad_(not self.cfg.freeze_dit)
        # Continuation unfreezes the pretrained WAN including camera/image conditioning.
        self.dit.transformer.patch_embedding.condition.requires_grad_(self.cfg.train_dit_condition)
        self.adapter.requires_grad_(self.cfg.use_gs)
        self.gs.requires_grad_(self.cfg.use_gs and not self.cfg.freeze_original_gs)
        if self.cfg.variant=="residual":
            # Native GS and the learned new-source modules now train together.
            self.gs.enc_dec_backbone.dit_kv_proj.requires_grad_(self.cfg.use_gs)
            self.gs.enc_dec_backbone.dit_k_proj_norm.requires_grad_(self.cfg.use_gs)
            for block in self.gs.enc_dec_backbone.decoder_blocks:
                block.dit_cross_attn.requires_grad_(self.cfg.use_gs)
                block.dit_cross_attn_scale.requires_grad_(self.cfg.use_gs)
        # The real-evidence encoder remains frozen in every stage.
        self.gs.patch_embed.requires_grad_(False)
        self.gs.patch_plucker_embed.requires_grad_(False)
        self.gs.enc_dec_backbone.encoder.requires_grad_(False)
        self.gs.enc_dec_backbone.encoder_norm.requires_grad_(False)
        self.vae.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.gs.patch_embed.eval()
        self.gs.patch_plucker_embed.eval()
        self.gs.enc_dec_backbone.encoder.eval()
        self.gs.enc_dec_backbone.encoder_norm.eval()
        return self

    def set_activation_recorder(self, callback):
        self.activation_recorder = callback
        self.gs.activation_head.activation_recorder = self._record if callback else None

    def _record(self, name, value):
        if self.activation_recorder is not None:
            self.activation_recorder(name, value)
            if value.requires_grad:
                callback = self.activation_recorder
                value.register_hook(lambda grad: (callback(name+"/backward", grad), grad)[1])

    def fsdp_block_types(self):
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock
        from tokengs.models.enc_dec import DecoderBlock
        from tokengs.models.attention import Block
        from .ablations.attention import SequentialDecoderBlock as SharedDecoderBlock
        return {WanTransformerBlock,DecoderBlock,SequentialDecoderBlock,SharedDecoderBlock,Block,PerViewVAE}

    def parameter_groups(self):
        from .residual_attention import is_new_dit_parameter
        groups = {"dit": {"params": [], "lr": self.cfg.lr_dit, "name": "dit"},
                  "new_branch": {"params": [], "lr": self.cfg.lr_gs, "name": "new_branch"},
                  "original_gs": {"params": [], "lr": self.cfg.lr_original_gs, "name": "original_gs"}}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            name = ".".join(x for x in name.split(".") if x not in ("module", "_fsdp_wrapped_module"))
            original_gs = name.startswith("gs.") and not (is_new_dit_parameter(name[3:]) or name.endswith(".completion_gate"))
            group = "dit" if name.startswith("dit.") else ("original_gs" if original_gs else "new_branch")
            groups[group]["params"].append(parameter)
        return [g for g in groups.values() if g["params"]]

    def freeze_manifest(self):
        return {name: {"trainable": p.requires_grad, "shape": list(p.shape), "numel": p.numel()}
                for name, p in self.named_parameters()}

    def source_gs_memories(self, features, images, c2w, intrinsics, observed_mask):
        """Same camera/raster contract; GS has two real views, DiT keeps ALL eight."""
        with torch.no_grad():
            rays = tokengs_plucker_maps(c2w, intrinsics, *images.shape[-2:])
            sparse = self.gs.sparse_image_features(images, rays, observed_mask)
        generated = self.adapter(features, rays)
        if self.cfg.variant!="residual":
            from .ablations.attention import missing_view_features
            generated=missing_view_features(generated,observed_mask)
        self._record("sparse_encoder_memory", sparse)
        self._record("dit_eight_view_memory", generated)
        return sparse, generated

    def _guard(self, name, value, bound):
        detached = value.detach()
        good = bool(torch.isfinite(detached).all()) and float(detached.abs().amax()) <= bound
        if good:
            return value
        self.numerical_rejections.append(name)
        # Only finish the distributed graph; the entire global batch is discarded.
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * 0

    def forward(self, batch, step=0, generator=None, timestep_generator=None,
                loss_mode=None, capture_monitor=False):
        if loss_mode not in (None, "fm", "img", "gs"):
            raise ValueError("Unknown loss component")
        self.numerical_rejections = []
        gaussian_statistics = {}
        use_img = loss_mode == "img" or (loss_mode is None and self.cfg.use_img)
        use_gs = loss_mode == "gs" or (loss_mode is None and self.cfg.use_gs)
        images, observed = batch["images"], batch["observed_mask"]
        if images.shape[1] != self.cfg.dense_views or not (observed.sum(1) == 2).all():
            raise ValueError("Expected configured view count and two observed images")
        with torch.no_grad():
            target = self._guard("encoded_latent", self.vae(images, generator=generator), self.cfg.latent_abs_max)
            condition = condition_latents(target, observed)
            rays = flashworld_camera_raymaps(batch["c2w"], batch["intrinsics_all"],
                *images.shape[-2:], *target.shape[-2:], scene_scale=self.cfg.scene_scale,
                normalization_eps=self.cfg.dit_camera_normalization_eps)
        if target.shape[2] != images.shape[1] or target.shape[-2:] != (images.shape[-2]//16, images.shape[-1]//16):
            raise ValueError("Wan VAE violated the one-view-one-latent contract")
        noise = torch.randn(target.shape, dtype=torch.float32, device=target.device, generator=generator)
        sigma = torch.rand((images.shape[0],), device=target.device, generator=(
            timestep_generator if timestep_generator is not None else generator))
        state, velocity_target = flow_training_pair(target.float(), noise, sigma)
        velocity = self._guard("velocity", self.dit(state, sigma, condition, rays, self.empty_text), self.cfg.latent_abs_max)
        clean = self._guard("clean_estimate", state-sigma[:, None, None, None, None]*velocity, self.cfg.latent_abs_max)
        fm = F.mse_loss(velocity.float(), velocity_target)
        losses, probe = {"fm": fm}, {}
        if use_img or use_gs:
            if use_img:
                features, predicted_rgb = self.vae(clean, operation="decode_both")
                predicted_rgb = self._guard("vae_rgb", predicted_rgb, self.cfg.feature_abs_max)
                losses["img"] = F.mse_loss(predicted_rgb.float(), images.float())
                self._record("vae_rgb", predicted_rgb)
            else:
                features = self.vae(clean, operation="decode_features")
            features = self._guard("vae_features", features, self.cfg.feature_abs_max)
            self._record("vae_decoder_features", features)
        if use_gs:
            sparse, generated = self.source_gs_memories(
                features, images, batch["c2w"], batch["intrinsics_all"], observed)
            sparse = self._guard("gs_real_memory", sparse, self.cfg.feature_abs_max)
            generated = self._guard("dit_memory", generated, self.cfg.feature_abs_max)
            patches = (images.shape[-2] // 8) * (images.shape[-1] // 8)
            if (sparse.shape != (images.shape[0], 2*patches, self.cfg.feature_dim) or
                    generated.shape != (images.shape[0], self.cfg.dit_memory_views*patches, self.cfg.feature_dim)):
                raise ValueError("Require same-grid 2-view GS and complete 8-view DiT memories")
            head = getattr(self.gs, "activation_head", None)
            detailed = step % self.cfg.diagnostic_capture_every == 0
            if head is not None:
                head.detailed_statistics = detailed
            capture_residuals = step % self.cfg.log_every == 0
            gaussians = self.gs.gaussians_from_source_features(
                sparse, generated, capture_stats=capture_residuals)
            gaussians, self.gs_active_mask, gaussian_statistics = prepare_gaussians(
                gaussians, detailed=detailed)
            if head is not None:
                gaussian_statistics.update(getattr(head, "gaussian_statistics", {}))
                invalid = getattr(head, "local_invalid_mask", None)
                if invalid is not None:
                    self.gs_active_mask = self.gs_active_mask & ~invalid
            gaussian_statistics["active_points"] = int(self.gs_active_mask.sum().item())
            gaussian_statistics["quarantined_points"] = self.gs_active_mask.numel() - gaussian_statistics["active_points"]
            gaussian_statistics["quarantined_fraction"] = gaussian_statistics["quarantined_points"] / self.gs_active_mask.numel()
            self._record("gs_real_features", sparse)
            self._record("gs_dit_features", generated)
            self._record("gaussians", gaussians)
            if self.numerical_rejections:
                losses["gs"] = gaussians.float().sum() * 0
            else:
                losses.update(self._gs_loss(gaussians, batch, step))
                gaussian_statistics.update(getattr(self, "gs_render_statistics", {}))
            if capture_monitor:
                probe["gaussians"] = gaussians.detach()
        loss = losses[loss_mode] if loss_mode else sum(
            getattr(self.cfg, key+"_weight") * losses[key] for key in ("fm", "img", "gs") if key in losses)
        self._record("velocity", velocity)
        self._record("clean_estimate", clean)
        if capture_monitor:
            probe.update(clean_estimate=clean.detach(), velocity=velocity.detach())
        diagnostics = {"time_mean": sigma.detach().mean(), "dit_forward_calls": sigma.new_tensor(1),
                       "training_rollout": sigma.new_tensor(0)}
        if use_gs and capture_residuals:
            diagnostics["residual_rms"] = self.gs.residual_statistics()
            if self.cfg.variant=="residual":
                diagnostics["residual_rms_columns"]=RESIDUAL_STAT_COLUMNS
            else:
                from .ablations.attention import RESIDUAL_STAT_COLUMNS as SHARED_COLUMNS
                diagnostics["residual_rms_columns"]=SHARED_COLUMNS
        return {"loss": loss, "losses": {k: v.detach() for k,v in losses.items()},
            "sample_timesteps": sigma.detach(), "probe_tensors": probe,
            "numerical_rejections": list(self.numerical_rejections),
            "gaussian_statistics": gaussian_statistics,
            "diagnostics": diagnostics}
