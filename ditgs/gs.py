"""TokenGS decoder, activation head and exact per-view renderer gradients."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from safetensors import safe_open
from .gaussian_safety import repair_renderer_gradient

def fp32_scope(x):
    return torch.autocast(device_type=x.device.type, enabled=False)

def _new_base_gs(cfg):
    from tokengs.options import Options
    from .residual_attention import SequentialSourceBackbone
    opt = Options(seed=cfg.seed, workspace=cfg.workspace, project_name="DiTGS-FlashWorld-LayerScale-CopyInit",
        enc_depth=3, dec_depth=12, enc_embed_dim=1024, token_dim=1024, enc_num_heads=16,
        num_gs_tokens=1024, img_size=(cfg.image_size, cfg.image_width), patch_size=8,
        dec_patch_size=8, use_latent_bottleneck=False, use_multiscale_encoder=False,
        time_embedding=False, num_dynamic_gs_tokens=0, lambda_lpips=0.,
        deferred_bp=False, mean_of_grads="none")
    if cfg.variant=="residual":
        core=SequentialSourceBackbone(opt,cfg.activation_checkpointing)
    else:
        from .ablations.attention import SequentialSourceBackbone as SharedBackbone
        core=SharedBackbone(opt,cfg.activation_checkpointing,cfg.variant)
    from .gaussian_safety import LocalSafeActivationHead
    core.activation_head = LocalSafeActivationHead(opt)
    return core


def _load_base_gs(core, ray_patch, path, adapter_output_norm=None):
    """Strict official native-key load, then independent nonzero source copies."""
    if getattr(core,"attention_variant","residual")!="residual":
        from .ablations.initialization import load_base_gs
        return load_base_gs(core,ray_patch,path,adapter_output_norm)
    from .residual_attention import is_new_dit_parameter, copy_native_source_initialization
    all_state = dict(core.state_dict())
    native = {key: value for key, value in all_state.items() if not is_new_dit_parameter(key)}
    added = {key for key in all_state if is_new_dit_parameter(key)}
    if len(core.enc_dec_backbone.decoder_blocks) != 12 or not added:
        raise ValueError("Sequential decoder topology differs from the base1024 contract")
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(native):
            raise ValueError(f"Official GS key mismatch missing={set(native)-keys}, unexpected={keys-set(native)}")
        with torch.no_grad():
            for key, destination in native.items():
                value = handle.get_tensor(key)
                if value.shape != destination.shape:
                    raise ValueError("Official GS tensor shape mismatch: " + key)
                destination.copy_(value)
            for key, destination in ray_patch.state_dict().items():
                value = handle.get_tensor("patch_plucker_embed." + key)
                if value.shape != destination.shape:
                    raise ValueError("Adapter ray embedding shape mismatch: " + key)
                destination.copy_(value)
    # Native tensors are now fully restored. Explicitly initialize EVERY added
    # tensor after meta/to_empty, then synchronize rank0 state through FSDP.
    copy_native_source_initialization(core, adapter_output_norm)

class GSLossMixin:
    def _gs_loss(self, gaussians, batch, step):
        from tokengs.rendering.fused_ssim import fused_ssim
        from .geometry_visibility import frustum_visibility
        gradient = torch.zeros_like(gaussians, dtype=torch.float32)
        image_gradient = torch.zeros_like(gradient)
        visibility_gradient = torch.zeros_like(gradient)
        alpha_means, alpha_coverages, geometry_rows = [], [], []
        self.gs_render_statistics = {"renderer_gradient_rows_zeroed": 0,
            "visibility_gradient_rows_zeroed": 0, "rendered_views": 0}
        zero = gaussians.new_zeros((), dtype=torch.float32)
        rgb_mean, ssim_mean, visibility_mean = zero.clone(), zero.clone(), zero.clone()
        group_sum = {k: zero.clone() for k in ("observed", "missing")}
        group_count = {k: 0 for k in group_sum}
        b = gaussians.shape[0]
        indices = batch["gs_render_indices"]
        if indices.shape != (b, self.cfg.gs_render_views):
            raise ValueError("Expected configured render indices per scene")
        active_mask = getattr(self, "gs_active_mask", None)
        for scene in range(b):
            active = (active_mask[scene] if active_mask is not None else
                      torch.ones(gaussians.shape[1], dtype=torch.bool, device=gaussians.device))
            if not bool(active.any()):
                continue
            scene_gaussians = gaussians[scene:scene+1, active]
            ids = list(map(int, indices[scene].tolist()))
            if len(set(ids)) != self.cfg.gs_render_views or min(ids) < 0 or max(ids) >= self.cfg.dense_views:
                raise ValueError("Invalid render indices")
            if ids != list(range(self.cfg.dense_views)):
                raise ValueError("Render supervision must cover all eight cameras in slot order")
            if sum(bool(batch["observed_mask"][scene, i]) for i in ids) != 2:
                raise ValueError("Full render supervision must include both observed views")
            for view in ids:
                leaf = scene_gaussians.detach().float().requires_grad_(True)
                with fp32_scope(gaussians), torch.enable_grad():
                    rendered = self.gs.render_gaussians(
                        leaf, batch["cam_view_all"][scene:scene+1, view:view+1],
                        intrinsics=batch["intrinsics_all"][scene:scene+1, view:view+1])
                    pred = rendered["images_pred"]
                    alpha = rendered["alphas_pred"].detach().float()
                    alpha_means.append(alpha.mean())
                    alpha_coverages.append((alpha > 1e-4).float().mean())
                    target = batch["images_all"][scene:scene+1, view:view+1].float()
                    rgb = F.mse_loss(pred, target)
                    ssim = (1-fused_ssim(pred.flatten(0, 1), target.flatten(0, 1)))/2
                    loss = (rgb+self.cfg.ssim_weight*ssim)/(b*self.cfg.gs_render_views)
                    view_grad = torch.autograd.grad(loss, leaf)[0]
                view_grad, repaired = repair_renderer_gradient(view_grad)
                self.gs_render_statistics["renderer_gradient_rows_zeroed"] += repaired
                self.gs_render_statistics["rendered_views"] += 1
                image_gradient[scene, active] += view_grad[0].detach()
                rgb_mean += rgb.detach()/(b*self.cfg.gs_render_views)
                ssim_mean += ssim.detach()/(b*self.cfg.gs_render_views)
                group = "observed" if bool(batch["observed_mask"][scene, view]) else "missing"
                group_sum[group] += rgb.detach(); group_count[group] += 1
                self._record(f"gs_render_gradient_{scene}_{view}", view_grad)
            # Direct3D geometry retains gradients for points culled by the renderer.
            if self.cfg.visibility_weight > 0:
                leaf = scene_gaussians[0].detach().float().requires_grad_(True)
                with fp32_scope(gaussians), torch.enable_grad():
                    visibility, geometry = frustum_visibility(leaf[:, :3],
                        batch["cam_view_all"][scene], batch["intrinsics_all"][scene],
                        self.cfg.image_size, self.cfg.image_width,
                        self.gs.opt.znear, self.gs.opt.zfar, self.cfg.visibility_distance_scale)
                    geometry_rows.append(geometry)
                    visibility_grad = torch.autograd.grad(self.cfg.visibility_weight*visibility/b, leaf)[0]
                visibility_grad, repaired = repair_renderer_gradient(visibility_grad)
                self.gs_render_statistics["visibility_gradient_rows_zeroed"] += repaired
                visibility_gradient[scene, active] += visibility_grad.detach()
                visibility_mean += visibility.detach()/b
        gradient = image_gradient + visibility_gradient
        self.gs_render_statistics.update(
            renderer_image_gradient_l2=float(image_gradient.double().norm()),
            visibility_gradient_l2=float(visibility_gradient.double().norm()),
            renderer_alpha_mean=float(torch.stack(alpha_means).mean()) if alpha_means else 0.,
            renderer_alpha_coverage=float(torch.stack(alpha_coverages).mean()) if alpha_coverages else 0.)
        for key in ("geometry_inside_any_frustum_fraction", "geometry_nearest_frustum_distance_mean",
                    "geometry_nearest_frustum_distance_max"):
            values = [row[key] for row in geometry_rows]
            self.gs_render_statistics[key] = sum(values)/len(values) if values else 0.
        value = rgb_mean+self.cfg.ssim_weight*ssim_mean+self.cfg.visibility_weight*visibility_mean
        # Exact first-order renderer VJP, preserving the shared DiT graph without retaining pixels.
        loss = value.detach()+((gaussians.float()-gaussians.detach().float())*gradient.detach()).sum()
        return {"gs": loss, "gs_rgb": rgb_mean, "gs_ssim": ssim_mean,
                "gs_visibility": visibility_mean,
                **{"gs_rgb_"+k: group_sum[k]/max(group_count[k], 1) for k in group_sum}}
