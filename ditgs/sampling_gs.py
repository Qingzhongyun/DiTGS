"""Pure-noise sampler -> VAE decoder features -> one TokenGS reconstruction."""
import torch
from .geometry import flashworld_camera_raymaps
from .gaussian_safety import prepare_gaussians
from .eval_protocol import seed_for

def finite_or_raise(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite inference tensor: "+name)

def make_condition(context, slots):
    if context.ndim != 5 or context.shape[2] != 2 or slots != [0, 7]:
        raise ValueError("Two endpoint context slots are required")
    condition = context.new_zeros(context.shape[0], context.shape[1], 8, *context.shape[-2:])
    condition[:, :, slots] = context
    return condition

class GSSamplingOperationsMixin:
    """Always invoke through root FSDP.forward, including the frozen VAE."""
    def forward(self, operation, **values):
        if operation == "encode_context":
            images = values["images"]
            if images.ndim != 5 or images.shape[:3] != (1, 2, 3):
                raise ValueError("Only the two condition RGB images may enter the VAE encoder")
            return self.vae(images, generator=values["generator"])
        if operation == "velocity":
            return self.dit(values["state"], values["sigma"], values["condition"],
                            values["rays"], self.empty_text)
        if operation == "reconstruct_gs":
            check = values.get("check", finite_or_raise)
            h, w = values["height"], values["width"]
            features = self.vae(values["state"], operation="decode_features")
            check("vae_decoder_features", features)
            images = values["observed_images"]
            mask = values["observed_mask"]
            if images.shape != (1, 2, 3, h, w):
                raise ValueError("Reconstruction requires exactly the original two real RGB images")
            sparse, generated = self.source_gs_memories(
                features, images, values["c2w"], values["intrinsics"], mask)
            patches = (h//8)*(w//8)
            if (sparse.shape != (1, 2*patches, self.cfg.feature_dim) or
                    generated.shape != (1, 8*patches, self.cfg.feature_dim)):
                raise ValueError("Rectangular 2-view / 8-view memory shape mismatch")
            check("gs_real_memory", sparse)
            check("dit_memory", generated)
            head = getattr(self.gs, "activation_head", None)
            if head is not None:
                head.detailed_statistics = True
            gaussians = self.gs.gaussians_from_source_features(
                sparse, generated, capture_stats=True)
            gaussians, active, stats = prepare_gaussians(gaussians, detailed=True)
            if head is not None:
                stats.update(getattr(head, "gaussian_statistics", {}))
                invalid = getattr(head, "local_invalid_mask", None)
                if invalid is not None:
                    active = active & ~invalid
            from .residual_attention import RESIDUAL_STAT_COLUMNS
            stats["residual_rms_columns"] = list(RESIDUAL_STAT_COLUMNS)
            stats["residual_rms"] = self.gs.residual_statistics().cpu().tolist()
            stats["gs_memory_tokens"] = sparse.shape[1]
            stats["dit_memory_tokens"] = generated.shape[1]
            stats["active_points"] = int(active.sum())
            stats["quarantined_points"] = active.numel()-stats["active_points"]
            stats["quarantined_fraction"] = stats["quarantined_points"]/active.numel()
            check("prepared_gaussians", gaussians)
            return gaussians, active, stats
        raise ValueError("Unknown inference operation: "+operation)

@torch.no_grad()
def sample_latents(model, context, c2w, intrinsics, frames, context_slots, *,
                   height, width, nfe, seed, scene_id, scene_scale=.15,
                   normalization_eps=.01, check=finite_or_raise):
    if type(nfe) is not int or not 1 <= nfe <= 256:
        raise ValueError("NFE must be an integer in [1,256]")
    if len(frames) != 8 or frames != sorted(set(frames)):
        raise ValueError("Expected eight ordered unique camera frames")
    if c2w.shape != (1, 8, 4, 4) or intrinsics.shape != (1, 8, 4):
        raise ValueError("Incorrect camera shape")
    b, c, _, h, w = context.shape
    if b != 1 or (height//16, width//16) != (h, w):
        raise ValueError("Context latent raster mismatch")
    condition = make_condition(context, context_slots)
    state = torch.cat([torch.randn((1, c, 1, h, w), dtype=torch.float32,
        device=context.device, generator=torch.Generator(device=context.device).manual_seed(
            seed_for(seed, scene_id, frame, "noise"))) for frame in frames], dim=2)
    # DiT uses the frozen NVS pretrain's OpenGL/max-radius convention. GS rays,
    # source memories and renderer keep their trained TokenGS coordinates.
    rays = flashworld_camera_raymaps(c2w, intrinsics, height, width, h, w,
                                    scene_scale=scene_scale, normalization_eps=normalization_eps)
    check("condition", condition)
    check("rays", rays)
    # Start at sigma=1 for EVERY slot. No target latent and no observed-slot clamp.
    for i in range(nfe):
        sigma = state.new_full((1,), 1-i/nfe)
        velocity = model("velocity", state=state, sigma=sigma, condition=condition, rays=rays)
        state = state-velocity.float()/nfe
        check(f"euler_{i}", torch.stack((state, velocity.float())))
    return state

@torch.no_grad()
def render_scene(renderer, gaussians, active, c2w, intrinsics, targets, background):
    """No model/FSDP calls here: one unchanged Gaussian set renders every target."""
    points = gaussians[:, active[0]].float()
    if points.shape[1] == 0:
        raise ValueError("All Gaussian points invalid; cannot evaluate this scene")
    predictions = []
    for frame in targets:
        view = torch.linalg.inv(c2w[frame:frame+1]).transpose(-1, -2).unsqueeze(0)
        with torch.autocast(device_type=points.device.type, enabled=False):
            result = renderer.render(points, view, bg_color=background,
                                     intrinsics=intrinsics[frame:frame+1].unsqueeze(0))
        image = result["images_pred"][0, 0].detach().float().cpu()
        finite_or_raise("render_frame_"+str(frame), image)
        predictions.append(image)
    return torch.stack(predictions)
