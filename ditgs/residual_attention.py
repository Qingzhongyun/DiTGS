"""Independent 2-view GS / 8-view DiT memories, with sequential residual CA."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.attention import LayerScale
from ditgs.tokengs_backbone import ViewBackbone

RESIDUAL_STAT_COLUMNS = ("input_rms", "gs_delta_rms", "after_gs_rms",
                         "dit_raw_delta_rms", "dit_delta_rms",
                         "gs_relative_rms", "dit_relative_rms", "after_dit_rms",
                         "dit_gamma_rms", "dit_gamma_abs_max", "dit_scaled_to_raw_rms")


def observed_indices(mask):
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] != 8:
        raise ValueError("Require boolean [B,8] observed camera-slot mask")
    if not bool((mask.sum(1) == 2).all()):
        raise ValueError("Exactly two real camera slots are required")
    slots = torch.arange(8, device=mask.device).expand_as(mask)
    return slots.masked_select(mask).reshape(mask.shape[0], 2)


def gather_views(value, indices):
    if value.ndim < 2 or indices.ndim != 2 or value.shape[0] != indices.shape[0]:
        raise ValueError("View values and per-scene indices must share batch size")
    rows = torch.arange(value.shape[0], device=value.device)[:, None]
    return value[rows, indices.to(device=value.device)]


def is_new_dit_parameter(name):
    return (name.startswith("enc_dec_backbone.dit_kv_proj.")
            or name.startswith("enc_dec_backbone.dit_k_proj_norm.")
            or (name.startswith("enc_dec_backbone.decoder_blocks.")
                and any(part in name for part in (".dit_cross_attn.", ".dit_cross_attn_scale."))))


@torch.no_grad()
def copy_native_source_initialization(core, adapter_output_norm=None):
    """Copy loaded OFFICIAL tensors after to_empty; never tie source parameters."""
    backbone = core.enc_dec_backbone
    backbone.dit_kv_proj.load_state_dict(backbone.kv_proj.state_dict(), strict=True)
    backbone.dit_k_proj_norm.load_state_dict(backbone.k_proj_norm.state_dict(), strict=True)
    for block in backbone.decoder_blocks:
        block.dit_cross_attn.load_state_dict(block.gs_cross_attn.state_dict(), strict=True)
        if not hasattr(block.gs_cross_attn_scale, "gamma"):
            raise ValueError("Official GS cross-attention LayerScale is required for copy initialization")
        # Bare CA output and native channel scale are separate learned parameters.
        # Never fold gamma into W/b here: applying LayerScale would double-scale.
        block.dit_cross_attn_scale.load_state_dict(
            block.gs_cross_attn_scale.state_dict(), strict=True)
    if adapter_output_norm is not None:
        adapter_output_norm.load_state_dict(backbone.encoder_norm.state_dict(), strict=True)


class SequentialDecoderBlock(DecoderBlock):
    """Native GS CA -> independently scaled native DiT CA -> native SA/FFN."""
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=True, ffn_bias=True,
                 qk_norm=True, init_values=None, attn_score_mod=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, ffn_bias,
                         qk_norm, init_values, attn_score_mod)
        self.dit_cross_attn = DecoderBlock.CrossAttnBlock(dim, num_heads, qkv_bias, qk_norm)
        # Always an actual native LayerScale with a1-D, per-channel parameter,
        # not Identity, a scalar gate, or an AdaLN-conditioned modulation.
        self.dit_cross_attn_scale = LayerScale(
            dim, init_values=init_values if init_values is not None else 1e-5)
        # Capture diagnostics inside the full block forward, before FSDP reshard.
        self.last_residual_stats = None

    def forward(self, gs_tokens, gs_keys, gs_values, dit_keys, dit_values,
                capture_stats=False):
        gs_delta = self.gs_cross_attn_scale(
            self.gs_cross_attn(gs_tokens, gs_keys, gs_values))
        after_gs = gs_tokens + gs_delta
        # CrossAttnBlock retains query Pre-Norm and per-head Q norm.
        dit_raw_delta = self.dit_cross_attn(after_gs, dit_keys, dit_values)
        dit_delta = self.dit_cross_attn_scale(dit_raw_delta)
        after_dit = after_gs + dit_delta
        if capture_stats:
            # Observability only; never modify/clamp the learned residual scale.
            with torch.no_grad():
                def rms(value):
                    return value.detach().float().square().mean().sqrt()
                x, g, u, d = (rms(v) for v in (gs_tokens, gs_delta, after_gs, dit_delta))
                raw = rms(dit_raw_delta)
                gamma = self.dit_cross_attn_scale.gamma.detach().float()
                self.last_residual_stats = torch.stack((x, g, u, raw, d,
                    g / x.clamp_min(1e-8), d / u.clamp_min(1e-8), rms(after_dit),
                    rms(gamma), gamma.abs().amax(), d / raw.clamp_min(1e-8)))
        else:
            self.last_residual_stats = None
        tokens = after_dit + self.gs_self_attn_scale(self.gs_self_attn(after_dit))
        return tokens + self.mlp_scale(self.mlp(tokens))


class SequentialSourceBackbone(ViewBackbone):
    def __init__(self, opt, activation_checkpointing=True):
        super().__init__(opt, activation_checkpointing)
        core = self.enc_dec_backbone
        # Original KV_G / key norm retain their native state-dict names.
        core.dit_kv_proj = nn.Linear(opt.enc_embed_dim, 2 * opt.enc_embed_dim, bias=True)
        core.dit_k_proj_norm = nn.LayerNorm(opt.enc_embed_dim // opt.enc_num_heads)
        native_scale = opt.dec_init_values if opt.dec_init_values is not None else 5e-3 * opt.gs_token_std
        core.decoder_blocks = nn.ModuleList([
            SequentialDecoderBlock(opt.enc_embed_dim, opt.enc_num_heads, opt.mlp_ratio,
                qkv_bias=True, ffn_bias=True, qk_norm=True, init_values=native_scale)
            for _ in range(opt.dec_depth)])

    @torch.no_grad()
    def sparse_image_features(self, images, plucker, mask):
        ids = observed_indices(mask)
        b, views = mask.shape
        if plucker.ndim != 5 or plucker.shape[:3] != (b, views, 6):
            raise ValueError("Full ray grids must retain all eight physical camera slots")
        if images.ndim != 5 or images.shape[0] != b or images.shape[2] != 3:
            raise ValueError("Real RGB must be [B,2 or 8,3,H,W]")
        if images.shape[-2:] != plucker.shape[-2:]:
            raise ValueError("Real RGB and camera rays must have the same crop/raster")
        if images.shape[1] == views:
            real_rgb = gather_views(images, ids)
        elif images.shape[1] == 2:
            # Compact inference RGB uses ascending observed-slot order.
            real_rgb = images
        else:
            raise ValueError("Unsupported real RGB view count")
        real_rays = gather_views(plucker, ids)
        mean = real_rgb.new_tensor((0.485, 0.456, 0.406)).reshape(1, 1, 3, 1, 1)
        std = real_rgb.new_tensor((0.229, 0.224, 0.225)).reshape(1, 1, 3, 1, 1)
        embedded = self.patch_embed(((real_rgb-mean)/std).flatten(0, 1))
        embedded = embedded + self.patch_plucker_embed(real_rays.flatten(0, 1).to(embedded))
        joint = embedded.reshape(b, 2 * embedded.shape[1], embedded.shape[2])
        return self.enc_dec_backbone._encode_features(joint)

    def source_keys_values(self, gs_features, dit_features):
        if gs_features.ndim != 3 or dit_features.ndim != 3:
            raise ValueError("Require independent [B,2P,C] and [B,8P,C] source memories")
        b, real_tokens, channels = gs_features.shape
        if real_tokens <= 0 or real_tokens % 2 or channels != self.opt.enc_embed_dim:
            raise ValueError("Invalid two-view GS memory")
        if dit_features.shape != (b, 4 * real_tokens, channels):
            raise ValueError("DiT memory must keep all eight views on the same per-view grid")
        core = self.enc_dec_backbone
        gs_keys, gs_values = core._features_to_kv(gs_features)
        dit_keys, dit_values = rearrange(core.dit_kv_proj(dit_features),
            "b n (kv h c) -> kv b h n c", kv=2, h=self.opt.enc_num_heads)
        dit_keys = core.dit_k_proj_norm(dit_keys)
        return gs_keys, gs_values, dit_keys, dit_values

    def gaussians_from_source_features(self, gs_features, dit_features,
                                      *, capture_stats=False, return_logits=False):
        # Two projections exactly once, OUTSIDE the 12-layer decoder loop.
        kv = self.source_keys_values(gs_features, dit_features)
        tokens = self.get_gs_tokens(gs_features.shape[0])
        for block in self.enc_dec_backbone.decoder_blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, *kv, capture_stats=capture_stats,
                                    use_reentrant=False)
            else:
                tokens = block(tokens, *kv, capture_stats=capture_stats)
        with torch.autocast(device_type=gs_features.device.type, enabled=False):
            return self.gaussians_from_token_features(tokens.float(), return_logits=return_logits)

    def residual_statistics(self):
        # Caches were captured inside complete block forwards, before FSDP reshard.
        values = [block.last_residual_stats for block in self.enc_dec_backbone.decoder_blocks]
        if any(value is None for value in values):
            raise RuntimeError("Residual statistics were not requested in the decoder forward")
        return torch.stack(values)

    def gaussians_from_image_features(self, *args, **kwargs):
        raise RuntimeError("Sequential CA requires TWO separate memories; use gaussians_from_source_features")
