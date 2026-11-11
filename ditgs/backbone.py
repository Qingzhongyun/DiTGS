"""Standard Wan velocity head and frozen per-view VAE decoder feature readout."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class SDPAProcessor:
    """Noncausal PyTorch SDPA with FP32 QK normalization/real RoPE on PPU."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        if attn.add_k_proj is not None or getattr(attn, "fused_projections", False):
            raise ValueError("Unexpected image-added/fused attention in standard TI2V 5B")
        source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(source), attn.to_v(source)
        dtype = q.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            q = attn.norm_q(q.float())
            k = attn.norm_k(k.float())
            q, k = q.unflatten(2, (attn.heads, -1)), k.unflatten(2, (attn.heads, -1))
            if rotary_emb is not None:
                cos, sin = rotary_emb
                cos, sin = cos[..., 0::2].float(), sin[..., 1::2].float()
                def rotate(x):
                    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
                    return torch.stack((a*cos - b*sin, a*sin + b*cos), dim=-1).flatten(-2)
                q, k = rotate(q), rotate(k)
        v = v.unflatten(2, (attn.heads, -1))
        out = F.scaled_dot_product_attention(q.to(dtype).transpose(1, 2), k.to(dtype).transpose(1, 2),
                v.transpose(1, 2), attn_mask=attention_mask, dropout_p=0, is_causal=False)
        out = out.transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](out))

class ConditionedPatchEmbedding(nn.Module):
    """Equivalent to concatenated-input convolution, with separately frozen weights."""
    def __init__(self, native, condition_channels):
        super().__init__()
        self.native = native
        self.latent_channels = native.in_channels
        self.in_channels = native.in_channels + condition_channels
        self.out_channels = native.out_channels
        self.condition = nn.Conv3d(condition_channels, native.out_channels,
            native.kernel_size, stride=native.stride, padding=native.padding, bias=False,
            device=native.weight.device, dtype=native.weight.dtype)
        nn.init.zeros_(self.condition.weight)

    def forward(self, value):
        return self.native(value[:, :self.latent_channels]) + self.condition(value[:, self.latent_channels:])

def extend_transformer(transformer):
    """Add camera/condition inputs while retaining the native velocity head."""
    original_in, channels = transformer.config.in_channels, transformer.config.out_channels
    patch = tuple(transformer.config.patch_size)
    if original_in != channels or patch != (1, 2, 2):
        raise ValueError("Need standard Wan latent input/output and independent temporal patches")
    old = transformer.patch_embedding
    transformer.patch_embedding = ConditionedPatchEmbedding(old, 6 + channels)
    transformer.register_to_config(in_channels=original_in + 6 + channels)
    # Float64 positional buffers can cause unsupported/expensive device kernels.
    transformer.rope.freqs_cos = transformer.rope.freqs_cos.float()
    transformer.rope.freqs_sin = transformer.rope.freqs_sin.float()
    for block in transformer.blocks:
        block.attn1.set_processor(SDPAProcessor())
        block.attn2.set_processor(SDPAProcessor())
    return channels

class MultiViewDiT(nn.Module):
    def __init__(self, transformer, checkpointing=True):
        super().__init__()
        self.latent_channels = extend_transformer(transformer)
        self.transformer = transformer
        if checkpointing:
            transformer.enable_gradient_checkpointing()

    def forward(self, noisy, sigma, condition, raymaps, text):
        if noisy.shape != condition.shape or raymaps.shape != (noisy.shape[0], 6, *noisy.shape[2:]):
            raise ValueError("Each noisy latent/condition/raymap must refer to the same view grid")
        output = self.transformer(
            torch.cat((noisy, raymaps.to(noisy), condition), dim=1),
            timestep=sigma * 1000,
            encoder_hidden_states=text.expand(noisy.shape[0], -1, -1).to(noisy),
            return_dict=False)[0]
        if output.shape != noisy.shape:
            raise ValueError("Native velocity output must preserve the latent view grid")
        return output

def crop_vae_causal_history(vae):
    """Keep current-frame spatial kernels; never crop the DiT view sequence.

    Call AFTER loading the native Wan checkpoint (or constructing on meta).
    The unpadded temporal resampling kernels are left intact: independent T=1
    encoding takes their first-frame bypass. This is not a video-mode VAE.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    cropped = {}
    for name, module in vae.named_modules():
        if not isinstance(module, WanCausalConv3d):
            continue
        padding = tuple(module._padding)
        history = padding[4]
        if history == 0:
            continue  # Includes already-cropped kernels: safe to call twice.
        before = tuple(module.weight.shape)
        if (padding[5] != 0 or before[2] - history != 1
                or module.stride[0] != 1 or module.dilation[0] != 1
                or padding[0] != padding[1] or padding[2] != padding[3]):
            raise ValueError(f"Unsupported causal VAE kernel for T=1 cropping: {name}")
        module.weight = nn.Parameter(
            module.weight[:, :, history:].detach().clone(),
            requires_grad=module.weight.requires_grad)
        # Move spatial padding from the causal wrapper to Conv3d; remove history.
        module.padding = (0, padding[2], padding[0])
        module._padding = (0, 0, 0, 0, 0, 0)
        module.kernel_size = tuple(module.weight.shape[2:])
        cropped[name] = {"before": list(before), "after": list(module.weight.shape),
                         "removed_history_slices": history}
    return cropped


class PerViewVAE(nn.Module):
    """Frozen VAE with no-grad encoding and differentiable, history-free decoding.

    Both operations use forward so FSDP gathers this frozen module's parameters.
    Decoder chunks use non-reentrant checkpointing; recomputation has no mutable
    temporal cache and retains gradients to predicted latents.
    """
    def __init__(self, vae, view_batch=1, checkpointing=True):
        super().__init__()
        self.vae = vae.requires_grad_(False).eval()
        self.view_batch = view_batch
        self.checkpointing = checkpointing
        self.channels = vae.config.z_dim
        self.feature_channels = vae.decoder.conv_out.in_channels
        self.decoder_upscale = 2 ** sum(block.upsampler is not None for block in vae.decoder.up_blocks)
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1))
        # Load native weights first; crop encoder AND decoder current-frame kernels.
        self.temporal_kernel_crops = crop_vae_causal_history(self.vae)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, value, generator=None, operation="encode"):
        if operation == "encode":
            return self._encode(value, generator)
        if operation == "decode_features":
            return self._decode_features(value)
        if operation == "features_to_rgb":
            return self._features_to_rgb(value)
        if operation == "decode_both":
            features = self._decode_features(value)
            return features, self._features_to_rgb(features)
        raise ValueError(f"Unknown per-view VAE operation: {operation}")

    @torch.no_grad()
    def _encode(self, images, generator=None):
        b, views, channels, h, w = images.shape
        if channels != 3:
            raise ValueError("Expected [B,V,3,H,W] RGB")
        flat = images.flatten(0, 1).unsqueeze(2).float() * 2 - 1
        latents = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            for chunk in flat.split(self.view_batch):
                self.vae.clear_cache()
                try:
                    latent = self.vae.encode(chunk).latent_dist.sample(generator=generator)
                finally:
                    self.vae.clear_cache()
                if latent.shape[1:3] != (self.channels, 1):
                    raise ValueError("Independent T=1 encoding must preserve one latent per image")
                latents.append((latent.float() - self.mean.float()) / self.std.float())
        result = torch.cat(latents).squeeze(2)
        return result.reshape(b, views, self.channels, *result.shape[-2:]).transpose(1, 2).contiguous()

    def _decode_chunk(self, normalized):
        # Independent T=1 matches the native first chunk: no time_conv resampling,
        # no history cache, and DupUp3D explicitly keeps just its first frame.
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            x = self.vae.post_quant_conv(normalized.float() * self.std.float() + self.mean.float())
            decoder = self.vae.decoder
            x = decoder.conv_in(x)
            x = decoder.mid_block(x)
            for block in decoder.up_blocks:
                x = block(x, first_chunk=True)
            # Stop BEFORE conv_out, RGB unpatchify and output clamp.
            return decoder.nonlinearity(decoder.norm_out(x))

    def _decode_features(self, clean):
        if clean.ndim != 5 or clean.shape[1] != self.channels:
            raise ValueError("Expected normalized predicted latents [B,C,V,h,w]")
        b, _, views, h, w = clean.shape
        flat = clean.transpose(1, 2).flatten(0, 1).unsqueeze(2)
        features = []
        for chunk in flat.split(self.view_batch):
            if self.checkpointing and torch.is_grad_enabled() and chunk.requires_grad:
                feature = checkpoint(self._decode_chunk, chunk, use_reentrant=False)
            else:
                feature = self._decode_chunk(chunk)
            expected = (self.feature_channels, 1, h * self.decoder_upscale, w * self.decoder_upscale)
            if feature.shape[1:] != expected:
                raise ValueError(f"VAE decoder violated the independent T=1 feature contract: {feature.shape}")
            features.append(feature.squeeze(2))
        result = torch.cat(features)
        return result.reshape(b, views, self.feature_channels, *result.shape[-2:])

    def _features_to_rgb(self, features):
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
        b, views, channels, h, w = features.shape
        flat = features.flatten(0, 1).unsqueeze(2)
        outputs = []
        with torch.autocast(device_type=features.device.type, enabled=False):
            for chunk in flat.split(self.view_batch):
                rgb = self.vae.decoder.conv_out(chunk.float())
                if self.vae.config.patch_size is not None:
                    rgb = unpatchify(rgb, self.vae.config.patch_size)
                outputs.append((rgb.squeeze(2) + 1) / 2)
        rgb = torch.cat(outputs)
        # Raw RGB: do not clamp away the IMG gradient on out-of-range predictions.
        return rgb.reshape(b, views, 3, *rgb.shape[-2:])

def condition_latents(clean, observed):
    if observed.dtype != torch.bool or observed.shape != (clean.shape[0], clean.shape[2]):
        raise ValueError("Condition mask must identify the observed views")
    return torch.where(observed[:, None, :, None, None], clean, torch.zeros_like(clean))

def flow_training_pair(target, noise, sigma):
    t = sigma[:, None, None, None, None]
    return (1-t)*target + t*noise, noise-target
