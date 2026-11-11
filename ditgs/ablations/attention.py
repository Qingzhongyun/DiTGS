"""Historical source-fusion definitions on the corrected-camera training engine.

Exactly2 real-view grids and6 missing-view grids; shared native Q/K/V/output.
Single: one softmax. Gated: two softmaxes and per-layer tanh(g),g=0initial.
Gate1: the same two bank outputs added with coefficient exactly1 from training.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tokengs.models.enc_dec import DecoderBlock
from ditgs.tokengs_backbone import ViewBackbone

RESIDUAL_STAT_COLUMNS=("input_rms","ca_delta_rms","after_ca_rms","ca_relative_rms","completion_gate")

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
    # No independent prior projection or second CA exists in these historical
    # architectures. Only learned completion gates belong to the new GS group.
    return name.endswith(".completion_gate")


def missing_view_features(memory, observed_mask):
    """Keep six generated grids; real encoder features replace the other two."""
    observed_indices(observed_mask)
    b,views=observed_mask.shape
    if memory.ndim!=3 or memory.shape[0]!=b or memory.shape[1]%views:
        raise ValueError("Require complete eight-camera adapter feature grids")
    patches=memory.shape[1]//views
    slots=torch.arange(views,device=observed_mask.device).expand_as(observed_mask)
    ids=slots.masked_select(~observed_mask).reshape(b,views-2)
    grid=memory.reshape(b,views,patches,memory.shape[-1])
    return gather_views(grid,ids).reshape(b,(views-2)*patches,memory.shape[-1])


class ObservedFirstCrossAttention(DecoderBlock.CrossAttnBlock):
    """Shared native projections; real/generated banks never share a softmax denominator."""
    def __init__(self, dim, num_heads, qkv_bias, q_norm, dense_views):
        super().__init__(dim, num_heads, qkv_bias, q_norm)
        if dense_views < 3:
            raise ValueError("Source-bank attention needs two real and at least one missing view")
        self.dense_views = dense_views
        # tanh'(0)=1: exact-zero initialization is learnable without a clamp dead zone.
        # FSDP rejects zero-dimensional Parameters. Store one element as [1],
        # then derive a scalar inside forward without changing the gate algebra.
        self.completion_gate = nn.Parameter(torch.zeros(1))
        self.last_gate_value = None  # Diagnostic only, not checkpoint/model state.

    def forward(self, gs_tokens, keys, values):
        if keys.ndim != 4 or keys.shape != values.shape:
            raise ValueError("Expected same-shaped [B,heads,V*P,head_dim] keys/values")
        total = keys.shape[-2]
        if total == 0 or total % self.dense_views:
            raise ValueError("KV memory must retain complete per-camera grids")
        real_length = getattr(self,"observed_views",2) * (total // self.dense_views)
        b, queries, channels = gs_tokens.shape
        projected = self.q_proj(self.gs_token_norm(gs_tokens))
        q = projected.reshape(b, queries, self.num_heads, channels // self.num_heads).transpose(1, 2)
        q = self.q_norm(q)
        real = F.scaled_dot_product_attention(q, keys[:, :, :real_length], values[:, :, :real_length])
        missing = F.scaled_dot_product_attention(q, keys[:, :, real_length:], values[:, :, real_length:])
        # Both banks are nonempty by construction. Always compute both, even at gate0,
        # so the gate has a gradient and every rank keeps the same backward schedule.
        gate = self.completion_gate.tanh().reshape(())
        # Cache a detached scalar INSIDE the full-parameter FSDP forward. Reading
        # the parameter after its block resharded could encounter an empty local shard.
        self.last_gate_value = gate.detach().clone()
        mixed = real + gate.to(real) * missing
        mixed = mixed.transpose(1, 2).reshape(b, queries, channels)
        # Apply the original projection bias ONCE, not once for each memory bank.
        return self.out_proj(mixed)


class ObservedFirstUngatedCrossAttention(DecoderBlock.CrossAttnBlock):
    """Native projections and two independent softmaxes, added without a gate."""
    def __init__(self, dim, num_heads, qkv_bias, q_norm, dense_views):
        super().__init__(dim, num_heads, qkv_bias, q_norm)
        if dense_views < 3:
            raise ValueError("Source-bank attention needs two real and at least one missing view")
        self.dense_views = dense_views

    def forward(self, gs_tokens, keys, values):
        if keys.ndim != 4 or keys.shape != values.shape:
            raise ValueError("Expected same-shaped [B,heads,V*P,head_dim] keys/values")
        total = keys.shape[-2]
        if total == 0 or total % self.dense_views:
            raise ValueError("KV memory must retain complete per-camera grids")
        real_length = getattr(self,"observed_views",2) * (total // self.dense_views)
        b, queries, channels = gs_tokens.shape
        projected = self.q_proj(self.gs_token_norm(gs_tokens))
        q = projected.reshape(b, queries, self.num_heads, channels // self.num_heads).transpose(1, 2)
        q = self.q_norm(q)
        real = F.scaled_dot_product_attention(q, keys[:, :, :real_length], values[:, :, :real_length])
        missing = F.scaled_dot_product_attention(q, keys[:, :, real_length:], values[:, :, real_length:])
        mixed = real + missing
        mixed = mixed.transpose(1, 2).reshape(b, queries, channels)
        # Preserve the native projection, applying its bias exactly once.
        return self.out_proj(mixed)


class SequentialDecoderBlock(DecoderBlock):
    """One native residual CA, then unchanged native SA/MLP."""
    def __init__(self,dim,num_heads,mlp_ratio,qkv_bias=True,ffn_bias=True,
                 qk_norm=True,init_values=None,attn_score_mod=None,attention_variant="single"):
        super().__init__(dim,num_heads,mlp_ratio,qkv_bias,ffn_bias,qk_norm,init_values,attn_score_mod)
        if attention_variant not in ("single","gated","gate1"):
            raise ValueError("Unknown historical source attention variant")
        self.attention_variant=attention_variant
        if attention_variant!="single":
            cls=ObservedFirstCrossAttention if attention_variant=="gated" else ObservedFirstUngatedCrossAttention
            self.gs_cross_attn=cls(dim,num_heads,qkv_bias,qk_norm,8)
        self.last_residual_stats=None

    def forward(self,gs_tokens,gs_keys,gs_values,dit_keys,dit_values,capture_stats=False):
        # Both banks use the SAME query and output projection. The original
        # CrossAttnBlock is used unchanged in Single; no second attention update.
        keys=torch.cat((gs_keys,dit_keys),dim=2)
        values=torch.cat((gs_values,dit_values),dim=2)
        delta=self.gs_cross_attn_scale(self.gs_cross_attn(gs_tokens,keys,values))
        after=gs_tokens+delta
        if capture_stats:
            with torch.no_grad():
                rms=lambda value:value.detach().float().square().mean().sqrt()
                x,d,u=rms(gs_tokens),rms(delta),rms(after)
                gate=(self.gs_cross_attn.last_gate_value if self.attention_variant=="gated"
                      else x.new_tensor(1.0 if self.attention_variant=="gate1" else 0.0))
                if gate is None:
                    raise RuntimeError("Gate diagnostic must be read inside its gathered block")
                self.last_residual_stats=torch.stack((x,d,u,d/x.clamp_min(1e-8),gate))
        else:
            self.last_residual_stats=None
        tokens=after+self.gs_self_attn_scale(self.gs_self_attn(after))
        return tokens+self.mlp_scale(self.mlp(tokens))


class SequentialSourceBackbone(ViewBackbone):
    def __init__(self,opt,activation_checkpointing=True,attention_variant="single"):
        super().__init__(opt,activation_checkpointing)
        self.attention_variant=attention_variant
        core=self.enc_dec_backbone
        native_scale=opt.dec_init_values if opt.dec_init_values is not None else 5e-3*opt.gs_token_std
        # Official tensors overwrite all native constructor samples. Keep module
        # replacement from changing the subsequent global training RNG stream.
        with torch.random.fork_rng(devices=[]):
            core.decoder_blocks=nn.ModuleList([
                SequentialDecoderBlock(opt.enc_embed_dim,opt.enc_num_heads,opt.mlp_ratio,
                    qkv_bias=True,ffn_bias=True,qk_norm=True,init_values=native_scale,
                    attention_variant=attention_variant) for _ in range(opt.dec_depth)])

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

    def source_keys_values(self,gs_features,dit_features):
        if gs_features.ndim!=3 or dit_features.ndim!=3:
            raise ValueError("Require [B,2P,C] real and [B,6P,C] generated memories")
        b,n,c=gs_features.shape
        if n<=0 or n%2 or c!=self.opt.enc_embed_dim or dit_features.shape!=(b,3*n,c):
            raise ValueError("Require exactly2 real and6 missing-view feature grids")
        # A single native KV/keynorm is shared by both sources and all12 layers.
        keys,values=self.enc_dec_backbone._features_to_kv(torch.cat((gs_features,dit_features),dim=1))
        return keys[:,:,:n],values[:,:,:n],keys[:,:,n:],values[:,:,n:]

    def gaussians_from_source_features(self,gs_features,dit_features,*,capture_stats=False,return_logits=False):
        kv=self.source_keys_values(gs_features,dit_features)
        tokens=self.get_gs_tokens(gs_features.shape[0])
        for block in self.enc_dec_backbone.decoder_blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                tokens=checkpoint(block,tokens,*kv,capture_stats=capture_stats,use_reentrant=False)
            else:
                tokens=block(tokens,*kv,capture_stats=capture_stats)
        with torch.autocast(device_type=gs_features.device.type,enabled=False):
            return self.gaussians_from_token_features(tokens.float(),return_logits=return_logits)

    def residual_statistics(self):
        values=[block.last_residual_stats for block in self.enc_dec_backbone.decoder_blocks]
        if any(value is None for value in values):
            raise RuntimeError("Source-CA diagnostics were not captured in decoder forward")
        return torch.stack(values)

    def gaussians_from_image_features(self,*args,**kwargs):
        raise RuntimeError("Use source features from exactly2 real and6 missing views")
