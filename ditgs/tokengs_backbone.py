"""Parameter-preserving interface to the native TokenGS decoder."""
import torch
from torch.utils.checkpoint import checkpoint
from tokengs.models.input_types import EncoderLatent
from tokengs.models.tokengs import TokenGS as OfficialTokenGS
from tokengs.options import Options
OFFICIAL_COMMIT="b16269c500a8894cda342bc9cf406e31169541e3"
OFFICIAL_REPOSITORY="https://github.com/nv-tlabs/TokenGS.git"

class TokenGSFeatures(OfficialTokenGS):
    def __init__(self, options, activation_checkpointing=True):
        if options.time_embedding or options.num_dynamic_gs_tokens:
            raise ValueError("DiTGS uses the official static base model")
        if options.use_multiscale_encoder or options.use_latent_bottleneck:
            raise ValueError("This adapter uses the official base, not the separate release bottleneck preset")
        if options.token_dim != options.enc_embed_dim:
            raise ValueError("Gaussian token and encoder widths must match")
        super().__init__(options)
        self.activation_checkpointing = activation_checkpointing

    def forward_encoder(self, encoder_input):
        if not self.activation_checkpointing or not torch.is_grad_enabled():
            return super().forward_encoder(encoder_input)
        tokens = self._embed_encoder_input(encoder_input)
        for block in self.enc_dec_backbone.encoder:
            tokens = checkpoint(block, tokens, use_reentrant=False)
        features = self.enc_dec_backbone.encoder_norm(tokens)
        keys, values = self.enc_dec_backbone._features_to_kv(features)
        return EncoderLatent(keys=keys, values=values)

    def forward_token_features(self, encoder_latent, gs_tokens=None):
        tokens = self.get_gs_tokens(encoder_latent.keys.shape[0]) if gs_tokens is None else gs_tokens
        for block in self.enc_dec_backbone.decoder_blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, encoder_latent.keys,
                                    encoder_latent.values, use_reentrant=False)
            else:
                tokens = block(tokens, encoder_latent.keys, encoder_latent.values)
        return tokens

    def gaussians_from_token_features(self, features, return_logits=False):
        if features.ndim != 3 or features.shape[1:] != (self.opt.num_gs_tokens, self.opt.enc_embed_dim):
            raise ValueError(f"Invalid Gaussian-token feature shape: {tuple(features.shape)}")
        result = (self.activation_head(features, return_logits=True) if return_logits
                  else self.activation_head(features))
        gaussians, logits = result if return_logits else (result, None)
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset
        return (gaussians, logits) if return_logits else gaussians

    def checkpoint_metadata(self):
        return {
            "architecture": "official_flat_encdec",
            "upstream_commit": OFFICIAL_COMMIT,
            "enc_depth": self.opt.enc_depth,
            "dec_depth": self.opt.dec_depth,
            "feature_dim": self.opt.enc_embed_dim,
            "heads": self.opt.enc_num_heads,
            "num_gs_tokens": self.opt.num_gs_tokens,
            "gaussians_per_token": self.opt.dec_patch_size ** 2,
        }

class ViewBackbone(TokenGSFeatures):
    def image_features(self,encoder):
        core=self.enc_dec_backbone
        return core.encoder_norm(core.encoder(self._embed_encoder_input(encoder)))
    def gaussians_from_image_features(self,features,return_logits=False):
        keys,values=self.enc_dec_backbone._features_to_kv(features)
        queries=self.forward_token_features(EncoderLatent(keys=keys,values=values))
        with torch.autocast(device_type=features.device.type,enabled=False):
            return self.gaussians_from_token_features(queries.float(),return_logits=return_logits)
