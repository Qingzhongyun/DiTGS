"""Learned decoder patches plus TokenGS's pretrained Pluecker embedding."""
from torch import nn
from tokengs.models.attention import PatchEmbed


class DecoderFeatureAdapter(nn.Module):
    """[B,V,C,H/2,W/2] and RGB-grid rays -> [B,V*(H/8)*(W/8),1024]."""
    def __init__(self, image_size=(256, 256), decoder_channels=256, feature_dim=1024):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.feature_dim = feature_dim
        self.visual_patch = PatchEmbed(
            img_size=tuple(x // 2 for x in image_size), patch_size=4,
            in_chans=decoder_channels, embed_dim=feature_dim, norm_layer=nn.LayerNorm)
        self.ray_patch = PatchEmbed(
            img_size=image_size, patch_size=8, in_chans=6,
            embed_dim=feature_dim, norm_layer=nn.LayerNorm)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, features, plucker):
        if features.ndim != 5 or plucker.ndim != 5:
            raise ValueError("Expected per-view decoder features and RGB-grid Pluecker maps")
        b, views, channels, h, w = features.shape
        if channels != self.decoder_channels or h % 4 or w % 4:
            raise ValueError("Decoder features must have the configured channels and a 4x4 patch grid")
        if plucker.shape != (b, views, 6, h * 2, w * 2):
            raise ValueError("Camera and decoder feature grids must refer to identical views/pixels")
        visual = self.visual_patch(features.flatten(0, 1))
        geometry = self.ray_patch(plucker.flatten(0, 1).to(features))
        memory = self.output_norm(visual + geometry)
        return memory.reshape(b, views * (h // 4) * (w // 4), self.feature_dim)
