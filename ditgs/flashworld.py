"""FlashWorld latent-velocity initialization; no reconstruction-decoder route."""
import json
from pathlib import Path
import torch

def extract(source, expected, allow_feedback=False):
    # expected: state_dict of existing split-native-plus-condition transformer.
    if set(source) != (set(expected) - {
        "patch_embedding.native.weight", "patch_embedding.native.bias",
        "patch_embedding.condition.weight"} | {"patch_embedding.weight", "patch_embedding.bias"}):
        wanted = set(expected) - {"patch_embedding.native.weight", "patch_embedding.native.bias",
            "patch_embedding.condition.weight"} | {"patch_embedding.weight", "patch_embedding.bias"}
        raise ValueError(f"Transformer key mismatch: missing={sorted(wanted-set(source))}, unexpected={sorted(set(source)-wanted)}")
    weight = source["patch_embedding.weight"]
    if weight.ndim != 5 or weight.shape[1] not in (102, 1174):
        raise ValueError(f"Unknown FlashWorld input layout: {tuple(weight.shape)}")
    feedback = weight.shape[1] == 1174
    if feedback and not allow_feedback:
        raise ValueError("Feedback model detected (1174 inputs). This is not a no-feedback pretrain layout. Explicit --allow-feedback-excision required.")
    result = {k: v for k,v in source.items() if not k.startswith("patch_embedding.")}
    result["patch_embedding.native.weight"] = weight[:, :48].contiguous()
    result["patch_embedding.native.bias"] = source["patch_embedding.bias"]
    result["patch_embedding.condition.weight"] = weight[:, 48:102].contiguous()
    for name in ("proj_out.weight", "proj_out.bias"):
        value = source[name]
        if value.shape[0] != 4 * (48 + 1024):
            raise ValueError(f"Unknown FlashWorld output layout: {name}={tuple(value.shape)}")
        # [patch_position, latent+GS_feature, hidden] -> keep each position's latent.
        result[name] = value.reshape(4, 1072, *value.shape[1:])[:, :48].reshape(192, *value.shape[1:]).contiguous()
    mismatches = {k: [list(result[k].shape), list(expected[k].shape)] for k in expected if result[k].shape != expected[k].shape}
    if set(result) != set(expected) or mismatches:
        raise ValueError(f"Converted transformer mismatch: {mismatches}")
    return result, {"input_channels": weight.shape[1], "feedback_excised": feedback,
        "retained_output_channels": 48, "discarded_output_channels": 1024,
        "condition_weight_policy": "retain_pretrained_6_ray_plus_48_image",
        "checkpoint_stage": "must_be_verified_from_provenance_not_inferred_from_shapes"}

def transformer_state(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected checkpoint mapping")
    if "transformer" in payload and isinstance(payload["transformer"], dict):
        return payload["transformer"], sorted(k for k in payload if k != "transformer")
    if "transformer.patch_embedding.weight" in payload:
        unknown = [k for k in payload if not k.startswith(("transformer.", "recon_decoder.")) and k not in ("latents_mean", "latents_std", "timesteps")]
        if unknown: raise ValueError(f"Unknown pretrain top-level keys: {unknown}")
        return {k[len("transformer."):]:v for k,v in payload.items() if k.startswith("transformer.")}, sorted(k for k in payload if not k.startswith("transformer."))
    raise ValueError("Unsupported checkpoint schema; expected explicit transformer namespace")

def asset_receipt(cfg):
    data=json.loads((Path(__file__).resolve().parent/"assets/flashworld_latents.json").read_text())
    return {**data,"source":str(Path(cfg.flashworld_transformer))}

def load_flashworld_transformer(transformer,cfg):
    path=Path(cfg.flashworld_transformer)
    if path.suffix==".safetensors":
        from safetensors.torch import load_file
        state=load_file(str(path),device="cpu")
        if set(state)!=set(transformer.state_dict()):state,_=extract(state,transformer.state_dict())
    else:
        payload=torch.load(path,map_location="cpu",weights_only=True)
        source,_=transformer_state(payload)
        state,_=extract(source,transformer.state_dict())
    transformer.load_state_dict(state,strict=True)
    del state
    return asset_receipt(cfg)

def apply_flashworld_latent_normalization(vae,receipt):
    with torch.no_grad():
        for name in ("mean","std"):
            values=torch.tensor(receipt["latents_"+name],dtype=torch.float32).reshape(1,48,1,1,1)
            if not torch.isfinite(values).all() or (name=="std" and not (values>0).all()):raise ValueError("Invalid latent normalization")
            getattr(vae,name).copy_(values)
