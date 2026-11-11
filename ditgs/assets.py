"""Local pretrained components; inference architectures are packaged as JSON."""
import json
from pathlib import Path

def validate_model_configs(folder):
    folder = Path(folder)
    packaged=Path(__file__).resolve().parent/"assets"
    d = json.loads(((folder/"transformer/config.json") if (folder/"transformer/config.json").is_file() else packaged/"wan_transformer.json").read_text())
    v = json.loads(((folder/"vae/config.json") if (folder/"vae/config.json").is_file() else packaged/"wan_vae.json").read_text())
    expected = {"in_channels": 48, "out_channels": 48, "num_layers": 30,
                "num_attention_heads": 24, "attention_head_dim": 128, "text_dim": 4096}
    if any(d.get(k) != value for k, value in expected.items()) or d.get("patch_size") != [1, 2, 2]:
        raise ValueError("Asset transformer is not standard Wan2.2 TI2V 5B")
    if (v.get("z_dim"), v.get("patch_size"), v.get("scale_factor_spatial")) != (48, 2, 16):
        raise ValueError("Unexpected Wan2.2 VAE")
    if (v.get("decoder_base_dim"), v.get("dim_mult"), v.get("is_residual"),
            v.get("in_channels"), v.get("out_channels")) != (256, [1, 2, 4, 4], True, 12, 12):
        raise ValueError("Expected Wan2.2 residual decoder with 256-channel final features")
    if len(v["latents_mean"]) != 48 or len(v["latents_std"]) != 48 or min(v["latents_std"]) <= 0:
        raise ValueError("Invalid pretrained latent normalization")
    return d, v

def load_empty_prompt(folder):
    """The same masked UMT5 empty prompt, cached after first training startup."""
    import gc,os
    import torch
    folder=Path(folder);cache=folder/"empty_prompt.pt"
    if cache.is_file():
        data=torch.load(cache,map_location="cpu",weights_only=True)
        embedding=data["embedding"] if isinstance(data,dict) else data
    else:
        from transformers import AutoTokenizer,UMT5EncoderModel
        tokenizer=AutoTokenizer.from_pretrained(folder/"tokenizer",local_files_only=True)
        encoder=UMT5EncoderModel.from_pretrained(folder/"text_encoder",torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False)
        tokens=tokenizer([""],padding="max_length",max_length=512,truncation=True,add_special_tokens=True,return_tensors="pt")
        with torch.no_grad():
            embedding=encoder(input_ids=tokens.input_ids,attention_mask=tokens.attention_mask).last_hidden_state.float()
            embedding=embedding*tokens.attention_mask[...,None]
        del encoder,tokenizer;gc.collect()
        temporary=cache.with_suffix(f".{os.getpid()}.tmp");torch.save({"embedding":embedding},temporary);temporary.replace(cache)
    if embedding.shape!=(1,512,4096) or not torch.isfinite(embedding).all():raise ValueError("Invalid empty prompt embedding")
    return embedding
