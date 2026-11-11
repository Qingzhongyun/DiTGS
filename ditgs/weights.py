"""Strict raw-weight loading and full-model export, including nonpersistent buffers."""
import json
from pathlib import Path

def resolve_checkpoint(path):
    path=Path(path).expanduser().resolve()
    if path.is_dir() and (path/"completion.json").is_file():
        return Path(json.loads((path/"completion.json").read_text())["checkpoint"])
    return path

def load_model(path,device,rank,world,model_dir=None):
    import torch
    from dataclasses import replace
    from .config import config_from_dict,config_from_json
    from .inference_runtime import InferenceModel
    from .checkpoint import load_model_weights,_restore_buffers
    path=resolve_checkpoint(path)
    if path.is_file():
        package=torch.load(path,map_location="cpu",weights_only=True)
        if package.get("format") not in ("ditgs-raw-full-model-v1","ditgs-paper-raw-full-model-v1"):
            raise ValueError("Expected a raw full-model package with training_config/model/all_buffers")
        cfg=config_from_dict(package["training_config"])
        if package["model_contract"]!=cfg.model_contract():raise ValueError("Full-model contract mismatch")
        if model_dir:cfg.model_dir=model_dir
        cfg=replace(cfg,activation_checkpointing=False)
        raw=InferenceModel(cfg,initialize_weights=False)
        raw.to_empty(device="cpu")
        raw.load_state_dict(package["model"],strict=True)
        _restore_buffers(raw,package["all_buffers"])
        del package
        raw.to(device).eval().requires_grad_(False)
        return raw,raw,cfg,False
    complete=json.loads((path/"COMPLETE.json").read_text())
    if complete["world_size"]!=world or not complete["sharded"]:
        raise ValueError(f"Sharded checkpoint requires its original{complete['world_size']}processes; use a full model for single-device inference")
    cfg=config_from_json(path.parent.parent/"launch_config.json")
    if model_dir:cfg.model_dir=model_dir
    cfg=replace(cfg,activation_checkpointing=False)
    raw=InferenceModel(cfg,initialize_weights=False)
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    def materialize(module):
        if any(p.is_meta for p in module.parameters(recurse=False)) or any(b.is_meta for b in module.buffers(recurse=False)):
            module.to_empty(device=device,recurse=False)
    wrapped=FSDP(raw,device_id=device,use_orig_params=True,sync_module_states=False,param_init_fn=materialize,
        backward_prefetch=None,forward_prefetch=False,limit_all_gathers=True,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=raw.fsdp_block_types()))
    load_model_weights(wrapped,path,rank=rank,world_size=world,sharded=True,expected_contract=cfg.model_contract())
    return raw,wrapped.eval(),cfg,True

def export_full(raw,model,cfg,path,rank,sharded):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
    if sharded:
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            state=model.state_dict()
    else:state=model.state_dict() if rank==0 else {}
    if rank==0:
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        buffers={name.replace("_fsdp_wrapped_module.",""):value.detach().cpu().clone() for name,value in raw.named_buffers()}
        package={"format":"ditgs-raw-full-model-v1","training_config":cfg.to_dict(),"model_contract":cfg.model_contract(),"model":state,"all_buffers":buffers}
        temporary=target.with_suffix(target.suffix+".writing");torch.save(package,temporary);temporary.replace(target)
